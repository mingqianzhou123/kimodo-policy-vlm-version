"""HEFT as the tracker of an Arena run (``KVLA_TRACKER=heft``): process-local, the shared simulator stays read-only.

Arena's SONIC provider decodes each 40-D action into one world reference frame (root position, root orientation,
29 joints in its Isaac Lab order) and hands it to SONIC. Here the same decoded frames are handed to HEFT
(``kimodo_vla.controllers.heft``) instead; everything else in the provider (decoding, hands, recording, the explicit
PD loop at every physics step) is reused.

Look-ahead: HEFT reads reference frames up to 8 steps ahead. The HTTP planner sends 25-frame chunks and plans each
from the last frame consumed, so the future cannot be fetched early without breaking its continuity. HEFT therefore
tracks the frame decoded ``LAG`` steps ago and reads the newer frames as its future, as the HITCH harness does when
it holds the reference tail before appending a segment. The newest decoded frame stays the provider's
``_latest_pose_payload`` (the planner's continuity point); the frame HEFT tracks is ``_heft_reference``.

Drives: explicit PD with the checkpoint's gains at every physics step, torques clipped to the effort limits of
HEFT's own MuJoCo model (``configs/heft-dynamics.json``, applied with the armatures by the ``heft`` dynamics profile).

At creation, ``heft-port.json`` in the run folder records the port checks: joint order, gains, effort limits,
control timing, the root body, and the masses of the hand links. Every control step adds the gyro and quaternion
convention checks to running maxima written at the end of the episode.
"""
import importlib
from collections import deque
import json
import os
from pathlib import Path

import numpy as np

from kimodo_vla.controllers.heft import CHECKPOINT_ENV, HEFT_JOINT_NAMES, HeftTracker, Reference, quat_to_matrix

ROOT = Path(__file__).resolve().parents[1]
LAG = 8   # = the checkpoint's largest future step; checked against it below


def _array(t):
    return t.detach().cpu().numpy().astype(np.float64)


def attach(provider, env, run):
    """Replace SONIC by HEFT inside one SONIC provider."""
    import torch
    tracker = HeftTracker(os.environ[CHECKPOINT_ENV], threads=2)
    spec = tracker.spec
    if spec.horizon != LAG or spec.control_hz != 50:
        raise ValueError(f'checkpoint horizon {spec.horizon} / rate {spec.control_hz} differ from the adapter')
    robot = env.scene['robot']
    joint_names = list(robot.joint_names)
    ids = provider._sonic_idx.detach().cpu().numpy().tolist()
    provider_order = [joint_names[i] for i in ids]
    if tuple(provider_order) != spec.joint_names or spec.joint_names != HEFT_JOINT_NAMES:
        raise ValueError('joint order: Arena provider order differs from the checkpoint order')
    limits = json.loads((ROOT/'configs/heft-dynamics.json').read_text())['actuator']['dof_max_forces']
    effort = np.array([limits[n] for n in spec.joint_names])
    device = provider.device
    provider._pd_kp_t = torch.tensor(spec.kps, dtype=torch.float32, device=device)
    provider._pd_kd_t = torch.tensor(spec.kds, dtype=torch.float32, device=device)
    provider._pd_effort_limit_t = torch.tensor(effort, dtype=torch.float32, device=device)
    provider._use_effort_control = True
    provider._effort_mode_runtime_configured = False
    if int(getattr(provider, '_sonic_output_delay_steps', 0)) != 0:
        raise ValueError('SONIC output delay must be 0 for HEFT')
    decimation = int(provider._decimation)
    body_names = list(robot.body_names)
    masses = _array(robot.root_physx_view.get_masses())[0]
    inertias = _array(robot.root_physx_view.get_inertias())[0]
    hand = {n: {'mass_kg': float(masses[i]), 'inertia_diag': inertias[i].reshape(3, 3).diagonal().tolist()
                if inertias[i].size == 9 else inertias[i].tolist()}
            for i, n in enumerate(body_names) if 'hand' in n or 'wrist' in n}
    port = {'tracker': 'heft', 'checkpoint': str(spec.policy_path.parent.name), 'lag_steps': LAG,
            'joint_order_matches': True, 'joint_order': list(provider_order),
            'kps': spec.kps.tolist(), 'kds': spec.kds.tolist(), 'effort_limit': effort.tolist(),
            'action_scale': spec.action_scale.tolist(), 'action_clip': spec.action_clip,
            'default_joint_pos': spec.default_joint_pos.tolist(),
            'physics_dt': float(env.physics_dt), 'decimation': decimation,
            'control_dt': float(env.physics_dt*decimation), 'root_body': body_names[0],
            'total_mass_kg': float(masses.sum()), 'hand_and_wrist_links': hand}
    if not np.isclose(port['control_dt'], 1/spec.control_hz):
        raise ValueError(f"control step {port['control_dt']} s, checkpoint runs at {spec.control_hz} Hz")
    (run/'heft-port.json').write_text(json.dumps(port, indent=1))

    # the newest decoded reference frames: (root pos, root quat wxyz, joints in checkpoint order); enough for the
    # tracked frame (LAG back), its history (-4) and its future (the newest)
    frames = deque(maxlen=LAG+1+spec.history)
    checks = {'max_gyro_frame_diff': 0., 'max_quat_vs_pelvis_body_diff': 0., 'steps': 0}
    pelvis = body_names.index('pelvis')

    def from_vla():
        action = provider._infer_lerobot_semantic_action()
        provider._apply_lerobot_semantic_action(action)
        payload = provider._raw_pose_payload
        frames.append((np.asarray(payload['body_pos'], float).copy(), np.asarray(payload['body_quat_w'], float).copy(),
                       np.asarray(payload['joint_pos'], float).copy()))
        ref = Reference(*(np.array([f[k] for f in frames]) for k in range(3)))
        ref.cursor = max(0, len(frames)-1-LAG)
        data = robot.data
        q = _array(data.joint_pos[0, provider._sonic_idx])
        dq = _array(data.joint_vel[0, provider._sonic_idx])
        quat = _array(data.root_state_w[0, 3:7])
        gyro = _array(data.root_ang_vel_b[0])
        # conventions: body-frame gyro = R^T world angular velocity; root quaternion = the pelvis body's
        world = _array(data.root_state_w[0, 10:13])
        checks['max_gyro_frame_diff'] = max(checks['max_gyro_frame_diff'],
                                            float(np.abs(quat_to_matrix(quat).T@world-gyro).max()))
        body_q = _array(data.body_quat_w[0, pelvis])
        checks['max_quat_vs_pelvis_body_diff'] = max(checks['max_quat_vs_pelvis_body_diff'],
                                                     float(min(np.abs(body_q-quat).max(), np.abs(body_q+quat).max())))
        checks['steps'] += 1
        target = tracker.act(tracker.observe(ref, q, dq, quat, gyro)).astype(np.float32)
        r = ref.cursor
        provider._heft_reference = {'body_pos': ref.root_pos[r].tolist(), 'body_quat_w': ref.root_quat[r].tolist(),
                                    'joint_pos': ref.joints[r].tolist()}
        provider._latest_decoder_target = target.copy()
        if checks['steps'] % 250 == 0:
            (run/'heft-checks.json').write_text(json.dumps(checks))
        return target

    def reset_frames():
        frames.clear()   # the tracker's own histories stay warm, as SONIC's do

    provider._run_gear_sonic_from_vla = from_vla
    provider._heft_reset = reset_frames
    provider._heft_checks = checks
    return provider


def install_heft():
    """Wrap Arena's provider factory so every provider it creates runs HEFT."""
    factory = importlib.import_module('action_provider.create_action_provider')
    original = factory.create_action_provider

    def create(env, args):
        provider = original(env, args)
        if not hasattr(provider, '_sonic_idx') or provider._lerobot_vla_runtime is None:
            raise ValueError('HEFT runs inside the SONIC VLA provider')
        return attach(provider, env, Path(os.environ['ORACLE_RUN']))
    factory.create_action_provider = create
