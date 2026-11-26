"""HEFT where it was released: HITCH's MuJoCo robot, HITCH's planner, the G1_Harness checkpoint (port check 0).

Runs in HITCH's own Python environment (``HITCH/.venv``, Python 3.10, the ``wbc`` package installed; ``linuxfd``
is not needed here), with this repository on ``sys.path``:

    HITCH_ROOT=<HITCH checkout> <HITCH>/.venv/bin/python heft_mujoco.py --out <dir> [--hand rubber|dex3]

``HITCH_ROOT/third_party/motion_tracking`` must be the controller branch ``sim-to-real-harness`` (``scripts/setup``
clones it). For every motion of the experiment 2 command set, planned the way the harness plans it
(``wbc.planner.request.plan`` from the reference tail, frame O at the tail with the measured heading), the script

1. steps the HITCH MuJoCo robot (``wbc_link_server.scenes`` ``flat``, 5 ms physics, torque-clipped joint PD, 4
   substeps per 50 Hz control step, as ``bridges/sim_bridge.py``) with ``kimodo_vla.controllers.heft``;
2. builds the release's own observation (``runtime.policy.TrackingPolicyRaw``) from the same reference and state at
   every step and records the largest difference from ours (must be ~1e-6: float32);
3. measures the HITCH paper's tracking errors (appendix D): walk error = |executed - planned pelvis displacement| /
   planned distance (walks >= 0.2 m); hand error = wrist distance from its reference in the pelvis frame, averaged
   over frames where the reference wrist is >= 5 cm from its start in the reference torso frame.

Writes ``<out>/heft-mujoco-<hand>.json`` (one row per motion).
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[4]))
from kimodo_vla.controllers.heft import HeftTracker, Reference, quat_to_matrix  # noqa: E402

# The experiment 2 command set (README protocol), as HITCH requests in frame O. The reach is the right palm 0.35 m
# ahead, 0.20 m right, 0.95 m high, palm forward, 1.5 s; HITCH places the palm point, not the wrist.
REQUESTS = {
    'forward': {'walk': {'target': [0.5, 0.0]}},
    'sidestep': {'walk': {'target': [0.0, 0.5]}},
    'backward': {'walk': {'target': [-0.5, 0.0]}},
    'turn': {'walk': {'target': [0.0, 0.0], 'turn_deg': 90}},
    'squat': {'height': 0.60},
    'lean': {'torso': {'pitch_deg': 30, 'yaw_deg': 0}},
    'reach': {'hands': {'right': {'waypoints': [{'position': [0.35, -0.20, 0.95], 'duration_s': 1.5,
                                                 'orientation': {'palm': 'forward'}}]}}},
}
SETTLE_S, TAIL_S = 1.0, 2.0


def release_policy(hitch, bundle, initial_wire, initial_root, channel_dir):
    """The release's TrackingPolicyRaw, configured as HITCH's launch/policy.py writes it."""
    import yaml
    from common.utils import DictToClass
    from motion_channel import MotionChannel
    from runtime.policy import TrackingPolicyRaw
    from wbc.core.layout import WIRE_JOINTS
    spec = yaml.safe_load((bundle/'deploy.yaml').read_text())
    obs = dict(spec['observation'])
    target, compliance = obs.pop('target_joint'), obs.pop('compliance')
    MotionChannel(channel_dir, create=True)
    cfg = DictToClass({**obs, 'target_joint_include_diff': target['layout'] == 'full',
                       'use_compliance_flag_obs': compliance['enabled'], 'compliance_flag_value': compliance['value'],
                       'compliance_flag_threshold': compliance['threshold'], 'policy_path': str(bundle/spec['policy']),
                       'action_scale': spec['action']['scale'], 'action_clip': spec['action']['clip'],
                       'action_joint_names': list(WIRE_JOINTS), 'dataset_joint_names': list(WIRE_JOINTS),
                       'transition_steps': 0, 'switch_tail_keep_steps': 8,
                       'motion_source': {'type': 'external', 'external': {'directory': str(channel_dir)}},
                       'motions': [], 'motion_clips': [{'name': 'default', 'joint_pos': list(map(float, initial_wire)),
                                                        'root_pos': list(map(float, initial_root[:3])),
                                                        'root_quat': list(map(float, initial_root[3:7]))}]})
    cfg._config_dir = str(bundle)

    class Controller:
        pass
    ctrl = Controller()
    ctrl.config = DictToClass({'policy_joint_names': list(WIRE_JOINTS)})
    ctrl.dof_size = 29
    ctrl.qj, ctrl.dqj = np.zeros(29, np.float32), np.zeros(29, np.float32)
    ctrl.quat, ctrl.gyro = np.array([1, 0, 0, 0], np.float32), np.zeros(3, np.float32)
    policy = TrackingPolicyRaw('tracking', cfg, ctrl)
    policy.fade_in()
    return policy, ctrl


def run_motion(name, request, hand, bundle):
    import mujoco
    from wbc.core.frames import SE2, heading
    from wbc.core.layout import CANONICAL_TO_WIRE, WIRE_TO_CANONICAL
    from wbc.core.request import parse_request
    from wbc.core.types import StartState, posture_for_height
    from wbc.planner.request import plan
    from wbc.paths import INITIAL_REFERENCE
    from wbc.robot.hands import current_hand
    from wbc.robot.model import body_joint_addresses, hold_fingers
    from wbc_link_server import scenes
    from wbc_link_server.bridges.sim_bridge import initialize_scene

    spec = scenes.scene('flat', 0)
    model = scenes.build_scene(spec)
    data = mujoco.MjData(model)
    initial = current_hand().rest(np.load(INITIAL_REFERENCE)['qpos'][0])
    initialize_scene(model, data, spec, initial)
    hold_fingers(model, data)
    qadr, vadr = body_joint_addresses(model)
    lo, hi = model.actuator_ctrlrange[:29, 0], model.actuator_ctrlrange[:29, 1]
    substeps = round(0.02/model.opt.timestep)
    fk = mujoco.MjData(model)
    body = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ('pelvis', 'torso_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link')}

    tracker = HeftTracker(bundle)
    to_wire = list(CANONICAL_TO_WIRE)
    policy, ctrl = release_policy(None, bundle, initial[7:][to_wire], initial, Path(tempfile.mkdtemp())/'channel')
    ref = Reference(policy.ref_root_pos, policy.ref_root_quat, policy.ref_joint_pos)
    ref.cursor = policy.ref_idx
    horizon = tracker.spec.horizon

    def append(frames36):
        remaining = len(ref)-1-ref.cursor
        tail = np.r_[ref.root_pos[-1], ref.root_quat[-1], ref.joints[-1]]
        frames = np.vstack([np.repeat(tail[None], max(0, horizon+1-remaining), 0), frames36])
        ref.append(frames[:, :3], frames[:, 3:7], frames[:, 7:])
        policy.append_ref_frames({'root_pos': frames[:, :3], 'root_quat': frames[:, 3:7], 'joint_pos': frames[:, 7:]})

    def measured():
        q = data.qpos[qadr][to_wire]
        dq = data.qvel[vadr][to_wire]
        return q, dq, data.qpos[3:7].copy(), data.qvel[3:6].copy()

    log, worst, worst_action = [], 0.0, [0.0]

    def control(steps):
        nonlocal worst
        for _ in range(steps):
            q, dq, quat, gyro = measured()
            ctrl.qj[:], ctrl.dqj[:], ctrl.quat[:], ctrl.gyro[:] = q, dq, quat, gyro
            policy.update_obs()
            target = tracker.step(ref, q, dq, quat, gyro)
            if ref.cursor != policy.ref_idx:
                raise AssertionError('reference cursors diverged')
            worst = max(worst, float(np.abs(policy.policy_input['policy'][0]-tracker.last_obs).max()))
            release_target = tracker.spec.default_joint_pos + policy.compute_action()   # wire order, as deploy.py
            worst_action[0] = max(worst_action[0], float(np.abs(release_target-target).max()))
            target_c = target[list(WIRE_TO_CANONICAL)]
            kp, kd = tracker.spec.kps[list(WIRE_TO_CANONICAL)], tracker.spec.kds[list(WIRE_TO_CANONICAL)]
            for _ in range(substeps):
                tau = kp*(target_c-data.qpos[qadr]) - kd*data.qvel[vadr]
                data.ctrl[:29] = np.clip(tau, lo, hi)
                mujoco.mj_step(model, data)
            r = ref.cursor
            log.append({'qpos': np.r_[data.qpos[:7], data.qpos[qadr]],
                        'ref': np.r_[ref.root_pos[r], ref.root_quat[r], ref.joints[r][list(WIRE_TO_CANONICAL)]]})

    control(int(SETTLE_S*50))
    tail = np.r_[ref.root_pos[-1], ref.root_quat[-1], ref.joints[-1][list(WIRE_TO_CANONICAL)]]
    to_world = SE2(float(tail[0]), float(tail[1]), heading(data.qpos[3:7]))
    start = StartState(to_world.inverse().apply_poses(tail), posture_for_height(float(tail[2])))
    planned = plan(parse_request(request), start=start, frame=to_world, horizon=horizon)
    if not planned.report.passed:
        raise RuntimeError(f'{name}: HITCH checks rejected the plan')
    world = planned.world.copy()
    world[:, 7:] = world[:, 7:][:, to_wire]
    t0 = len(log)
    append(world[1:])
    control(len(world)+int(TAIL_S*50))

    def bodies(pose):
        fk.qpos[:7] = pose[:7]
        fk.qpos[qadr] = pose[7:]
        mujoco.mj_kinematics(model, fk)
        return {n: (fk.xpos[i].copy(), fk.xmat[i].reshape(3, 3).copy()) for n, i in body.items()}

    actual = [bodies(x['qpos']) for x in log[t0:]]
    reference = [bodies(x['ref']) for x in log[t0:]]
    out = {'motion': name, 'request': request, 'hand': hand, 'frames': len(world),
           'max_obs_diff_vs_release': worst, 'max_target_diff_vs_release_rad': worst_action[0],
           'fell': bool(min(x['qpos'][2] for x in log) < 0.4)}
    # walk error: pelvis displacement over the step, each from its own start pose
    da = actual[-1]['pelvis'][0][:2]-actual[0]['pelvis'][0][:2]
    dr = reference[-1]['pelvis'][0][:2]-reference[0]['pelvis'][0][:2]
    out['planned_distance_m'] = float(np.linalg.norm(dr))
    out['executed_displacement_m'] = da.tolist()
    if np.linalg.norm(dr) >= 0.2:
        out['walk_error'] = float(np.linalg.norm(da-dr)/np.linalg.norm(dr))
    yaw = lambda m: np.arctan2(m[1, 0], m[0, 0])
    out['yaw_executed_deg'] = float(np.degrees(np.angle(np.exp(1j*(yaw(actual[-1]['pelvis'][1])-yaw(actual[0]['pelvis'][1]))))))
    out['yaw_planned_deg'] = float(np.degrees(np.angle(np.exp(1j*(yaw(reference[-1]['pelvis'][1])-yaw(reference[0]['pelvis'][1]))))))
    out['pelvis_height_end_m'] = [float(actual[-1]['pelvis'][0][2]), float(reference[-1]['pelvis'][0][2])]
    # hand error (HITCH appendix D)
    for side in ('left', 'right'):
        w = f'{side}_wrist_yaw_link'
        in_pelvis = lambda b: b['pelvis'][1].T@(b[w][0]-b['pelvis'][0])
        in_torso = lambda b: b['torso_link'][1].T@(b[w][0]-b['torso_link'][0])
        moved = [np.linalg.norm(in_torso(r)-in_torso(reference[0])) >= 0.05 for r in reference]
        errs = [np.linalg.norm(in_pelvis(a)-in_pelvis(r)) for a, r in zip(actual, reference)]
        if any(moved):
            out[f'hand_error_{side}_m'] = float(np.mean([e for e, m in zip(errs, moved) if m]))
        out[f'hand_end_world_m_{side}'] = float(np.linalg.norm(actual[-1][w][0]-reference[-1][w][0]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--hand', choices=['rubber', 'dex3'], default='rubber')
    p.add_argument('--only', nargs='+', choices=list(REQUESTS))
    a = p.parse_args()
    os.environ['WBC_HAND'] = a.hand
    hitch = Path(os.environ['HITCH_ROOT'])
    if 'linuxfd' not in sys.modules:   # only the controller's wall-clock Timer uses it; nothing here does
        import types
        sys.modules['linuxfd'] = types.ModuleType('linuxfd')
    sys.path.insert(0,str(hitch/'third_party/motion_tracking/sim2real/src'))
    bundle = Path(os.environ.get('KVLA_HEFT_CHECKPOINT', hitch/'third_party/motion_tracking/sim2real/config/g1/ckpts/G1_Harness'))
    rows = []
    for name in a.only or REQUESTS:
        row = run_motion(name, REQUESTS[name], a.hand, bundle)
        print(json.dumps(row), flush=True)
        rows.append(row)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/f'heft-mujoco-{a.hand}.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
