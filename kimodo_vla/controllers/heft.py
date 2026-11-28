"""HEFT, the whole-body tracker of the HITCH harness (checkpoint ``G1_Harness``), as a simulator-independent step.

Source of truth: ``github.com/Axellwppr/motion_tracking``, branch ``sim-to-real-harness``,
``sim2real/src/runtime/{policy,observation}.py`` and ``G1_Harness/deploy.yaml``; the HITCH harness writes the
tracking configuration from that file (``HITCH/src/wbc_link_server/launch/policy.py``: ``target_joint.layout:
target_only`` -> no target-minus-current half, ``compliance.enabled: false`` -> no compliance flag). Read 2026-11-27.

What the policy consumes (nothing else): a 50 Hz reference of root position (world, m), root orientation (world,
wxyz) and 29 joint angles in the checkpoint's joint order; the measured joint angles and velocities, the pelvis
orientation (world, wxyz) and the pelvis angular velocity in the pelvis frame. It does NOT read ``local_body_pos``
or any body positions: those are fields of the training motion files only.

Observation (1184 values at the default layout), in this order:
  boot indicator (1); reference root position offsets of the future frames from the current one, in the current
  reference root frame (3 per non-zero offset) and the 6D rotation of current-robot^-1 * reference for every offset
  (the first column, then the second: not the row-major layout of Arena's 40-D action); target joints (29 per offset); target root height (1 per offset); gravity in each
  target root frame (3 per offset); angular velocity history; projected gravity history; joint position history;
  joint velocity history (absolute values, not relative to the default pose); previous raw actions (8 x 29).
Action: target = default_joint_pos + scale * clip(raw, -clip, clip), tracked by joint PD with kps / kds.

Histories start at zero and the boot indicator at its maximum at ``reset``; the reference cursor advances before
the observation is built, as in ``TrackingPolicyRaw.update_obs``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CHECKPOINT_ENV = 'KVLA_HEFT_CHECKPOINT'   # directory holding deploy.yaml and policy.onnx (G1_Harness)
# G1_Harness/deploy.yaml action.joint_names (Isaac Lab breadth-first order; HITCH calls it the wire order)
HEFT_JOINT_NAMES = (
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint',
    'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint', 'left_knee_joint',
    'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint',
    'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint',
    'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint',
    'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint',
    'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint')


def quat_rotate_inverse(q_wxyz, v):
    """Rotate vectors ``v`` (..., 3) by the inverse of quaternions ``q_wxyz`` (..., 4)."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    q = q/np.linalg.norm(q, axis=-1, keepdims=True)
    w, u = q[..., :1], -q[..., 1:]
    v = np.asarray(v, dtype=np.float64)
    t = 2.0*np.cross(u, v)
    return v + w*t + np.cross(u, t)


def quat_to_matrix(q_wxyz):
    q = np.asarray(q_wxyz, dtype=np.float64)
    q = q/np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = (q[..., i] for i in range(4))
    return np.stack([
        np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
        np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
        np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], -2)


@dataclass(frozen=True)
class HeftSpec:
    joint_names: tuple[str, ...]
    kps: np.ndarray
    kds: np.ndarray
    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    action_clip: float
    control_hz: int
    future_steps: tuple[int, ...]
    angvel_steps: tuple[int, ...]
    gravity_steps: tuple[int, ...]
    joint_pos_steps: tuple[int, ...]
    joint_vel_steps: tuple[int, ...]
    prev_action_steps: int
    boot_max: int
    include_diff: bool
    policy_path: Path

    @classmethod
    def load(cls, bundle) -> 'HeftSpec':
        import yaml
        bundle = Path(bundle)
        spec = yaml.safe_load((bundle/'deploy.yaml').read_text())
        action, obs = spec['action'], spec['observation']
        if obs.get('compliance', {}).get('enabled'):
            raise ValueError('compliance-flag checkpoints are not supported')
        f = lambda key: np.asarray(action[key], dtype=np.float64)
        return cls(tuple(action['joint_names']), f('kps'), f('kds'), f('default_joint_pos'), f('scale'),
                   float(action['clip']), int(spec['control_hz']), tuple(obs['future_steps']),
                   tuple(obs['root_angvel_history_steps']), tuple(obs['projected_gravity_history_steps']),
                   tuple(obs['joint_pos_history_steps']), tuple(obs['joint_vel_history_steps']),
                   int(obs['prev_action_steps']), int(obs['boot_indicator_max']),
                   obs['target_joint']['layout'] == 'full', bundle/spec['policy'])

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    @property
    def horizon(self) -> int:
        return max(self.future_steps)

    @property
    def history(self) -> int:
        return max(0, -min(self.future_steps))

    def obs_size(self) -> int:
        n, j = len(self.future_steps), self.n_joints
        return (1 + (n-1)*3 + n*6 + n*j*(2 if self.include_diff else 1) + n + n*3
                + 3*len(self.angvel_steps) + 3*len(self.gravity_steps)
                + j*len(self.joint_pos_steps) + j*len(self.joint_vel_steps) + j*self.prev_action_steps)


class Reference:
    """The reference timeline the tracker reads: frames are appended, a cursor marks the current one.

    Arrays: ``root_pos`` (T, 3), ``root_quat`` (T, 4, wxyz), ``joints`` (T, J) in the checkpoint order. Reads past
    either end are clamped (the tail is held), as ``_clamp_indices`` does in the release.
    """

    def __init__(self, root_pos, root_quat, joints):
        self.root_pos = np.atleast_2d(np.asarray(root_pos, dtype=np.float64)).copy()
        self.root_quat = np.atleast_2d(np.asarray(root_quat, dtype=np.float64)).copy()
        self.joints = np.atleast_2d(np.asarray(joints, dtype=np.float64)).copy()
        self.cursor = 0

    def __len__(self):
        return len(self.root_pos)

    def append(self, root_pos, root_quat, joints):
        self.root_pos = np.vstack([self.root_pos, np.atleast_2d(root_pos)])
        self.root_quat = np.vstack([self.root_quat, np.atleast_2d(root_quat)])
        self.joints = np.vstack([self.joints, np.atleast_2d(joints)])

    def advance(self):
        """Move to the next frame unless at the last one (``update_obs``'s cursor rule)."""
        if self.cursor < len(self)-1:
            self.cursor += 1

    def window(self, offsets):
        idx = np.clip(self.cursor+np.asarray(offsets), 0, len(self)-1)
        return self.root_pos[idx], self.root_quat[idx], self.joints[idx]


class HeftTracker:
    """One HEFT control step: measured state + reference -> joint position targets (checkpoint joint order)."""

    def __init__(self, bundle, providers=None, threads=1):
        self.spec = HeftSpec.load(bundle)
        self._session = None
        self._providers = providers or ['CPUExecutionProvider']
        self._threads = threads
        self.reset()

    # -- policy --------------------------------------------------------------------------------------------
    @property
    def session(self):
        if self._session is None:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = self._threads
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(str(self.spec.policy_path), sess_options=options,
                                                 providers=self._providers)
            shape = self._session.get_inputs()[0].shape
            if isinstance(shape[-1], int) and shape[-1] != self.spec.obs_size():
                raise ValueError(f'observation size {self.spec.obs_size()} != policy input {shape[-1]}')
        return self._session

    def reset(self):
        s = self.spec
        j = s.n_joints
        self.angvel = np.zeros((max(s.angvel_steps)+1, 3))
        self.gravity = np.zeros((max(s.gravity_steps)+1, 3))
        self.qpos = np.zeros((max(s.joint_pos_steps)+1, j))
        self.qvel = np.zeros((max(s.joint_vel_steps)+1, j))
        self.prev_actions = np.zeros((s.prev_action_steps, j))
        self.last_action = np.zeros(j)
        self.boot = s.boot_max
        self.last_obs = None

    def observe(self, reference: Reference, q, dq, quat_wxyz, gyro_body):
        """Build one observation (and update the histories) for the reference's current cursor."""
        s = self.spec
        q, dq = np.asarray(q, dtype=np.float64), np.asarray(dq, dtype=np.float64)
        # update phase, in module order: boot, histories, previous actions
        self.boot = max(self.boot-1, 0)
        self.angvel = np.roll(self.angvel, 1, axis=0)
        self.angvel[0] = gyro_body
        g = quat_rotate_inverse(quat_wxyz, np.array([0., 0., -1.]))
        self.gravity = np.roll(self.gravity, 1, axis=0)
        self.gravity[0] = g/(np.linalg.norm(g)+1e-8)
        self.qpos = np.roll(self.qpos, 1, axis=0)
        self.qpos[0] = q
        self.qvel = np.roll(self.qvel, 1, axis=0)
        self.qvel[0] = dq
        self.prev_actions = np.roll(self.prev_actions, 1, axis=0)
        self.prev_actions[0] = self.last_action

        pos, quat, joints = reference.window(s.future_steps)
        pos_diff = quat_rotate_inverse(quat[0], pos[1:]-pos[:1])
        rel = quat_to_matrix(quat_wxyz).T[None] @ quat_to_matrix(quat)
        rot6d = rel[:, :, :2].transpose(0, 2, 1).reshape(-1)
        targets = joints.reshape(-1) if not s.include_diff else np.concatenate([joints.reshape(-1), (joints-q).reshape(-1)])
        target_gravity = quat_rotate_inverse(quat, np.array([[0., 0., -1.]]))
        obs = np.concatenate([
            [self.boot/s.boot_max], pos_diff.reshape(-1), rot6d, targets, pos[:, 2], target_gravity.reshape(-1),
            self.angvel[list(s.angvel_steps)].reshape(-1), self.gravity[list(s.gravity_steps)].reshape(-1),
            self.qpos[list(s.joint_pos_steps)].reshape(-1), self.qvel[list(s.joint_vel_steps)].reshape(-1),
            self.prev_actions.reshape(-1)]).astype(np.float32)
        if obs.shape != (s.obs_size(),):
            raise AssertionError(obs.shape)
        self.last_obs = obs
        return obs

    def act(self, obs):
        """Raw policy output -> joint position targets; the raw (clipped) action becomes the next 'previous'."""
        s = self.spec
        session = self.session
        raw = session.run(None, {session.get_inputs()[0].name: obs[None]})[0][0].astype(np.float64)
        raw = np.clip(raw, -s.action_clip, s.action_clip)
        self.last_action = raw
        return s.default_joint_pos + s.action_scale*raw

    def step(self, reference: Reference, q, dq, quat_wxyz, gyro_body):
        """Advance the reference cursor, observe, act (``TrackingPolicyRaw`` order)."""
        reference.advance()
        return self.act(self.observe(reference, q, dq, quat_wxyz, gyro_body))

    def pd_torque(self, target, q, dq, effort_limit=None):
        """Explicit joint PD with the checkpoint's gains, clipped to ``effort_limit`` if given."""
        tau = self.spec.kps*(np.asarray(target)-q) - self.spec.kds*np.asarray(dq)
        return tau if effort_limit is None else np.clip(tau, -effort_limit, effort_limit)


def reorder(values, source_names, target_names):
    """Columns of ``values`` (..., len(source_names)) rearranged into ``target_names`` order."""
    index = [list(source_names).index(n) for n in target_names]
    return np.asarray(values)[..., index]
