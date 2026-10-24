"""BiGym 2.0 <-> shared types.

Official interface (bigym2.github.io/docs, "Official configuration"):

* ``low_dim_obs``: [waist (3) if torso pitch is commanded] + left arm (7) +
  left fingers (2) + right arm (7) + right fingers (2) + pelvis x, y, z, yaw;
  then the velocities of the same joints; then 2 gripper states; then the
  pelvis x, y, z, yaw tracked by the lower-body controller.
* ``rgb_obs``: (3 cameras, 3, H, W) uint8 in the order head, right_wrist, left_wrist.
* action, normalized to [-1, 1]: vx, vy (body frame), pelvis height, yaw rate,
  [torso pitch], 14 arm joint targets, 2 grippers (0 open, 1 closed).

The legs (and the waist, on tasks without the pitch command) belong to the
frozen GR00T-WBC, so they are not observed and the adapter drops their targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from kimodo_vla.core.types import ActionChunk, Observation, joint_indices, yaw_matrix, yaw_of

ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
WAIST_JOINT_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
CAMERA_NAMES = ("head", "right_wrist", "left_wrist")
CONTROL_FPS = 50.0


def task_instructions() -> dict[str, str]:
    """Task name -> the sentence BiGym stamps into its LeRobot demonstrations (``task_instructions.yaml``)."""
    import bigym.loco.demos as demos

    return yaml.safe_load((Path(demos.__file__).parent / "task_instructions.yaml").read_text())


def task_instruction(task: str) -> str:
    return task_instructions()[task.removesuffix("_g1")]


@dataclass
class BiGym2Adapter:
    instruction: str
    pitch_command: bool          # True on the 14 tasks whose action has a torso-pitch slot
    action_low: np.ndarray
    action_high: np.ndarray
    cameras: tuple[str, ...] = CAMERA_NAMES  # a subset in BiGym's order; () for state-only tools
    control_fps: float = CONTROL_FPS
    # rad/s of yaw-rate command per rad of heading error. Off: replaying the operator's
    # commands through the adapter, gain 2 cut move_plate from 97% to 23% success and
    # pick_box from 77% to 2%, because it fights the controller's natural heading drift.
    yaw_gain: float = 0.0
    _previous_yaw: float | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_env(cls, env, task: str) -> "BiGym2Adapter":
        if env.config.frame_stack != 1 or env.config.normalize_low_dim_obs:
            raise ValueError("BiGym2Adapter expects frame_stack=1 and raw low_dim_obs")
        cameras = tuple(env.config.camera_keys)
        if cameras != tuple(c for c in CAMERA_NAMES if c in cameras):  # any subset, in BiGym's order
            raise ValueError(f"unexpected cameras: {cameras}")
        layout = env.wholebody_action_layout()
        if tuple(layout["limb_names"]) != ARM_JOINT_NAMES:
            raise ValueError(f"unexpected BiGym arm order: {layout['limb_names']}")
        pitch = layout["base_dim"] == 5
        adapter = cls(
            instruction=task_instruction(task),
            pitch_command=pitch,
            action_low=np.asarray(layout["action_low"], dtype=np.float32),
            action_high=np.asarray(layout["action_high"], dtype=np.float32),
            cameras=cameras,
        )
        expected = 2 * adapter._proprio_width + 2 + 4
        if env.low_dim_raw_observation_space.shape != (expected,):
            raise ValueError(f"low_dim_obs has {env.low_dim_raw_observation_space.shape}, expected ({expected},)")
        return adapter

    @property
    def _proprio_width(self) -> int:
        return (3 if self.pitch_command else 0) + 18 + 4

    def reset(self) -> None:
        self._previous_yaw = None

    def to_observation(self, timestep) -> Observation:
        low = np.asarray(timestep.low_dim_obs, dtype=np.float32)
        width = self._proprio_width
        qpos = low[:width]
        waist = qpos[:3] if self.pitch_command else np.zeros(3, np.float32)
        arm = qpos[-22:-4]  # left arm, left fingers, right arm, right fingers
        arms = np.concatenate([arm[0:7], arm[9:16]])
        grippers = low[2 * width : 2 * width + 2]
        x, y, z, yaw = low[2 * width + 2 : 2 * width + 6]

        joint_pos = np.zeros(29, np.float32)
        joint_mask = np.zeros(29, bool)
        # Unreported joints stay at 0: the controller keeps the waist within ~0.06 rad
        # when it owns it, and leg angles do not affect any observed feature.
        joint_pos[joint_indices(ARM_JOINT_NAMES)] = arms
        joint_mask[joint_indices(ARM_JOINT_NAMES)] = True
        joint_pos[joint_indices(WAIST_JOINT_NAMES)] = waist
        joint_mask[joint_indices(WAIST_JOINT_NAMES)] = self.pitch_command

        rgb = np.asarray(timestep.rgb_obs)
        images = {name: np.ascontiguousarray(rgb[i, -3:].transpose(1, 2, 0)) for i, name in enumerate(self.cameras)}
        return Observation(
            images=images,
            root_pos=np.array([x, y, z], np.float32),
            root_rot=yaw_matrix(float(yaw)),  # pelvis roll/pitch are passive and not observed
            joint_pos=joint_pos,
            joint_mask=joint_mask,
            hand=np.clip(grippers, 0.0, 1.0),
            instruction=self.instruction,
        )

    def to_raw_action(self, target: ActionChunk, obs: Observation, previous_yaw: float | None = None) -> np.ndarray:
        """Target frame 0 in BiGym's raw (unnormalized) action units.

        Velocities are backward differences into frame 0, like ``root_delta_xy``:
        the yaw rate runs from ``previous_yaw`` (the last step's target heading,
        or the measured one at the start) to frame 0. A forward difference leads
        the operator by one step and cost pick_box ~20% in demo replay.
        """
        fps = target.fps
        vx, vy = target.root_delta_xy[0] * fps
        yaw_now = float(yaw_of(obs.root_rot))
        yaw_target = float(yaw_of(target.root_rot[0]))
        yaw_from = yaw_now if previous_yaw is None else previous_yaw
        yaw_rate = _wrap(yaw_target - yaw_from) * fps + self.yaw_gain * _wrap(yaw_target - yaw_now)
        base = [vx, vy, target.root_height[0], yaw_rate]
        if self.pitch_command:
            base.append(target.joint_pos[0, joint_indices(["waist_pitch_joint"])[0]])
        arms = target.joint_pos[0, joint_indices(ARM_JOINT_NAMES)]
        grippers = (target.hand[0] >= 0.5).astype(np.float32)
        return np.concatenate([np.asarray(base, np.float32), arms, grippers])

    def to_env_action(self, target: ActionChunk, obs: Observation) -> np.ndarray:
        if abs(target.fps - self.control_fps) > 1e-6:
            raise ValueError(f"target is at {target.fps} fps, BiGym runs at {self.control_fps}")
        raw = self.to_raw_action(target, obs, self._previous_yaw)
        self._previous_yaw = float(yaw_of(target.root_rot[0]))
        normalized = 2.0 * (raw - self.action_low) / (self.action_high - self.action_low) - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)
