"""The two data types every benchmark adapter and every planner agree on.

Conventions shared by both types:

* MuJoCo world frame, z up, metres and radians.
* Joint vectors use ``G1_JOINT_NAMES`` order (29 body joints of the Unitree G1).
* ``hand`` is a per-side closure in [0, 1], left first; 0 = open, 1 = closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

# Canonical 29-joint order. It matches the Kimodo-Policy 40-D action layout,
# so released checkpoints and datasets stay valid without reindexing.
G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint", "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)
G1_JOINT_INDEX = {name: i for i, name in enumerate(G1_JOINT_NAMES)}


def joint_indices(names: Sequence[str]) -> np.ndarray:
    """Positions of ``names`` inside ``G1_JOINT_NAMES``."""
    return np.asarray([G1_JOINT_INDEX[name] for name in names], dtype=np.int64)


def yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def yaw_of(rotation: np.ndarray) -> np.ndarray:
    """Heading angle of one ``(3, 3)`` or a batch ``(..., 3, 3)`` of rotations."""
    return np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0])


@dataclass
class Observation:
    """What the robot senses at one control step."""

    images: dict[str, np.ndarray]  # camera name -> (H, W, 3) uint8
    root_pos: np.ndarray           # (3,) pelvis position
    root_rot: np.ndarray           # (3, 3) pelvis orientation
    joint_pos: np.ndarray          # (29,) G1_JOINT_NAMES order
    joint_mask: np.ndarray         # (29,) bool; False where the benchmark does not report the joint
    hand: np.ndarray               # (2,) measured closure
    instruction: str
    step: int = 0                  # control steps since episode start; stamped by Agent

    def __post_init__(self) -> None:
        if self.root_pos.shape != (3,) or self.root_rot.shape != (3, 3):
            raise ValueError("root_pos must be (3,) and root_rot (3, 3)")
        if self.joint_pos.shape != (29,) or self.joint_mask.shape != (29,):
            raise ValueError("joint_pos and joint_mask must be (29,)")
        if self.hand.shape != (2,):
            raise ValueError("hand must be (2,)")


@dataclass
class ActionChunk:
    """A whole-body reference trajectory at ``fps``; frame 0 is the next step's target.

    ``root_delta_xy[t]`` is the pelvis displacement from frame ``t - 1`` to ``t``
    expressed in frame ``t``'s pelvis frame (frame ``-1`` is the pose the chunk
    starts from). ``root_height`` is absolute; ``root_rot`` is in the world frame.
    """

    root_delta_xy: np.ndarray  # (T, 2)
    root_height: np.ndarray    # (T,)
    root_rot: np.ndarray       # (T, 3, 3)
    joint_pos: np.ndarray      # (T, 29)
    hand: np.ndarray           # (T, 2)
    fps: float
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = len(self.root_height)
        shapes = {
            "root_delta_xy": (self.root_delta_xy.shape, (t, 2)),
            "root_rot": (self.root_rot.shape, (t, 3, 3)),
            "joint_pos": (self.joint_pos.shape, (t, 29)),
            "hand": (self.hand.shape, (t, 2)),
        }
        for name, (got, want) in shapes.items():
            if got != want:
                raise ValueError(f"{name} has shape {got}, expected {want}")

    def __len__(self) -> int:
        return len(self.root_height)

    def __getitem__(self, index: slice) -> "ActionChunk":
        if not isinstance(index, slice):
            raise TypeError("ActionChunk only supports slicing; use chunk[i:i + 1] for one frame")
        return ActionChunk(
            self.root_delta_xy[index], self.root_height[index], self.root_rot[index],
            self.joint_pos[index], self.hand[index], self.fps, dict(self.meta),
        )

    # Kimodo-Policy's 40-D action: [dx, dy, height, root rot6d (6), joints (29), hands (2)],
    # where rot6d is the first two columns of the rotation matrix, row-major.
    @classmethod
    def from_array40(cls, actions: np.ndarray, fps: float) -> "ActionChunk":
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 40:
            raise ValueError(f"expected (T, 40) actions, got {actions.shape}")
        return cls(
            root_delta_xy=actions[:, 0:2].copy(),
            root_height=actions[:, 2].copy(),
            root_rot=rot6d_to_matrix(actions[:, 3:9]),
            joint_pos=actions[:, 9:38].copy(),
            hand=actions[:, 38:40].copy(),
            fps=float(fps),
        )

    def to_array40(self) -> np.ndarray:
        rot6d = self.root_rot[:, :, :2].reshape(len(self), 6)
        return np.concatenate(
            [self.root_delta_xy, self.root_height[:, None], rot6d, self.joint_pos, self.hand],
            axis=1,
        ).astype(np.float32)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Inverse of ``R[:, :, :2].reshape(-1, 6)`` via Gram-Schmidt."""
    cols = np.asarray(rot6d, dtype=np.float32).reshape(-1, 3, 2)
    a, b = cols[..., 0], cols[..., 1]
    x = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (x * b).sum(-1, keepdims=True) * x
    y = b / np.linalg.norm(b, axis=-1, keepdims=True)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


class Planner(Protocol):
    """A model that turns recent observations into the next reference chunk.

    Kimodo-Policy and the upcoming VLM variant both implement this; the Agent
    and every benchmark adapter only ever see this interface.
    """

    history_steps: int  # observations (at control rate) the planner wants to see

    def reset(self, seed: int | None = None) -> None: ...

    def plan(self, history: Sequence[Observation]) -> ActionChunk: ...
