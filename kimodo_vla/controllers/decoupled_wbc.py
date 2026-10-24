"""NVIDIA's decoupled WBC (GR00T Balance/Walk) as SIMPLE and HumanoidToolBench drive it: the 36-D command.

One row per 50 Hz step::

    [0:7]   left Dex3 hand joint targets, thumb (3), middle (2), index (2)
    [7:14]  right Dex3 hand joint targets, thumb (3), index (2), middle (2)
    [14:28] left then right arm: shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
    [28:31] torso roll, pitch, yaw
    [31]    pelvis height (m)
    [32:34] vx, vy (m/s)
    [34]    turn flag: at |flag| >= 0.1 the controller turns towards [35]
    [35]    target heading (rad), absolute, in the episode-relative frame of the 40-D action

Unlike GR00T-WBC in BiGym (``groot_wbc``), the heading is a target, not a rate.
The flag is per benchmark: SIMPLE's original evaluation sent 0 (heading not
tracked); whenever |vx, vy, flag| >= 0.1 the controller uses its walking policy.
Hand joint targets come from the caller (``hands``), in the natural order
thumb (3), index (2), middle (2) for both hands.

``encode`` reproduces the original repository's ``arena_action_to_simple``
row for row (``tests/test_simple_port.py``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from kimodo_vla.core.types import G1_JOINT_INDEX
from kimodo_vla.motion.g1_reference import rot6d_row_to_matrix

ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
TORSO_JOINT_NAMES = ("waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint")
DIMENSION = 36

# (T, 2) closure, 0 open / 1 closed -> (T, 14) joint targets, left then right, natural order
HandTargets = Callable[[np.ndarray], np.ndarray]


def natural_to_command_order(hands: np.ndarray) -> np.ndarray:
    """``(T, 14)`` natural order -> the command's: the left hand is thumb, middle, index."""
    left, right = hands[:, :7], hands[:, 7:]
    return np.concatenate([left[:, :3], left[:, 5:7], left[:, 3:5], right], axis=1)


def command_to_natural_order(hands: np.ndarray) -> np.ndarray:
    """Inverse of ``natural_to_command_order`` (the left-hand swap is its own inverse)."""
    return natural_to_command_order(hands)


def encode(actions40: np.ndarray, fps: float, hand_targets: HandTargets, turn_flag: float = 0.0) -> np.ndarray:
    """Kimodo's 40-D reference rows -> ``(T, 36)`` commands.

    The 40-D root displacement is per frame in that frame's pelvis frame, so the
    velocity is it times ``fps``; the height is absolute.
    """
    actions = np.asarray(actions40, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 40:
        raise ValueError(f"expected (T, 40) actions, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("actions contain NaN or Inf")
    joints = actions[:, 9:38]
    command = np.zeros((len(actions), DIMENSION), np.float32)
    command[:, 0:14] = natural_to_command_order(np.asarray(hand_targets(actions[:, 38:40]), np.float32))
    command[:, 14:28] = joints[:, [G1_JOINT_INDEX[name] for name in ARM_JOINT_NAMES]]
    command[:, 28:31] = joints[:, [G1_JOINT_INDEX[name] for name in TORSO_JOINT_NAMES]]
    command[:, 31] = actions[:, 2]
    command[:, 32] = actions[:, 0] * float(fps)
    command[:, 33] = actions[:, 1] * float(fps)
    command[:, 34] = turn_flag
    rotation = rot6d_row_to_matrix(torch.from_numpy(actions[:, 3:9].copy())).numpy()
    command[:, 35] = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
    return command
