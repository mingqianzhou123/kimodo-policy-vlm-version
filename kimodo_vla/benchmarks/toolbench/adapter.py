"""HumanoidToolBench <-> shared types.

The evaluator (MuJoCo, its own environment and process) posts to ``/act``
(SNU-PI/HumanoidToolBench ``docs/PUBLIC_EVALUATION.md``, "Policy interface")::

    {"image": {"rgb_head_stereo_left": (360, 640, 3) uint8},
     "instruction": str, "state": {"states": (1, 32)}, "history": {"reset": bool}, "session": str}

and executes the whole returned ``(T, 36)`` chunk before asking again. State::

    [0:7]  left Dex3 hand (thumb, middle, index)   [7:14]  right hand (thumb, index, middle)
    [14:21] left arm   [21:28] right arm   [28:31] torso roll, pitch, yaw   [31] previous height command

Requests now include every measured control step since the previous request:
``state.state_history [N,32]`` and ``state.base_pose_history [N,7]``
(position xyz, quaternion wxyz). Only the newest step carries an image.
No predicted action is converted into a sensed state. Pelvis pose comes from
MuJoCo and from ToolBook's observation.base_pose during training.

The answer is NVIDIA's decoupled-WBC command (``controllers.decoupled_wbc``).
Its target heading is absolute: every scenario spawns the robot facing
``SPAWN_YAW``, the episode frame's zero. The turn flag follows the operators by
default (the turning rate while turning, else 0; ToolBook demos turn on 0-5% of
frames).

The head image is resized to ``IMAGE_SIZE`` here and in the training data alike
(``resize_image``), so both reach DINOv3 as the same pixels.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from scipy.spatial.transform import Rotation

from kimodo_vla.benchmarks.arena.adapter import decode_image
from kimodo_vla.controllers import decoupled_wbc
from kimodo_vla.core.types import ActionChunk, Observation, joint_indices, yaw_matrix

CONTROL_FPS = 50.0
CAMERA = "head"
SPAWN_YAW = -np.pi / 2  # G1ToolbenchTabletop's fixed robot orientation, quaternion (0.7071, 0, 0, -0.7071)
IMAGE_SIZE = 224  # DINOv3's input size; storing 360x640 frames would take ~1.2 TB for ToolBook
# Closed Dex3 poses (natural order: thumb 3, index 2, middle 2) that ToolBook's operators command; their
# hand commands are open, closed, or a short ramp between. (Some right-hand recordings use SIMPLE's pose.)
CLOSED_HANDS = np.asarray([[0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5],
                           [0.0, -0.7, -0.7, 1.0, 1.5, 1.0, 1.5]], np.float32)


def hand_closure(hand_q: np.ndarray) -> np.ndarray:
    """``(T, 14)`` natural-order hand joints -> ``(T, 2)`` closure: the projection onto ``CLOSED_HANDS``."""
    hand_q = np.asarray(hand_q, np.float32).reshape(-1, 2, 7)
    closure = np.sum(hand_q * CLOSED_HANDS[None], axis=2) / np.sum(CLOSED_HANDS**2, axis=1)
    return np.clip(closure, 0.0, 1.0).astype(np.float32)


def hand_targets(closure: np.ndarray) -> np.ndarray:
    """``(T, 2)`` closure -> ``(T, 14)`` natural-order joint targets."""
    closure = np.clip(np.asarray(closure, np.float32), 0.0, 1.0)
    return (closure[:, :, None] * CLOSED_HANDS[None]).reshape(-1, 14)
LEFT_ARM = decoupled_wbc.ARM_JOINT_NAMES[:7]
RIGHT_ARM = decoupled_wbc.ARM_JOINT_NAMES[7:]
OBSERVED_JOINTS = (*decoupled_wbc.ARM_JOINT_NAMES, *decoupled_wbc.TORSO_JOINT_NAMES)
OBSERVED_MASK = np.isin(np.arange(29), joint_indices(OBSERVED_JOINTS))


HISTORY_SOURCE = "measured_fullbody29_control_steps_v3"
IMAGE_PREPROCESSING = "stretch_bilinear_antialias_224_v1"
CURRENT_OBSERVATION_SOURCE = "disabled_v1_history_semantics"


def resize_image(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Original direct square resize, shared with training."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be HWC RGB uint8")
    h, w = image.shape[:2]
    if min(h, w, size) < 1:
        raise ValueError("image dimensions and size must be positive")
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float()
    resized = torch.nn.functional.interpolate(tensor, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return resized[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()



def measured_pose_observation(state: np.ndarray, base_pose: np.ndarray, instruction: str,
                              image: np.ndarray | None = None, body_q: np.ndarray | None = None) -> Observation:
    """Measured joints and pelvis pose, in the episode frame (spawn yaw removed)."""
    pose = np.asarray(base_pose, np.float32)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("base_pose must be 7 finite values: xyz + quaternion wxyz")
    if abs(float(np.linalg.norm(pose[3:])) - 1.0) > 0.01:
        raise ValueError("base_pose quaternion must have unit length")
    obs = measured_observation(state, 0.0, instruction, image)
    if body_q is not None:
        from .demos import BODY_JOINT_NAMES
        body_q = np.asarray(body_q, np.float32)
        if body_q.shape != (29,) or not np.isfinite(body_q).all():
            raise ValueError("body_q must be 29 finite measured joint angles")
        obs.joint_pos[joint_indices(BODY_JOINT_NAMES)] = body_q
        obs.joint_mask[:] = True
        if not np.allclose(body_q[15:29], np.asarray(state)[14:28], atol=1e-6, rtol=0) or not np.allclose(body_q[[13,14,12]], np.asarray(state)[28:31], atol=1e-6, rtol=0):
            raise ValueError("body_q_history disagrees with legacy arm/waist measurements")
    relative = yaw_matrix(SPAWN_YAW).T
    obs.root_pos = (relative @ pose[:3]).astype(np.float32)
    obs.root_rot = (relative @ Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()).astype(np.float32)
    return obs


def measured_observation(state: np.ndarray, heading: float, instruction: str, image: np.ndarray | None = None) -> Observation:
    """The evaluator's 32-D state, facing ``heading`` (episode frame): no pelvis pose is measured."""
    state = np.asarray(state, np.float32).reshape(-1)
    if state.shape != (32,) or not np.isfinite(state).all():
        raise ValueError(f"state must be 32 finite values, got {state.shape}")
    joint_pos = np.zeros(29, np.float32)
    joint_pos[joint_indices(LEFT_ARM)] = state[14:21]
    joint_pos[joint_indices(RIGHT_ARM)] = state[21:28]
    joint_pos[joint_indices(decoupled_wbc.TORSO_JOINT_NAMES)] = state[28:31]
    hand_q = decoupled_wbc.command_to_natural_order(state[None, :14])
    return Observation(
        images={} if image is None else {CAMERA: resize_image(image)},
        root_pos=np.array([0.0, 0.0, state[31]], np.float32),
        root_rot=yaw_matrix(heading),
        joint_pos=joint_pos,
        joint_mask=OBSERVED_MASK.copy(),
        hand=hand_closure(hand_q)[0],
        instruction=instruction,
    )


class ToolBenchAdapter:
    control_fps = CONTROL_FPS

    def __init__(self, turn: str | float = "operator"):
        """``turn``: ``"operator"``, or a constant flag (0 never tracks the heading, 1 always does).

        Binary and continuous hand checkpoints both send closure times ``CLOSED_HANDS``
        (a binary closure is 0 or 1).
        """
        if turn != "operator":
            turn = float(turn)
        self.turn = turn
        self.reset()

    def reset(self) -> None:
        self._expected_states = 1

    def to_observations(self, request: Mapping[str, Any]) -> list[Observation]:
        instruction = str(request.get("instruction", "")).strip()
        payload = request["state"]
        if "state_history" not in payload or "base_pose_history" not in payload:
            raise ValueError("ToolBench requires measured state_history and base_pose_history; update the simulator client")
        states = np.asarray(payload["state_history"], np.float32)
        poses = np.asarray(payload["base_pose_history"], np.float32)
        if "body_q_history" not in payload:
            raise ValueError("ToolBench requires measured body_q_history; update simulator client")
        bodies = np.asarray(payload["body_q_history"], np.float32)
        if bodies.shape != (self._expected_states, 29):
            raise ValueError(f"expected {self._expected_states} new measured body states, got {bodies.shape}")
        if states.shape != (self._expected_states, 32) or poses.shape != (self._expected_states, 7):
            raise ValueError(f"expected {self._expected_states} new measured control states, got {states.shape}, {poses.shape}")
        if not np.isfinite(states).all() or not np.isfinite(poses).all():
            raise ValueError("measured history contains NaN or Inf")
        latest = np.asarray(payload["states"], np.float32)
        if latest.shape != (1, 32) or not np.array_equal(latest[0], states[-1]):
            raise ValueError("states must equal the newest measured state_history row")
        image = decode_image(request["image"]["rgb_head_stereo_left"])
        return [measured_pose_observation(s, p, instruction, image if i == len(states) - 1 else None, body_q=b)
                for i, (s, p, b) in enumerate(zip(states, poses, bodies))]

    def to_response(self, chunk: ActionChunk, request: Mapping[str, Any]) -> np.ndarray:
        rows = chunk.to_array40()
        heading = decoupled_wbc.headings(rows)
        flags = decoupled_wbc.operator_turn_flags(heading, self.control_fps) if self.turn == "operator" else self.turn
        command = decoupled_wbc.encode(rows, self.control_fps, hand_targets, flags, heading_offset=SPAWN_YAW)
        self._expected_states = len(chunk)
        return command
