"""HumanoidToolBench <-> shared types.

The evaluator (MuJoCo, its own environment and process) posts to ``/act``
(SNU-PI/HumanoidToolBench ``docs/PUBLIC_EVALUATION.md``, "Policy interface")::

    {"image": {"rgb_head_stereo_left": (360, 640, 3) uint8},
     "instruction": str, "state": {"states": (1, 32)}, "history": {"reset": bool}, "session": str}

and executes the whole returned ``(T, 36)`` chunk before asking again. State::

    [0:7]  left Dex3 hand (thumb, middle, index)   [7:14]  right hand (thumb, index, middle)
    [14:21] left arm   [21:28] right arm   [28:31] torso roll, pitch, yaw   [31] previous height command

There is no pelvis position or heading, and only one state per chunk, while
Kimodo is conditioned on a 50 Hz history with a root heading. So the adapter
reports the heading it last commanded, and fills the steps between two requests
with the targets it sent for them (``docs/benchmarks/toolbench.md``). Training
data has to be built the same way.

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


def resize_image(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """``(H, W, 3)`` uint8 -> ``(size, size, 3)`` uint8, bilinear with antialiasing."""
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float()
    resized = torch.nn.functional.interpolate(tensor, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return resized[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()


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


def commanded_observation(joint_pos: np.ndarray, root_rot: np.ndarray, root_height: float, hand: np.ndarray,
                          instruction: str) -> Observation:
    """A step between two requests, reported as what was commanded for it."""
    return Observation(
        images={}, root_pos=np.array([0.0, 0.0, root_height], np.float32), root_rot=np.asarray(root_rot, np.float32),
        joint_pos=np.asarray(joint_pos, np.float32).copy(), joint_mask=OBSERVED_MASK.copy(),
        hand=np.asarray(hand, np.float32).copy(), instruction=instruction,
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
        self._sent: ActionChunk | None = None
        self._heading = 0.0  # the episode starts facing yaw 0 of our own frame

    def to_observations(self, request: Mapping[str, Any]) -> list[Observation]:
        instruction = str(request.get("instruction", "")).strip()
        image = decode_image(request["image"]["rgb_head_stereo_left"])
        latest = measured_observation(np.asarray(request["state"]["states"])[-1], self._heading, instruction, image)
        if self._sent is None:
            return [latest]
        # The evaluator ran every row of the last chunk; the newest state comes after its final row.
        sent = self._sent
        filled = [
            commanded_observation(sent.joint_pos[i], sent.root_rot[i], sent.root_height[i], sent.hand[i], instruction)
            for i in range(len(sent) - 1)
        ]
        return filled + [latest]

    def to_response(self, chunk: ActionChunk, request: Mapping[str, Any]) -> np.ndarray:
        rows = chunk.to_array40()
        heading = decoupled_wbc.headings(rows)
        flags = decoupled_wbc.operator_turn_flags(heading, self.control_fps) if self.turn == "operator" else self.turn
        command = decoupled_wbc.encode(rows, self.control_fps, hand_targets, flags, heading_offset=SPAWN_YAW)
        self._sent = chunk
        self._heading = float(heading[-1])
        return command
