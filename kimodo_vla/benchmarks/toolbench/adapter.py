"""HumanoidToolBench <-> shared types.

The evaluator (MuJoCo, its own environment and process) posts to ``/act``
(SNU-PI/HumanoidToolBench ``docs/PUBLIC_EVALUATION.md``, "Policy interface")::

    {"image": {"rgb_head_stereo_left": (360, 640, 3) uint8},
     "instruction": str, "state": {"states": (1, 32)}, "history": {"reset": bool}, "session": str}

and executes the whole returned ``(T, 36)`` chunk before asking again. State::

    [0:7]  left Dex3 hand (thumb, middle, index)   [7:14]  right hand (thumb, index, middle)
    [14:21] left arm   [21:28] right arm   [28:31] torso roll, pitch, yaw   [31] previous height command

There is no pelvis position or heading, and only one state per chunk, while
Kimodo is conditioned on a 50 Hz history with a root heading. So (ARCHITECTURE
section 12, option a) the adapter reports the heading it last commanded, and
fills the steps between two requests with the targets it sent for them. Training
data has to be built the same way.

The answer is NVIDIA's decoupled-WBC command (``controllers.decoupled_wbc``).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from kimodo_vla.benchmarks.arena.adapter import decode_image
from kimodo_vla.benchmarks.simple import hands
from kimodo_vla.controllers import decoupled_wbc
from kimodo_vla.core.types import ActionChunk, Observation, joint_indices, yaw_matrix, yaw_of

CONTROL_FPS = 50.0
CAMERA = "head"
LEFT_ARM = decoupled_wbc.ARM_JOINT_NAMES[:7]
RIGHT_ARM = decoupled_wbc.ARM_JOINT_NAMES[7:]
OBSERVED_JOINTS = (*decoupled_wbc.ARM_JOINT_NAMES, *decoupled_wbc.TORSO_JOINT_NAMES)


class ToolBenchAdapter:
    control_fps = CONTROL_FPS

    def __init__(self, hand_mode: str = "continuous", turn_flag: float = 1.0):
        # Dex3 targets per closure: the SIMPLE teleop poses (ToolBench is derived from SIMPLE);
        # to be checked against ToolBench's recorded hand commands.
        self.hand_targets = hands.continuous_targets if hand_mode == "continuous" else hands.binary_targets
        self.turn_flag = float(turn_flag)
        self.reset()

    def reset(self) -> None:
        self._sent: ActionChunk | None = None
        self._heading = 0.0  # the episode starts facing yaw 0 of our own frame

    def measured(self, state: np.ndarray, instruction: str, image: np.ndarray | None) -> Observation:
        state = np.asarray(state, np.float32).reshape(-1)
        if state.shape != (32,) or not np.isfinite(state).all():
            raise ValueError(f"state must be 32 finite values, got {state.shape}")
        joint_pos = np.zeros(29, np.float32)
        joint_pos[joint_indices(LEFT_ARM)] = state[14:21]
        joint_pos[joint_indices(RIGHT_ARM)] = state[21:28]
        joint_pos[joint_indices(decoupled_wbc.TORSO_JOINT_NAMES)] = state[28:31]
        joint_mask = np.zeros(29, bool)
        joint_mask[joint_indices(OBSERVED_JOINTS)] = True
        hand_q = decoupled_wbc.command_to_natural_order(state[None, :14])
        return Observation(
            images={} if image is None else {CAMERA: image},
            root_pos=np.array([0.0, 0.0, state[31]], np.float32),
            root_rot=yaw_matrix(self._heading),
            joint_pos=joint_pos,
            joint_mask=joint_mask,
            hand=hands.closure_of(hand_q)[0],
            instruction=instruction,
        )

    def to_observations(self, request: Mapping[str, Any]) -> list[Observation]:
        instruction = str(request.get("instruction", "")).strip()
        image = decode_image(request["image"]["rgb_head_stereo_left"])
        latest = self.measured(np.asarray(request["state"]["states"])[-1], instruction, image)
        if self._sent is None:
            return [latest]
        # The evaluator ran every row of the last chunk; the newest state comes after its final row.
        sent = self._sent
        filled = [
            Observation(images={}, root_pos=np.array([0.0, 0.0, sent.root_height[i]], np.float32), root_rot=sent.root_rot[i],
                        joint_pos=sent.joint_pos[i].copy(), joint_mask=latest.joint_mask.copy(), hand=sent.hand[i].copy(),
                        instruction=instruction)
            for i in range(len(sent) - 1)
        ]
        return filled + [latest]

    def to_response(self, chunk: ActionChunk, request: Mapping[str, Any]) -> np.ndarray:
        command = decoupled_wbc.encode(chunk.to_array40(), self.control_fps, self.hand_targets, self.turn_flag)
        self._sent = chunk
        self._heading = float(yaw_of(chunk.root_rot[-1]))
        return command
