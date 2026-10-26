"""HumanoidArena <-> shared types, over the original repository's HTTP protocol.

The simulator (Isaac Lab, its own process) posts to ``/infer``::

    {"task": str,
     "observation": {"images": {"front": {"shape", "dtype", "data_b64"}},  # HWC RGB(A)
                     "state_history": [T, 64]}}

``state_history`` holds every 50 Hz step since its previous request; a row is the
root orientation as row-major rot6d (relative to the episode's first heading),
the 29 joint angles in ``G1_JOINT_NAMES`` order and their 29 velocities. The
answer is ``{"action_chunk": [T', 40]}``, the 40-D whole-body reference that
SONIC or TWIST2 tracks on the simulator side, so no controller conversion is
needed here. ``/reset {"seed"}`` starts an episode.
"""

from __future__ import annotations

import base64
from typing import Any, Mapping

import numpy as np
import torch

from kimodo_vla.core.types import ActionChunk, Observation
from kimodo_vla.motion.g1_reference import rot6d_row_to_matrix

CONTROL_FPS = 50.0
# Arena's "front" image is the egocentric view the released checkpoints call "head".
CAMERA = "head"


def decode_image(value: Any) -> np.ndarray:
    """``{"shape", "dtype", "data_b64"}`` (or an array) -> ``(H, W, 3)`` uint8."""
    if isinstance(value, Mapping):
        shape = tuple(int(v) for v in value["shape"])
        value = np.frombuffer(base64.b64decode(value["data_b64"]), dtype=np.dtype(value["dtype"])).reshape(shape)
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"expected an HWC RGB/RGBA image, got {image.shape}")
    return np.array(image[..., :3], dtype=np.uint8, order="C")  # a writable copy, as torch wants


def state64_observations(states: Any, instruction: str, image: np.ndarray | None) -> list[Observation]:
    """Rows of Arena's 64-D state as Observations; the newest one carries ``image``."""
    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 1:
        states = states[None]
    if states.ndim != 2 or states.shape[1] != 64 or len(states) == 0:
        raise ValueError(f"state_history must have shape [T, 64], got {states.shape}")
    if not np.isfinite(states).all():
        raise ValueError("state_history contains NaN or Inf")
    rotations = rot6d_row_to_matrix(torch.from_numpy(states[:, :6])).numpy()
    observations = []
    for i, row in enumerate(states):
        observations.append(Observation(
            images={CAMERA: image} if image is not None and i == len(states) - 1 else {},
            root_pos=np.zeros(3, np.float32),  # Arena reports no root position; the history never uses it
            root_rot=rotations[i],
            joint_pos=row[6:35].copy(),
            joint_mask=np.ones(29, bool),
            hand=np.zeros(2, np.float32),  # the hand condition is the policy's own previous output
            instruction=instruction,
        ))
    return observations


class ArenaAdapter:
    control_fps = CONTROL_FPS

    def reset(self) -> None:
        pass

    def to_observations(self, request: Mapping[str, Any]) -> list[Observation]:
        observation = request["observation"]
        if observation.get("state_history") is None:
            raise ValueError("observation.state_history is required: the policy is conditioned on the history")
        image = decode_image(observation["images"]["front"])
        return state64_observations(observation["state_history"], str(request.get("task", "")).strip(), image)

    def to_response(self, chunk: ActionChunk, request: Mapping[str, Any]) -> dict:
        return {"action_chunk": chunk.to_array40().tolist()}
