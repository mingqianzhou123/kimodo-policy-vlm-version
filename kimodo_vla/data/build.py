"""Recorded demonstrations -> the training files of ``kimodo_vla.data.episodes``.

A benchmark provides each demo as a ``benchmarks.base.Demonstration``;
``training_episode`` does the rest the same way for every benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from kimodo_vla.core.types import yaw_matrix, yaw_of
from kimodo_vla.model.features import KimodoFeatures
from kimodo_vla.motion.g1_reference import resample_hand_binary, resample_hand_continuous

if TYPE_CHECKING:
    from kimodo_vla.benchmarks.base import Demonstration


@dataclass
class DemoMotion:
    """Row ``t`` is the pose at control step ``t`` of the episode."""

    root_pos: np.ndarray   # (T, 3)
    root_rot: np.ndarray   # (T, 3, 3)
    joint_pos: np.ndarray  # (T, 29) G1_JOINT_NAMES order
    hand: np.ndarray       # (T, 2) commanded closure, 0 open / 1 closed (in between for continuous hands)


def training_episode(demo: "Demonstration", kimodo: KimodoFeatures, *, task: str, episode_id: str) -> tuple[dict, np.ndarray]:
    """One demo as a training record at model rate, and every camera's image at each of its frames.

    The history goes through the same ``to_observation`` and ``KimodoFeatures.history``
    calls as evaluation, so training sees exactly what the policy will see.
    """
    if abs(kimodo.control_fps - demo.control_fps) > 1e-6:
        raise ValueError(f"features are for {kimodo.control_fps} fps, the demo runs at {demo.control_fps}")
    observations = demo.observations()
    first_heading = yaw_matrix(float(yaw_of(observations[0].root_rot)))
    observed, mask, _, _ = kimodo.history(observations, first_heading)

    commanded = demo.commanded_motion()
    target = kimodo.target(commanded.joint_pos, commanded.root_pos, commanded.root_rot, first_heading)
    resample_hand = resample_hand_continuous if demo.hand_mode == "continuous" else resample_hand_binary
    hand = resample_hand(commanded.hand, demo.control_fps, kimodo.model_fps)
    if not (observed.shape == target.shape and hand.shape[0] == target.shape[0]):
        raise RuntimeError(f"length mismatch: observed {tuple(observed.shape)}, target {tuple(target.shape)}, hand {tuple(hand.shape)}")

    # Frame k of the model-rate grid is control step round(k * control / model): the step at which
    # evaluation makes its k-th cut (round(step * model / control) == k).
    steps = np.minimum(
        np.round(np.arange(target.shape[0]) * demo.control_fps / kimodo.model_fps).astype(int),
        len(observations) - 1,
    )
    record = {
        "instruction": demo.instruction,
        "fps": kimodo.model_fps,
        "observed_motion": observed.float(),
        "observed_mask": mask[0].clone(),
        "target_motion": target.float(),
        "hand": torch.as_tensor(hand, dtype=torch.float32),
        "task": task,
        "episode_id": episode_id,
        "cameras": list(demo.cameras),
    }
    if demo.hand_mode == "continuous":
        record["observed_hand"] = kimodo.hand_history(observations).float()
    return record, demo.images(steps)
