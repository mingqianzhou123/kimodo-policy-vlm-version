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


def current_observation_fields(observations, kimodo, steps):
    """Measurements at exactly the control steps selected for RGB, not a resampled surrogate."""
    first_heading = yaw_matrix(float(yaw_of(observations[0].root_rot)))
    selected = [observations[int(i)] for i in steps]
    current, mask = kimodo.current_observations(selected, first_heading)
    return {"current_observed_motion": current.float(),
            "current_observed_mask": mask[0].clone(),
            "current_observed_hand": torch.as_tensor(np.stack([o.hand for o in selected]), dtype=torch.float32)}


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

    padding = int(getattr(demo, "pose_padding_frames", 0))
    commanded = demo.commanded_motion(include_terminal=True) if padding else demo.commanded_motion()
    if padding:
        # Complete the final action at its actual control timestamp, then hold
        # the terminal pose BEFORE computing derivatives and smooth-root features.
        extra = int(np.ceil((padding + 2) * demo.control_fps / kimodo.model_fps))
        for name in ("joint_pos", "root_pos", "root_rot", "hand"):
            value = getattr(commanded, name)
            setattr(commanded, name, np.concatenate([value, np.repeat(value[-1:], extra, axis=0)]))
    target = kimodo.target(commanded.joint_pos, commanded.root_pos, commanded.root_rot, first_heading)
    resample_hand = resample_hand_continuous if demo.hand_mode == "continuous" else resample_hand_binary
    hand = resample_hand(commanded.hand, demo.control_fps, kimodo.model_fps)
    target_padded, hand_padded = None, None
    if padding:
        n = len(observed)
        target_padded = target[:n + padding].float()
        hand_padded = torch.as_tensor(hand[:n + padding], dtype=torch.float32)
        target, hand = target[:n], hand[:n]
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
    if padding:
        record.update(target_motion_padded=target_padded, target_hand_padded=hand_padded)
    if demo.hand_mode == "continuous":
        record["observed_hand"] = kimodo.hand_history(observations).float()
    if hasattr(demo, "preparation_metadata"):
        record.update(demo.preparation_metadata())
        if record.get("current_observation_source") and not record["current_observation_source"].startswith("disabled") :
            record.update(current_observation_fields(observations, kimodo, steps))
    return record, demo.images(steps)
