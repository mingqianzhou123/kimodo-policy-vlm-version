"""SIMPLE <-> shared types.

SIMPLE's official evaluation runs the simulator and the policy in one process;
its client hands the policy what the original runtime's ``/infer`` received::

    {"task": str,
     "observation": {"images": {"front": image},              # head_stereo_left, HWC
                     "state_history": [T, 64],                 # Arena's state layout
                     "hand_closure_history": [T, 2]},          # continuous checkpoints
     "hand_observation": {"left_hand_q", "left_hand_dq", ...}} # binary checkpoints' debounce

The 64-D state is HumanoidArena's (``benchmarks.arena.adapter``), built by
SIMPLE's client from the floating-base quaternion and body joints, relative to
the episode's first heading. The answer is the 40-D reference; SIMPLE's
decoupled WBC takes it as the 36-D command (``controllers.decoupled_wbc``).

Differences from Arena, all ported from the original runtime:
the root displacement starts from ``2 r0 - r1`` (the state has no root
position); binary hands go through ``HandDebounce``; continuous hands are
conditioned on the measured closure; and a chunk faster than
``max_navigation_speed`` is refused.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from kimodo_vla.benchmarks.arena.adapter import decode_image, state64_observations
from kimodo_vla.benchmarks.simple.hands import HandDebounce
from kimodo_vla.core.types import ActionChunk, Observation

CONTROL_FPS = 50.0
IMAGE_KEYS = ("head_stereo_left", "front_stereo_left", "rgb_head_stereo_left", "front")


def find_image(images: Any) -> np.ndarray:
    """The head image from SIMPLE's image dict (or a bare array)."""
    if isinstance(images, Mapping):
        for key in IMAGE_KEYS:
            if key in images:
                return decode_image(images[key])
        for value in images.values():  # /act names the camera freely
            return decode_image(value)
    if images is not None:
        return decode_image(images)
    raise ValueError("SIMPLE request has no HWC RGB image")


class SimpleAdapter:
    control_fps = CONTROL_FPS

    def __init__(self, hand_mode: str = "binary", debounce: HandDebounce | None = None, max_navigation_speed: float = 1.5):
        if hand_mode not in ("binary", "continuous"):
            raise ValueError(f"hand_mode must be 'binary' or 'continuous', got {hand_mode!r}")
        if not (np.isfinite(max_navigation_speed) and max_navigation_speed > 0):
            raise ValueError(f"max_navigation_speed must be finite and positive, got {max_navigation_speed}")
        self.hand_mode = hand_mode
        self.debounce = debounce
        self.max_navigation_speed = float(max_navigation_speed)

    def reset(self) -> None:
        if self.debounce is not None:
            self.debounce.reset()

    def to_observations(self, request: Mapping[str, Any]) -> list[Observation]:
        if self.debounce is not None:
            self.debounce.advance(request.get("hand_observation"))
        observation = request["observation"]
        if observation.get("state_history") is None:
            raise ValueError("observation.state_history is required: the policy is conditioned on the history")
        observations = state64_observations(
            observation["state_history"], str(request.get("task", "")).strip().removeprefix("simple/"),
            find_image(observation.get("images")),
        )
        if self.hand_mode == "continuous":
            closure = observation.get("hand_closure_history")
            if closure is None:
                raise ValueError("continuous hands need observation.hand_closure_history, one row per state")
            closure = np.asarray(closure, np.float32).reshape(-1, 2)
            if len(closure) != len(observations) or not np.isfinite(closure).all():
                raise ValueError(f"hand_closure_history must be finite with {len(observations)} rows, got {closure.shape}")
            if ((closure < -1e-6) | (closure > 1 + 1e-6)).any():
                raise ValueError("hand_closure_history must be within [0, 1]")
            for obs, value in zip(observations, np.clip(closure, 0.0, 1.0)):
                obs.hand = value
        return observations

    def to_response(self, chunk: ActionChunk, request: Mapping[str, Any]) -> np.ndarray:
        """The 40-D reference rows; refused when the planned walk is implausibly fast."""
        speed = np.linalg.norm(chunk.root_delta_xy, axis=1) * self.control_fps
        if float(speed.max()) > self.max_navigation_speed:
            raise ValueError(
                f"Predicted navigation speed exceeds safety threshold: first={speed[0]:.3f} m/s "
                f"max={speed.max():.3f} m/s threshold={self.max_navigation_speed:.3f} m/s"
            )
        if self.debounce is not None:
            self.debounce.executed(len(chunk))
        return chunk.to_array40()
