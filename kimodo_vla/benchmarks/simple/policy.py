"""Entry points: a Kimodo checkpoint for SIMPLE, in its process or over HTTP.

In process (SIMPLE's official evaluation): ``policy(request)`` with the request
of ``benchmarks.simple.adapter`` gives the 40-D rows, ``command(rows, ...)`` the
36-D WBC command. Over HTTP: ``routes(policy)`` with ``NUMPY_JSON`` serves the
original server's ``/health``, ``/reset``, ``/infer`` (40-D) and ``/act`` (36-D).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from kimodo_vla.benchmarks.server import ChunkPolicy, Route
from kimodo_vla.benchmarks.simple import hands
from kimodo_vla.benchmarks.simple.adapter import CONTROL_FPS, SimpleAdapter
from kimodo_vla.controllers import decoupled_wbc
from kimodo_vla.core.agent import Agent

# SIMPLE's original evaluation never enabled heading tracking (ARCHITECTURE section 12).
TURN_FLAG = 0.0


def make_policy(planner, hand_mode: str, debounce: hands.HandDebounce | None = None, max_navigation_speed: float = 1.5) -> ChunkPolicy:
    return ChunkPolicy(Agent(planner), SimpleAdapter(hand_mode, debounce, max_navigation_speed))


def command(rows40: np.ndarray, hand_mode: str, fps: float = CONTROL_FPS) -> np.ndarray:
    """40-D reference rows -> SIMPLE's 36-D decoupled-WBC command."""
    targets = hands.continuous_targets if hand_mode == "continuous" else hands.binary_targets
    return decoupled_wbc.encode(rows40, fps, targets, turn_flag=TURN_FLAG)


def _state_history(request: dict) -> np.ndarray:
    state = request.get("state") or {}
    candidates = [state.get(k) for k in ("state_history", "observation_state_history", "states")] if isinstance(state, dict) else []
    candidates += [request.get("state_history"), request.get("observation_state_history")]
    for candidate in candidates:
        if candidate is not None:
            array = np.asarray(candidate, np.float32)
            array = array.reshape(1, -1) if array.ndim == 1 else array
            if array.ndim == 2 and array.shape[1] == 64:
                return array
    raise ValueError("SIMPLE /act requires state['state_history'] with shape (T, 64)")


def routes(policy: ChunkPolicy) -> dict[tuple[str, str], Route]:
    hand_mode = policy.adapter.hand_mode

    def health(_):
        return 200, {"status": "ok", "protocol": "simple", "state_dim": 64, "action_dim": 40}

    def reset(request):
        policy.reset(request.get("seed"))
        return 200, {"status": "ok"}

    def infer(request):
        return 200, {"action_chunk": policy(request).tolist()}

    def act(request: dict[str, Any]):
        task = str(request.get("task") or request.get("instruction") or "").strip()  # text embeddings are keyed by both
        images = request.get("image") or request.get("images")
        rows = policy({"task": task, "observation": {"images": images, "state_history": _state_history(request)}})
        return 200, {"action": command(rows, hand_mode), "err": 0.0, "traj_image": np.zeros((1, 1, 3), np.uint8)}

    return {("GET", ""): health, ("GET", "/health"): health, ("GET", "/healthz"): health,
            ("POST", "/reset"): reset, ("POST", "/infer"): infer, ("POST", "/act"): act}


def make_kimodo_policy(
    checkpoint_dir: str,
    text_embedding_cache: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    execution_frames: int = 50,
    diffusion_steps: int = 10,
    rtc_overlap_frames: int = 0,
    timing: str = "exact",
    max_navigation_speed: float = 1.5,
) -> ChunkPolicy:
    from kimodo_vla.model.planner import KimodoPlanner

    planner = KimodoPlanner.from_checkpoint(
        checkpoint_dir, text_embedding_cache, device=device, dtype=dtype, control_fps=CONTROL_FPS,
        execution_frames=execution_frames, diffusion_steps=diffusion_steps, rtc_overlap_frames=rtc_overlap_frames, timing=timing, root_boundary="extrapolate",
    )
    debounce = hands.HandDebounce(CONTROL_FPS) if planner.hand_mode == "binary" else None
    planner.hand_filter = debounce
    return make_policy(planner, planner.hand_mode, debounce, max_navigation_speed)
