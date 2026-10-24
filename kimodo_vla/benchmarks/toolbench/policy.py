"""Entry point: a Kimodo checkpoint served to HumanoidToolBench's evaluator.

Routes (with ``NUMPY_JSON``), as the benchmark's own server: ``POST /act``,
``GET /info`` and ``POST /release``. One evaluator at a time: while a session
is active (a request in the last ``SESSION_IDLE_SECONDS``, not released), another
session gets HTTP 409, since its episode resets would clear this one's history.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import torch

from kimodo_vla.benchmarks.server import ChunkPolicy, Route
from kimodo_vla.benchmarks.toolbench.adapter import CONTROL_FPS, ToolBenchAdapter
from kimodo_vla.core.agent import Agent

SESSION_IDLE_SECONDS = 60.0
ANONYMOUS_SESSION = "anonymous"


def make_policy(planner, hand_mode: str = "continuous", turn_flag: float = 1.0) -> ChunkPolicy:
    return ChunkPolicy(Agent(planner), ToolBenchAdapter(hand_mode, turn_flag))


def routes(policy: ChunkPolicy, *, name: str = "kimodo-vla", checkpoint: str | None = None,
           base_seed: int | None = None) -> dict[tuple[str, str], Route]:
    """``base_seed`` makes the diffusion noise of episode ``i`` reproducible (seed ``base_seed + i``)."""
    lock = threading.Lock()
    active = {"session": None, "seen": 0.0, "busy": False, "episode": 0}

    def owner() -> str | None:
        if active["session"] is not None and (active["busy"] or time.monotonic() - active["seen"] < SESSION_IDLE_SECONDS):
            return active["session"]
        return None

    def info(_):
        current = owner()
        return 200, {
            "policy": name,
            "checkpoint": checkpoint,
            "action_schema": "decoupled_v1",
            "server_state": {
                "exclusive": True,
                "active_session": current,
                "request_in_progress": bool(current and active["busy"]),
                "seconds_since_request": round(time.monotonic() - active["seen"], 1) if current else None,
            },
        }

    def act(request: dict[str, Any]):
        session = request.get("session") or ANONYMOUS_SESSION
        with lock:
            current = owner()
            if current not in (None, session):
                return 409, {"error": "This policy server is serving another evaluator; run one server per evaluator."}
            active.update(session=session, seen=time.monotonic(), busy=True)
            try:
                if (request.get("history") or {}).get("reset", False):
                    seed = None if base_seed is None else base_seed + active["episode"]
                    active["episode"] += 1
                    policy.reset(seed)
                return 200, {"action": policy(request)}
            finally:
                active.update(seen=time.monotonic(), busy=False)

    def release(request: dict[str, Any]):
        session = request.get("session") or ANONYMOUS_SESSION
        if active["session"] == session and not active["busy"]:
            active["session"] = None
        return 200, {"released": active["session"] is None}

    return {("GET", "/info"): info, ("POST", "/act"): act, ("POST", "/release"): release}


def make_kimodo_policy(
    checkpoint_dir: str,
    text_embedding_cache: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    execution_frames: int = 50,
    diffusion_steps: int = 10,
    rtc: bool = False,
    turn_flag: float = 1.0,
    root_boundary: str = "extrapolate",
) -> ChunkPolicy:
    """``turn_flag`` and ``root_boundary`` are first guesses (SIMPLE's controller family, heading tracked);
    the round-trip check on ToolBench demos decides them."""
    from kimodo_vla.model.planner import KimodoPlanner

    planner = KimodoPlanner.from_checkpoint(
        checkpoint_dir, text_embedding_cache, device=device, dtype=dtype, control_fps=CONTROL_FPS,
        execution_frames=execution_frames, diffusion_steps=diffusion_steps, rtc=rtc, root_boundary=root_boundary,
    )
    return make_policy(planner, planner.hand_mode, turn_flag)
