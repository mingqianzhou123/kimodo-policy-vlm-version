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


def make_policy(planner, turn: str | float = "operator") -> ChunkPolicy:
    return ChunkPolicy(Agent(planner), ToolBenchAdapter(turn))


def routes(policy: ChunkPolicy, *, name: str = "kimodo-vla", checkpoint: str | None = None,
           base_seed: int | None = None, metadata: dict | None = None) -> dict[tuple[str, str], Route]:
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
            **(metadata or {}),
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
            reset = (request.get("history") or {}).get("reset", False)
            if active["session"] != session:
                if not reset:
                    return 400, {"error": "A new evaluator session must start with history.reset=true."}
                active["episode"] = 0
            active.update(session=session, seen=time.monotonic(), busy=True)
            try:
                if reset:
                    seed = None if base_seed is None else base_seed + active["episode"]
                    policy.reset(seed)
                action = policy(request)
                if reset:
                    active["episode"] += 1
                return 200, {"action": action}
            finally:
                active.update(seen=time.monotonic(), busy=False)

    def release(request: dict[str, Any]):
        session = request.get("session") or ANONYMOUS_SESSION
        with lock:
            if active["session"] == session:
                active["session"] = None
            return 200, {"released": active["session"] is None}

    return {("GET", "/info"): info, ("POST", "/act"): act, ("POST", "/release"): release}


def make_kimodo_policy(
    checkpoint_dir: str,
    text_embedding_cache: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    execution_frames: int = 20,
    diffusion_steps: int = 10,
    rtc_overlap_frames: int = 0,
    turn: str | float = "operator",
    root_boundary: str = "extrapolate",
    condition_current_observation: bool | None = None,
) -> ChunkPolicy:
    """``root_boundary`` follows SIMPLE (same controller, no root position in the state) until the
    round-trip check on ToolBook demos says otherwise."""
    from kimodo_vla.model.planner import KimodoPlanner

    conditioning = {} if condition_current_observation is None else {"condition_current_observation": condition_current_observation}
    planner = KimodoPlanner.from_checkpoint(
        checkpoint_dir, text_embedding_cache, device=device, dtype=dtype, control_fps=CONTROL_FPS,
        execution_frames=execution_frames, diffusion_steps=diffusion_steps, rtc_overlap_frames=rtc_overlap_frames, root_boundary=root_boundary,
        align_execution_grid=True, **conditioning,
    )
    return make_policy(planner, turn)
