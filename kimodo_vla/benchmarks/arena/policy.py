"""Entry point: a Kimodo checkpoint served to HumanoidArena's evaluator."""

from __future__ import annotations

import torch

from kimodo_vla.benchmarks.arena.adapter import CONTROL_FPS, ArenaAdapter
from kimodo_vla.benchmarks.server import ChunkPolicy, Route
from kimodo_vla.core.agent import Agent
from kimodo_vla.core.types import Planner

# Where the first root displacement of a chunk starts, per lower-body controller
# (ARCHITECTURE section 12): TWIST2's formal runs used --anchor-root 1.
ROOT_BOUNDARY = {"sonic": "history", "twist2": "first_prediction"}


def make_policy(planner: Planner) -> ChunkPolicy:
    return ChunkPolicy(Agent(planner), ArenaAdapter())


def routes(policy: ChunkPolicy) -> dict[tuple[str, str], Route]:
    """The original server's routes: ``/health``, ``/reset {seed}``, ``/infer``."""

    def health(_):
        return 200, {"status": "ok"}

    def reset(request):
        policy.reset(request.get("seed"))
        return 200, {"status": "ok"}

    def infer(request):
        return 200, policy(request)

    return {("GET", ""): health, ("GET", "/health"): health, ("POST", "/reset"): reset, ("POST", "/infer"): infer}


def make_kimodo_policy(
    checkpoint_dir: str,
    text_embedding_cache: str,
    *,
    controller: str = "sonic",
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    execution_frames: int = 15,
    diffusion_steps: int = 10,
    rtc: bool = False,
) -> ChunkPolicy:
    from kimodo_vla.model.planner import KimodoPlanner

    planner = KimodoPlanner.from_checkpoint(
        checkpoint_dir,
        text_embedding_cache,
        device=device,
        dtype=dtype,
        control_fps=CONTROL_FPS,
        execution_frames=execution_frames,
        diffusion_steps=diffusion_steps,
        rtc=rtc,
        root_boundary=ROOT_BOUNDARY[controller],
    )
    return make_policy(planner)
