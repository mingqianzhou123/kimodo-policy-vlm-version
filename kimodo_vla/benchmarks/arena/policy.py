"""Entry point: a Kimodo checkpoint served to HumanoidArena's evaluator."""

from __future__ import annotations

import torch
from pathlib import Path

from kimodo_vla.benchmarks.arena.adapter import CONTROL_FPS, ArenaAdapter
from kimodo_vla.benchmarks.server import ChunkPolicy, Route
from kimodo_vla.core.agent import Agent
from kimodo_vla.core.types import Planner

# Where the first root displacement of a chunk starts, per lower-body controller
# (docs/benchmarks/arena_simple.md): TWIST2's formal runs used --anchor-root 1.
ROOT_BOUNDARY = {"sonic": "history", "twist2": "first_prediction"}


def make_policy(planner: Planner, instruction_aliases=None) -> ChunkPolicy:
    return ChunkPolicy(Agent(planner), ArenaAdapter(instruction_aliases))


def cached_instruction_aliases(cache_dir):
    """Match the original server: simulator task ID -> cached natural language."""
    aliases = {}
    for path in sorted(Path(cache_dir).expanduser().glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        instruction = str(payload["instruction"]).strip()
        task_id = str(payload.get("task_id", "")).strip()
        for alias in (task_id, task_id.removeprefix("HumanoidArena::"),
                      payload.get("task_name"), instruction):
            if alias:
                if alias in aliases and aliases[alias] != instruction:
                    raise ValueError(f"Conflicting instruction for Arena task {alias!r}")
                aliases[alias] = instruction
    return aliases


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
    rtc_overlap_frames: int = 0,
    timing: str = "exact",
) -> ChunkPolicy:
    """``timing="original"`` decodes plans as the original repository did (``OriginalTimingPlanner``)."""
    from kimodo_vla.model.planner import KimodoPlanner
    from kimodo_vla.model.planner_original import OriginalTimingPlanner

    planner_class = OriginalTimingPlanner if timing == "original" else KimodoPlanner
    planner = planner_class.from_checkpoint(
        checkpoint_dir,
        text_embedding_cache,
        device=device,
        dtype=dtype,
        control_fps=CONTROL_FPS,
        execution_frames=execution_frames,
        diffusion_steps=diffusion_steps,
        rtc_overlap_frames=rtc_overlap_frames,
        root_boundary=ROOT_BOUNDARY[controller],
    )
    return make_policy(planner, cached_instruction_aliases(text_embedding_cache))
