"""Entry point: a BiGym 2.0 policy callable built from a Kimodo checkpoint."""

from __future__ import annotations

import torch

from kimodo_vla.benchmarks.base import AgentPolicy
from kimodo_vla.benchmarks.bigym2.adapter import CONTROL_FPS, BiGym2Adapter
from kimodo_vla.core.agent import Agent
from kimodo_vla.core.types import Planner

# BiGym's evaluation block uses seeds 620000 + i; reuse them for the diffusion noise.
EVAL_SEED_BASE = 620000


def make_policy(env, task: str, planner: Planner) -> AgentPolicy:
    """Wrap any planner as ``policy(timestep) -> action`` for ``bigym.loco.eval.evaluate``."""
    policy = AgentPolicy(Agent(planner), BiGym2Adapter.from_env(env, task))
    policy.base_seed = EVAL_SEED_BASE
    return policy


def make_kimodo_policy(
    env,
    task: str,
    checkpoint_dir: str,
    text_embedding_cache: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    execution_frames: int = 50,
    diffusion_steps: int = 10,
    weights: str = "model",
    **planner_options,
) -> AgentPolicy:
    """``weights``: ``"model"`` or a saved average such as ``"ema0.9999"``. ``planner_options`` go to
    ``KimodoPlanner`` (noise, rtc_overlap_frames, rtc_frozen_frames, blend_steps)."""
    from kimodo_vla.model.planner import KimodoPlanner

    planner = KimodoPlanner.from_checkpoint(
        checkpoint_dir,
        text_embedding_cache,
        device=device,
        dtype=dtype,
        control_fps=CONTROL_FPS,
        execution_frames=execution_frames,
        diffusion_steps=diffusion_steps,
        weights=weights,
        **planner_options,
    )
    return make_policy(env, task, planner)
