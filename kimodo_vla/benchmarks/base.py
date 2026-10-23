"""The contract every benchmark implements, plus the glue that turns it into a policy."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from kimodo_vla.core.agent import Agent
from kimodo_vla.core.types import ActionChunk, Observation


class BenchmarkAdapter(Protocol):
    """Translate between one benchmark's raw interface and the shared types."""

    control_fps: float

    def reset(self) -> None: ...

    def to_observation(self, raw_obs: Any) -> Observation: ...

    def to_env_action(self, target: ActionChunk, obs: Observation) -> np.ndarray:
        """``target`` is at ``control_fps``; frame 0 is what the next env step should reach."""
        ...


class AgentPolicy:
    """``policy(raw_obs) -> env action`` for any benchmark, with episode-level ``reset``."""

    def __init__(self, agent: Agent, adapter: BenchmarkAdapter):
        self.agent = agent
        self.adapter = adapter
        self._episode = 0
        self.base_seed: int | None = None

    def reset(self) -> None:
        seed = None if self.base_seed is None else self.base_seed + self._episode
        self._episode += 1
        self.adapter.reset()
        self.agent.reset(seed)

    def __call__(self, raw_obs: Any) -> np.ndarray:
        obs = self.adapter.to_observation(raw_obs)
        target = self.agent.step(obs)
        return self.adapter.to_env_action(target, obs)
