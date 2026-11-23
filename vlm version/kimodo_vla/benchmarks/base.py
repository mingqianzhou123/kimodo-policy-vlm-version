"""The contract every benchmark implements, plus the glue that turns it into a policy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Sequence

import numpy as np

from kimodo_vla.core.agent import Agent
from kimodo_vla.core.types import ActionChunk, Observation

if TYPE_CHECKING:
    from kimodo_vla.data.build import DemoMotion


class BenchmarkAdapter(Protocol):
    """Translate between one benchmark's raw interface and the shared types."""

    control_fps: float

    def reset(self) -> None: ...

    def to_observation(self, raw_obs: Any) -> Observation: ...

    def to_env_action(self, target: ActionChunk, obs: Observation) -> np.ndarray:
        """``target`` is at ``control_fps``; frame 0 is what the next env step should reach."""
        ...


class Demonstration(Protocol):
    """One recorded demo, as ``kimodo_vla.data.build.training_episode`` needs it."""

    instruction: str
    control_fps: float
    cameras: Sequence[str]  # names, in the order of ``images``
    hand_mode: str          # "binary" (open/closed) or "continuous" (closure in [0, 1])

    def observations(self) -> list[Observation]:
        """``to_observation`` at every control step, ``step`` filled in; images may be left out."""
        ...

    def commanded_motion(self) -> "DemoMotion":
        """The training target: what the operator commanded, as a pose trajectory at control rate.

        Not the motion the robot executed: under a whole-body controller that one
        applies arm sag twice when replayed as a target (docs/benchmarks/bigym2.md, section 3).
        """
        ...

    def images(self, steps: np.ndarray) -> np.ndarray:
        """``[len(steps), cameras, H, W, 3]`` uint8: every camera's recorded image at the given control steps."""
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
