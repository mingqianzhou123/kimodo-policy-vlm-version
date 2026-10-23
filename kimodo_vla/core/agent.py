"""Benchmark-agnostic control loop: keep recent observations, replan when the plan runs out."""

from __future__ import annotations

from collections import deque

from kimodo_vla.core.types import ActionChunk, Observation, Planner


class Agent:
    """Turns one observation per control step into a reference target.

    ``step`` returns the remaining part of the current plan: frame 0 is the
    target for the next control step, later frames are look-ahead that an
    adapter may use (for example to estimate a velocity command). The planner
    is called again once every frame of the previous plan has been handed out.
    """

    def __init__(self, planner: Planner):
        self.planner = planner
        self.history: deque[Observation] = deque(maxlen=max(1, planner.history_steps))
        self._plan: ActionChunk | None = None
        self._cursor = 0
        self._steps = 0
        self.num_plans = 0

    def reset(self, seed: int | None = None) -> None:
        self.planner.reset(seed)
        self.history.clear()
        self._plan = None
        self._cursor = 0
        self._steps = 0
        self.num_plans = 0

    def step(self, observation: Observation) -> ActionChunk:
        observation.step = self._steps
        self._steps += 1
        self.history.append(observation)
        if self._plan is None or self._cursor >= len(self._plan):
            self._plan = self.planner.plan(list(self.history))
            if len(self._plan) == 0:
                raise RuntimeError("planner returned an empty chunk")
            self._cursor = 0
            self.num_plans += 1
        target = self._plan[self._cursor:]
        self._cursor += 1
        return target
