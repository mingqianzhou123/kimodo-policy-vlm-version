import numpy as np

from kimodo_vla.core.agent import Agent
from kimodo_vla.core.types import ActionChunk, Observation


def observation(i=0):
    return Observation(
        images={"head": np.zeros((4, 4, 3), np.uint8)},
        root_pos=np.array([i, 0, 0.75], np.float32),
        root_rot=np.eye(3, dtype=np.float32),
        joint_pos=np.zeros(29, np.float32),
        joint_mask=np.ones(29, bool),
        hand=np.zeros(2, np.float32),
        instruction="test",
    )


class CountingPlanner:
    history_steps = 4

    def __init__(self, chunk_len=3):
        self.chunk_len = chunk_len
        self.calls = []
        self.seeds = []

    def reset(self, seed=None):
        self.seeds.append(seed)

    def plan(self, history):
        self.calls.append(len(history))
        n = self.chunk_len
        joint = np.full((n, 29), len(self.calls), np.float32)
        return ActionChunk(
            np.zeros((n, 2), np.float32), np.full(n, 0.75, np.float32),
            np.repeat(np.eye(3, dtype=np.float32)[None], n, 0), joint, np.zeros((n, 2), np.float32), 50.0,
        )


def test_replans_only_when_plan_is_used_up():
    planner = CountingPlanner(chunk_len=3)
    agent = Agent(planner)
    lengths = [len(agent.step(observation(i))) for i in range(7)]
    assert lengths == [3, 2, 1, 3, 2, 1, 3]
    assert agent.num_plans == 3


def test_history_is_bounded_and_passed_in_order():
    planner = CountingPlanner(chunk_len=1)
    agent = Agent(planner)
    for i in range(10):
        agent.step(observation(i))
    assert planner.calls == [1, 2, 3, 4, 4, 4, 4, 4, 4, 4]
    assert [o.root_pos[0] for o in agent.history] == [6, 7, 8, 9]


def test_reset_clears_plan_and_forwards_seed():
    planner = CountingPlanner(chunk_len=5)
    agent = Agent(planner)
    agent.step(observation())
    agent.reset(seed=620001)
    assert len(agent.history) == 0
    assert len(agent.step(observation())) == 5
    assert planner.seeds == [620001]


def test_observe_then_replan_matches_step():
    stepped, requested = CountingPlanner(chunk_len=3), CountingPlanner(chunk_len=3)
    agent, server = Agent(stepped), Agent(requested)
    agent.reset(), server.reset()
    pending = []
    for i in range(9):
        agent.step(observation(i))
        pending.append(observation(i))
        if i % 3 == 0:  # a client asking for a chunk whenever the previous one is used up
            for obs in pending:
                server.observe(obs)
            server.replan()
            pending = []
    assert stepped.calls == requested.calls == [1, 4, 4]
    assert [o.step for o in server.history] == [3, 4, 5, 6]  # stamped in arrival order
