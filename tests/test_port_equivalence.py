"""Migration check: feed the same episode to the original HumanoidArena runtime and to Agent+KimodoPlanner."""

import numpy as np
import pytest
import torch

from original_runtime import FakeModel, arena_payload, compare_calls, episode, old_runtime

from kimodo_vla.core.agent import Agent
from kimodo_vla.core.types import Observation, yaw_matrix
from kimodo_vla.model.planner import KimodoPlanner


@pytest.mark.parametrize("execution_frames, rtc", [(50, False), (15, False), (15, True)])
def test_planner_matches_original_runtime(execution_frames, rtc):
    data = episode()
    first = yaw_matrix(data[0][0])

    # Original runtime: the client sends every state since its previous request.
    old_model = FakeModel()
    rt = old_runtime(old_model, execution_frames, rtc)
    old_actions, pending, cursor, plan = [], [], 0, None
    for yaw, joints, image in data:
        relative = first.T @ yaw_matrix(yaw)
        pending.append(np.concatenate([relative[:, :2].reshape(6), joints, np.zeros(29, np.float32)]))
        if plan is None or cursor >= len(plan):
            plan, cursor, pending = rt.infer(arena_payload("t", image, pending)), 0, []
        old_actions.append(plan[cursor])
        cursor += 1

    # New code path.
    new_model = FakeModel()
    planner = KimodoPlanner(new_model, {"do it": torch.zeros(1, 8)}, control_fps=50.0, execution_frames=execution_frames, rtc=rtc)
    agent = Agent(planner)
    agent.reset()
    new_actions = []
    for yaw, joints, image in data:
        obs = Observation(
            images={"head": image}, root_pos=np.zeros(3, np.float32), root_rot=yaw_matrix(yaw),
            joint_pos=joints, joint_mask=np.ones(29, bool), hand=np.zeros(2, np.float32), instruction="do it",
        )
        target = agent.step(obs)[0:1]
        target.root_rot = np.einsum("ij,tjk->tik", first.T, target.root_rot)
        new_actions.append(target.to_array40()[0])

    worst = compare_calls(old_model.calls, new_model.calls)
    action_diff = float(np.abs(np.stack(old_actions) - np.stack(new_actions)).max())
    print(f"plans: {len(new_model.calls)}  history length per plan: {[c['motion'].shape[1] for c in new_model.calls]}")
    print("max |old - new| model inputs:", {k: f"{v:.2e}" for k, v in worst.items()})
    print(f"max |old - new| over {len(new_actions)} executed 40-D actions: {action_diff:.2e}")
    assert all(v < 1e-4 for v in worst.values()) and action_diff < 1e-4
    assert any(c["rtc_motion"].numel() for c in new_model.calls) == rtc

