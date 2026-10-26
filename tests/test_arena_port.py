"""The Arena adapter on the chunk server answers the original HumanoidArena server's requests identically."""

import numpy as np
import pytest
import torch

from original_runtime import FakeModel, arena_payload, compare_calls, episode, old_runtime

from kimodo_vla.benchmarks.arena.policy import ROOT_BOUNDARY, make_policy, routes
from kimodo_vla.core.types import yaw_matrix
from kimodo_vla.model.planner import KimodoPlanner
from kimodo_vla.model.planner_original import OriginalTimingPlanner


def requests(data, execution_frames):
    """The simulator's requests: every state since the previous one, whenever the last chunk has run out."""
    first = yaw_matrix(data[0][0])
    control_frames = round(execution_frames * 50 / 30)
    pending = []
    for step, (yaw, joints, image) in enumerate(data):
        relative = first.T @ yaw_matrix(yaw)
        pending.append(np.concatenate([relative[:, :2].reshape(6), joints, np.zeros(29, np.float32)]))
        if step % control_frames == 0:
            yield arena_payload("t", image, pending)
            pending = []


@pytest.mark.parametrize("controller, rtc", [("sonic", False), ("sonic", True), ("twist2", False)])
def test_arena_server_matches_original(controller, rtc, monkeypatch):
    if controller == "twist2":
        monkeypatch.setenv("KIMODO_SERVER_ANCHOR_ROOT", "1")
    data = episode(steps=240)
    old_model, new_model = FakeModel(), FakeModel()
    rt = old_runtime(old_model, 15, rtc)
    planner = OriginalTimingPlanner(new_model, {"t": torch.zeros(1, 8)}, control_fps=50.0, execution_frames=15, rtc_overlap_frames=12 if rtc else 0,
                            root_boundary=ROOT_BOUNDARY[controller])
    handlers = routes(make_policy(planner))
    rt.reset(7)
    assert handlers[("POST", "/reset")]({"seed": 7}) == (200, {"status": "ok"})

    worst_action = 0.0
    for payload in requests(data, 15):
        old = rt.infer(payload)
        status, response = handlers[("POST", "/infer")](payload)
        new = np.asarray(response["action_chunk"], np.float32)
        assert status == 200 and new.shape == old.shape
        worst_action = max(worst_action, float(np.abs(new - old).max()))
    worst = compare_calls(old_model.calls, new_model.calls)
    print(controller, rtc, {k: f"{v:.1e}" for k, v in worst.items()}, f"actions {worst_action:.1e}")
    assert all(v < 1e-4 for v in worst.values()) and worst_action < 1e-4
