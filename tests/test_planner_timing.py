"""KimodoPlanner turns predictions into control targets at the right time; needs no checkpoint.

The fake model predicts a pelvis height that rises linearly with model time and a
gripper that closes at 1 s, both as functions of absolute time (prediction frame i is
at model frame cut + i; early in an episode the history length equals the cut). A
planner on the right clock hands out, at every control step and across plan
boundaries, exactly the height of that step's time.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kimodo_vla.core.agent import Agent
from kimodo_vla.core.types import Observation, yaw_matrix
from kimodo_vla.model.planner import KimodoPlanner

MODEL_FPS, CONTROL_FPS = 30, 50.0


def height(frames: torch.Tensor) -> torch.Tensor:
    return 0.7 + 0.1 * frames / MODEL_FPS


class FakeModel(torch.nn.Module):
    def __init__(self, disagree: float = 0.0):
        super().__init__()
        self.controlnet = torch.nn.Linear(1, 1)
        self.fps = MODEL_FPS
        self.config = SimpleNamespace(action_chunk=50, action_history=100, hand_control_mode="binary")
        self.text_encoder = None
        self.calls = []
        self.disagree = disagree  # height offset that flips sign with every plan
        self.seeds, self.references = [], []

    @property
    def device(self):
        return torch.device("cpu")

    def predict_future(self, *, history_motion, hand_history, generator=None, rtc_motion_reference=None, **_):
        cut = history_motion.shape[1]  # valid while the history is shorter than action_history
        self.seeds.append(None if generator is None else generator.initial_seed())
        self.references.append(None if rtc_motion_reference is None else (cut, rtc_motion_reference[0, :, 0].clone()))
        offset = self.disagree * (-1) ** len(self.calls)
        self.calls.append((cut, hand_history[0].clone()))
        frames = torch.arange(cut, cut + 50, dtype=torch.float32)
        root = torch.stack([torch.zeros(50), height(frames) + offset, torch.zeros(50)], dim=1)
        features = torch.zeros(50, 417)
        features[:, 0] = frames  # lets the test check which frames RTC is handed
        return dict(
            motion_features=features,
            local_rot_mats=torch.eye(3).repeat(50, 34, 1, 1),
            root_positions=root,
            hand_binary=(frames >= MODEL_FPS).float().unsqueeze(1).expand(-1, 2).clone(),
            history_last_root_position=torch.tensor([0.0, float(height(torch.tensor(cut - 1.0))), 0.0]),
        )


def observation():
    return Observation(
        images={"head": np.zeros((4, 4, 3), np.uint8)}, root_pos=np.zeros(3, np.float32), root_rot=yaw_matrix(0.0),
        joint_pos=np.zeros(29, np.float32), joint_mask=np.ones(29, bool), hand=np.zeros(2, np.float32), instruction="go",
    )


def run(model, steps=160, seed=None, **options):
    agent = Agent(KimodoPlanner(model, {"go": torch.zeros(1, 8)}, control_fps=CONTROL_FPS, **options))
    agent.reset(seed)
    return agent, np.array([agent.step(observation()).root_height[0] for _ in range(steps)])


@pytest.mark.parametrize("options", [
    dict(execution_frames=10), dict(execution_frames=50),
    dict(execution_frames=10, noise="fixed", rtc_overlap_frames=12, blend_steps=8),  # a consistent model is left alone
])
def test_targets_follow_the_control_clock(options):
    model = FakeModel()
    steps = 160  # the history stays shorter than action_history, so the fake model can read the cut
    agent, targets = run(model, steps, **options)
    expected = height(torch.arange(1, steps + 1) * MODEL_FPS / CONTROL_FPS).numpy()  # target of step t is for t + 1
    assert agent.num_plans >= 2
    np.testing.assert_allclose(targets, expected, atol=1e-5)

    for cut, hand in model.calls:  # the hand history is what earlier plans commanded for those frames
        frames = torch.arange(cut - hand.shape[0], cut)
        np.testing.assert_array_equal(hand[:, 0].numpy(), (frames >= MODEL_FPS).float().numpy())


def test_fixed_noise_reuses_the_first_plans_seed():
    fixed, per_plan = FakeModel(), FakeModel()
    run(fixed, seed=7, execution_frames=10, noise="fixed")
    run(per_plan, seed=7, execution_frames=10)
    assert len(set(fixed.seeds)) == 1 and fixed.seeds[0] == per_plan.seeds[0]
    assert len(set(per_plan.seeds)) == len(per_plan.seeds)


def test_rtc_reference_is_the_previous_plan_at_the_same_frames():
    model = FakeModel()
    run(model, execution_frames=10, rtc_overlap_frames=12)
    assert model.references[0] is None
    for cut, frames in model.references[1:]:
        np.testing.assert_array_equal(frames.numpy(), np.arange(cut, cut + 12))


def test_blending_removes_the_jump_between_disagreeing_plans():
    _, raw = run(FakeModel(disagree=0.01), execution_frames=10)
    _, blended = run(FakeModel(disagree=0.01), execution_frames=10, blend_steps=8)
    assert np.abs(np.diff(raw)).max() > 0.015  # a 0.02 jump at every replan
    assert np.abs(np.diff(blended)).max() < 0.006
