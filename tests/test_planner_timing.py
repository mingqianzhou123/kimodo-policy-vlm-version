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
    def __init__(self):
        super().__init__()
        self.controlnet = torch.nn.Linear(1, 1)
        self.fps = MODEL_FPS
        self.config = SimpleNamespace(action_chunk=50, action_history=100, hand_control_mode="binary")
        self.text_encoder = None
        self.calls = []

    @property
    def device(self):
        return torch.device("cpu")

    def predict_future(self, *, history_motion, hand_history, **_):
        cut = history_motion.shape[1]  # valid while the history is shorter than action_history
        self.calls.append((cut, hand_history[0].clone()))
        frames = torch.arange(cut, cut + 50, dtype=torch.float32)
        root = torch.stack([torch.zeros(50), height(frames), torch.zeros(50)], dim=1)
        return dict(
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


@pytest.mark.parametrize("execution_frames", [10, 50])
def test_targets_follow_the_control_clock(execution_frames):
    model = FakeModel()
    agent = Agent(KimodoPlanner(model, {"go": torch.zeros(1, 8)}, control_fps=CONTROL_FPS, execution_frames=execution_frames))
    agent.reset()
    steps = 160  # the history stays shorter than action_history, so the fake model can read the cut
    targets = np.array([agent.step(observation()).root_height[0] for _ in range(steps)])
    expected = height(torch.arange(1, steps + 1) * MODEL_FPS / CONTROL_FPS).numpy()  # target of step t is for t + 1
    assert agent.num_plans >= 2
    np.testing.assert_allclose(targets, expected, atol=1e-5)

    for cut, hand in model.calls:  # the hand history is what earlier plans commanded for those frames
        frames = torch.arange(cut - hand.shape[0], cut)
        np.testing.assert_array_equal(hand[:, 0].numpy(), (frames >= MODEL_FPS).float().numpy())
