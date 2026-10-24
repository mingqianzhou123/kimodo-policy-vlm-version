"""Several cameras per sample: image tokens in camera order, one camera unchanged; needs no checkpoint."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from kimodo_vla.core.types import Observation, yaw_matrix
from kimodo_vla.model.kimodo_policy import KimodoPolicy, KimodoPolicyConfig
from kimodo_vla.model.modules.hand_control import HandTokenDiffusionDenoiser
from kimodo_vla.model.planner import KimodoPlanner

CAMERAS = ("head", "right_wrist", "left_wrist")


class PixelTokens(nn.Module):
    """Stands in for DINOv3: 196 tokens that depend on the image, so views can be told apart."""

    def forward(self, images):
        return images.flatten(1)[:, :196, None].float().expand(-1, -1, 8)


def policy_with(num_cameras):
    policy = KimodoPolicy.__new__(KimodoPolicy)
    nn.Module.__init__(policy)
    policy.config = KimodoPolicyConfig(num_cameras=num_cameras)
    policy.image_encoder = PixelTokens()
    return policy


def planner_with(cameras):
    config = SimpleNamespace(hand_control_mode="binary", action_chunk=50, action_history=100, num_cameras=len(cameras))
    return KimodoPlanner(SimpleNamespace(config=config, fps=30, text_encoder=None), {}, control_fps=50.0, cameras=cameras)


def observation(images):
    return Observation(
        images=images,
        root_pos=np.zeros(3, np.float32),
        root_rot=yaw_matrix(0.0),
        joint_pos=np.zeros(29, np.float32),
        joint_mask=np.ones(29, bool),
        hand=np.zeros(2, np.float32),
        instruction="test",
    )


def test_one_camera_tokens_are_the_encoder_output():
    egoview = torch.randint(0, 255, (2, 3, 84, 84), dtype=torch.uint8)
    assert torch.equal(policy_with(1)._encode_images(egoview), PixelTokens()(egoview))


def test_camera_tokens_are_concatenated_in_camera_order():
    egoview = torch.randint(0, 255, (2, 3, 3, 84, 84), dtype=torch.uint8)
    tokens = policy_with(3)._encode_images(egoview)
    assert tokens.shape == (2, 3 * 196, 8)
    for view in range(3):
        assert torch.equal(tokens[:, 196 * view : 196 * (view + 1)], PixelTokens()(egoview[:, view]))
    with pytest.raises(ValueError):
        policy_with(3)._encode_images(egoview[:, :2])


def test_planner_stacks_cameras_in_order_and_keeps_one_camera_unchanged():
    images = {name: np.full((84, 84, 3), i, np.uint8) for i, name in enumerate(CAMERAS)}
    single = planner_with(("head",)).egoview(observation(images))
    assert single.shape == (3, 84, 84) and torch.equal(single, torch.from_numpy(images["head"]).permute(2, 0, 1))
    stacked = planner_with(CAMERAS).egoview(observation(images))
    assert stacked.shape == (3, 3, 84, 84)
    assert [int(stacked[i].max()) for i in range(3)] == [0, 1, 2]
    with pytest.raises(ValueError):
        KimodoPlanner(planner_with(CAMERAS).model, {}, cameras=("head",))


def test_hand_head_takes_three_cameras_of_tokens():
    head = HandTokenDiffusionDenoiser(input_dim=16, hidden_dim=32, future_token_count=50, image_token_count=3 * 196)
    out = head(torch.randn(2, 3 * 196, 16), torch.randn(2, 50, 16), torch.randn(2, 50, 2), torch.zeros(2, 2), torch.zeros(2, dtype=torch.long))
    assert out.shape == (2, 50, 2)
