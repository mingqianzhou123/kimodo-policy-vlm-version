"""Analytic trajectories check signs, units, heading and timing independently."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kimodo_vla.benchmarks.toolbench.adapter import SPAWN_YAW
from kimodo_vla.controllers.decoupled_wbc import encode
from kimodo_vla.core.types import Observation, yaw_matrix
from kimodo_vla.model.planner import KimodoPlanner


@pytest.mark.parametrize('heading', [-2.8, .3, 1.1])
def test_analytic_velocity_height_heading_survive_training_features_and_decode(heading):
    config = SimpleNamespace(hand_control_mode='continuous', action_chunk=50, action_history=100)
    planner = KimodoPlanner(SimpleNamespace(config=config, fps=30), {}, control_fps=50, execution_frames=15)
    # Construct the trajectory with elementary trigonometry; do not use the
    # command integrator or the reference action encoder to make the expected data.
    t = np.arange(151) / 50
    vx, vy = .2, -.1
    c, s = np.cos(heading), np.sin(heading)
    position = np.stack([t * (c * vx - s * vy), t * (s * vx + c * vy), .74 + .01 * t], axis=1)
    joints = np.zeros((len(t), 29), np.float32)
    rotation = np.repeat(yaw_matrix(heading)[None], len(t), axis=0)
    first_heading = yaw_matrix(.37)  # observed initial heading differs from commanded heading
    features = planner.kimodo.target(joints, position, rotation, first_heading)
    decoded = planner.kimodo.representation.inverse(features, is_normalized=False, return_numpy=False)
    cut = 30  # at control step 50
    output = {key: value[cut:cut + 50] for key, value in decoded.items() if torch.is_tensor(value)}
    output.update(hand_closure=torch.zeros(50, 2),
                  history_last_root_position=decoded['root_positions'][cut - 1])
    obs = Observation(images={}, root_pos=position[50], root_rot=rotation[50],
                      joint_pos=joints[50], joint_mask=np.ones(29, bool), hand=np.zeros(2), instruction='x', step=50)
    planner._first_heading = first_heading
    chunk = planner._decode(output, obs, cut, torch.empty(0), torch.empty(0))
    command = encode(chunk.to_array40(), 50, lambda closure: np.zeros((len(closure), 14)), heading_offset=SPAWN_YAW)
    assert len(command) == 25
    np.testing.assert_allclose(command[:, 32:34], np.tile([vx, vy], (25, 1)), atol=2e-5)
    np.testing.assert_allclose(command[:, 31], .74 + .01 * np.arange(51, 76) / 50, atol=2e-6)
    error = (command[:, 35] - (heading + SPAWN_YAW) + np.pi) % (2 * np.pi) - np.pi
    np.testing.assert_allclose(error, 0, atol=2e-6)
