"""BiGym2Adapter against the real BiGym 2.0 environment (CPU, no checkpoint)."""

import numpy as np
import pytest

pytest.importorskip("bigym")

from bigym.loco import make  # noqa: E402

from kimodo_vla.benchmarks.bigym2.adapter import ARM_JOINT_NAMES, BiGym2Adapter  # noqa: E402
from kimodo_vla.controllers.groot_wbc import base_command  # noqa: E402
from kimodo_vla.core.types import ActionChunk, G1_JOINT_INDEX, joint_indices, yaw_matrix, yaw_of  # noqa: E402

SEED = 620000


@pytest.fixture(scope="module", params=["move_plate", "pick_box"])
def env_and_adapter(request):
    env = make(request.param)
    yield env, BiGym2Adapter.from_env(env, request.param)
    env.close()


def sim_joint(env, name):
    robot = env.inner_env.robot
    model, data = robot.simulation.model, robot.simulation.data
    return float(data.qpos[model.joint(f"{robot.namespace}{name}").qposadr[0]])


def hold_target(start, step=0, frames=5, dx=0.0, yaw_rate=0.0, fps=50.0):
    """Keep the start pose; optionally walk forward at ``dx`` per frame and turn at ``yaw_rate``."""
    yaws = yaw_of(start.root_rot) + yaw_rate / fps * np.arange(step + 1, step + frames + 1)
    return ActionChunk(
        root_delta_xy=np.tile([dx, 0.0], (frames, 1)).astype(np.float32),
        root_height=np.full(frames, start.root_pos[2], np.float32),
        root_rot=np.stack([yaw_matrix(y) for y in yaws]),
        joint_pos=np.tile(start.joint_pos, (frames, 1)),
        hand=np.zeros((frames, 2), np.float32),
        fps=fps,
    )


def rollout(env, adapter, steps, **target_kwargs):
    start = obs = adapter.to_observation(env.reset(seed=SEED))
    adapter.reset()
    for step in range(steps):
        target = hold_target(start, step, **target_kwargs)
        obs = adapter.to_observation(env.step(adapter.to_env_action(target, obs)))
    return start, obs


def test_observation_matches_simulator(env_and_adapter):
    env, adapter = env_and_adapter
    obs = adapter.to_observation(env.reset(seed=SEED))
    for name in ARM_JOINT_NAMES:
        assert obs.joint_pos[G1_JOINT_INDEX[name]] == pytest.approx(sim_joint(env, name), abs=1e-5)
    assert obs.joint_mask[joint_indices(ARM_JOINT_NAMES)].all()
    assert not obs.joint_mask[G1_JOINT_INDEX["left_knee_joint"]]
    assert obs.joint_mask[G1_JOINT_INDEX["waist_pitch_joint"]] == adapter.pitch_command
    assert 0.6 < obs.root_pos[2] < 0.9
    assert obs.images["head"].shape == (84, 84, 3) and obs.images["head"].dtype == np.uint8
    assert obs.instruction


def test_action_normalization_inverts_bigym(env_and_adapter):
    env, adapter = env_and_adapter
    obs = adapter.to_observation(env.reset(seed=SEED))
    adapter.reset()
    target = hold_target(obs, dx=0.004)
    raw = adapter.to_raw_action(base_command(target, obs.root_rot), target)
    np.testing.assert_allclose(env.denormalize_action(adapter.to_env_action(target, obs)), raw, atol=1e-5)


def test_holding_the_current_pose_keeps_the_robot_still(env_and_adapter):
    env, adapter = env_and_adapter
    start, end = rollout(env, adapter, steps=150)
    assert not env.episode_fell()
    arms = joint_indices(ARM_JOINT_NAMES)
    assert np.abs(end.joint_pos[arms] - start.joint_pos[arms]).max() < 0.1
    assert np.linalg.norm(end.root_pos[:2] - start.root_pos[:2]) < 0.1


def test_root_delta_sets_walking_velocity(env_and_adapter):
    # Backwards, because the counter blocks forward walking in some scenes.
    env, adapter = env_and_adapter
    start, end = rollout(env, adapter, steps=150, dx=-0.3 / 50)  # -0.3 m/s for 3 s
    assert not env.episode_fell()
    heading = start.root_rot[:2, 0]
    assert float(heading @ (end.root_pos[:2] - start.root_pos[:2])) < -0.6  # measured -0.73


def test_yaw_target_turns_left(env_and_adapter):
    env, adapter = env_and_adapter
    start, end = rollout(env, adapter, steps=150, yaw_rate=0.3)  # 0.3 rad/s for 3 s
    assert not env.episode_fell()
    turned = float(yaw_of(end.root_rot) - yaw_of(start.root_rot))
    assert 0.4 < turned < 1.2
