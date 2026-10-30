"""ToolBench demos: training observations equal what the adapter reports at evaluation, and the
commands rebuilt from a demo's commanded motion are the recorded ones (no simulator)."""

import numpy as np
import pytest

from kimodo_vla.benchmarks.replay import DirectPlanner
from kimodo_vla.benchmarks.toolbench import demos
from kimodo_vla.benchmarks.toolbench.adapter import CLOSED_HANDS, SPAWN_YAW, ToolBenchAdapter
from kimodo_vla.core.types import yaw_of


def recording(steps=500, seed=0):
    """Columns like a ToolBook episode: walking, two turns with the flag, a grasp with a ramp."""
    rng = np.random.default_rng(seed)
    t = np.arange(steps) / 50.0
    phase = rng.uniform(0, 2 * np.pi, 43)
    state = (0.3 * np.sin(0.7 * t[:, None] + phase)).astype(np.float64)
    action = state + 0.02
    closed = np.clip((t - 4.0) / 0.1, 0.0, 1.0)  # closes at 4 s over 0.1 s
    right = closed[:, None] * CLOSED_HANDS[1]
    action[:, 36:43] = right[:, [3, 4, 5, 6, 0, 1, 2]]  # recorded order: index, middle, thumb
    action[:, 22:29] = 0.0
    rate = np.where((t > 2) & (t < 3), 0.8, 0.0) + np.where((t > 6) & (t < 6.5), -0.6, 0.0)
    yaw = SPAWN_YAW + np.concatenate([[0.0], np.cumsum(rate[1:]) / 50.0])
    nav = np.stack([0.3 * np.sin(0.5 * t), 0.1 * np.cos(0.3 * t), rate, yaw], axis=1)
    height = (0.74 + 0.02 * np.sin(0.2 * t))[:, None]
    return {"observation.state": state, "action": action, "teleop.navigate_command": nav, "teleop.base_height_command": height}


def test_recording_names():
    assert demos.canonical("G1HookRetrieve-L0-R-v0") == "G1BallRetrieve-L0-R"
    assert demos.canonical("G1IceBreak-L2-S-v0") == "G1IceBreak-L2-S"
    assert demos.instruction("G1StickMove-L0-S-v0") == "Pick the tool for moving the ball to the target."


@pytest.mark.parametrize('execution', [10, 20, 30, 50])
def test_measured_history_schedule_matches_real_planner(execution):
    from kimodo_vla.model.planner import KimodoPlanner
    planner = object.__new__(KimodoPlanner)
    planner.model_fps, planner.control_fps, planner.execution_frames = 30, 50, execution
    actual, step = set(), 0
    while step < 3000:
        actual.add(step)
        step += len(planner.control_offsets(step, 50)) - 1
    assert demos.request_steps(3000, execution) == actual
    if execution == 50:
        assert sorted(actual)[:4] == [0, 81, 163, 245]


@pytest.mark.parametrize("interval", [83, 30])
def test_adapter_reports_what_training_saw_and_sends_the_recorded_commands(interval):
    columns = recording()
    demo = demos.ToolBenchDemo(columns, "Pick the tool and move the ball to the target.", request_interval=interval)
    trained = demo.observations()
    motion = demo.commanded_motion()
    states = demos.policy_state(demo.state, demo.previous_height)

    adapter, planner = ToolBenchAdapter(), DirectPlanner(motion, chunk=interval)
    adapter.reset()
    seen, sent = [], []
    for step in range(0, len(demo) - interval, interval):
        request = {"image": {"rgb_head_stereo_left": np.zeros((360, 640, 3), np.uint8)}, "instruction": demo.instruction,
                   "state": {"states": states[step][None]}}
        observations = adapter.to_observations(request)
        seen += observations
        observations[-1].step = step
        sent.append(adapter.to_response(planner.plan(observations), request))

    assert len(seen) == 1 + (len(sent) - 1) * interval  # the first request reports one step
    for t, (eval_obs, train_obs) in enumerate(zip(seen, trained)):
        np.testing.assert_allclose(eval_obs.joint_pos[eval_obs.joint_mask], train_obs.joint_pos[train_obs.joint_mask], atol=1e-6, err_msg=str(t))
        assert np.array_equal(eval_obs.joint_mask, train_obs.joint_mask)
        assert float(yaw_of(eval_obs.root_rot)) == pytest.approx(float(yaw_of(train_obs.root_rot)), abs=1e-5), t
        np.testing.assert_allclose(eval_obs.hand, train_obs.hand, atol=1e-6, err_msg=str(t))

    # The 36-D commands for steps 1.. are the recorded commands for those steps.
    command = np.concatenate(sent)
    steps = np.arange(1, len(command) + 1)
    np.testing.assert_allclose(command[:, 32:34], demo.nav[steps, :2], atol=1e-4)            # vx, vy
    np.testing.assert_allclose(command[:, 35], demo.nav[steps, 3], atol=1e-5)                # target heading
    np.testing.assert_allclose(command[:, 31], demo.height[steps], atol=1e-6)                # height
    np.testing.assert_allclose(command[:, 14:21], demo.command[steps, 15:22], atol=1e-6)     # left arm
    np.testing.assert_allclose(command[:, 28:31], demo.command[steps][:, [13, 14, 12]], atol=1e-6)  # torso
    np.testing.assert_allclose(command[:, 7:14], CLOSED_HANDS[1] * motion.hand[steps, 1:], atol=1e-6)
    recorded = demo.recorded_commands()[steps]  # ToolBench's own reading of the recording
    np.testing.assert_allclose(command[:, :34], recorded[:, :34], atol=1e-4)
    np.testing.assert_allclose(command[:, 35], recorded[:, 35], atol=1e-5)
    turning = np.abs(demo.nav[steps, 2]) >= 0.1
    assert np.mean((np.abs(command[:, 34]) >= 0.1) == turning) > 0.99  # the flag is on exactly while the operator turned
