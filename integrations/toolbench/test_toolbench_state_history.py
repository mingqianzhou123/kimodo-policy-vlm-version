"""The simulation transport retains every measured step while commands are queued."""
import numpy as np
import pytest

from humanoidtoolbench.core.action import ActionCmd
from humanoidtoolbench.policies.composition import HumanoidPolicyAgent
from humanoidtoolbench.policies.remote_humanoid import HttpTaskPolicy


class RecordingClient:
    def __init__(self):
        self.requests = []

    def query_action(self, images, instruction, state, history):
        self.requests.append((state, history))
        action = np.zeros((3, 36), np.float32)
        action[:, 14:28] = -1  # deliberately unlike actual observations
        action[:, 31] = [.71, .72, .73]
        return action


class Controller:
    action_schema = "decoupled_v1"

    def step(self, goal, observation, **kwargs):
        return ActionCmd("test")

    def reset(self, **kwargs):
        pass


def observation(t):
    return {"joint_qpos": np.full(43, t / 10, np.float32),
            "base_pose": np.array([t / 100, 0, .74, 1, 0, 0, 0], np.float32),
            "head_stereo_left": np.zeros((8, 10, 3), np.uint8)}


def test_every_step_sent_once_with_consumed_height_and_snapshot_copies():
    client = RecordingClient()
    policy = HttpTaskPolicy("unused", 0, client=client)
    agent = HumanoidPolicyAgent(policy, Controller())
    for t in range(7):
        obs = observation(t)
        agent.get_action(obs, "x")
        obs["base_pose"][:] = -10  # simulator may reuse its arrays
        obs["joint_qpos"][:] = -10
    assert [len(s["state_history"]) for s, _ in client.requests] == [1, 3, 3]
    states = np.concatenate([s["state_history"] for s, _ in client.requests])
    poses = np.concatenate([s["base_pose_history"] for s, _ in client.requests])
    np.testing.assert_allclose(states[:, 14], np.arange(7) / 10)
    np.testing.assert_allclose(poses[:, 0], np.arange(7) / 100)
    np.testing.assert_allclose(states[:, 31], [.74, .71, .72, .73, .71, .72, .73])
    assert [h.get("reset", False) for _, h in client.requests] == [True, False, False]
    for s, _ in client.requests:
        np.testing.assert_array_equal(s["states"], s["state_history"][-1:])
    agent.reset()
    agent.get_action(observation(0), "x")
    s, h = client.requests[-1]
    assert h["reset"] and len(s["state_history"]) == 1
    assert s["states"][0, 31] == pytest.approx(.74)


def test_retrying_a_controller_step_does_not_duplicate_its_measurement():
    class RetryController(Controller):
        fail = True

        def step(self, goal, observation, **kwargs):
            if self.fail:
                self.fail = False
                raise RuntimeError('retry')
            return super().step(goal, observation, **kwargs)

    client = RecordingClient()
    agent = HumanoidPolicyAgent(HttpTaskPolicy('unused', 0, client=client), RetryController())
    with pytest.raises(RuntimeError, match='retry'):
        agent.get_action(observation(0), 'x')
    agent.get_action(observation(0), 'x')
    for t in range(1, 4):
        agent.get_action(observation(t), 'x')
    assert [len(s['state_history']) for s, _ in client.requests] == [1, 3]
    np.testing.assert_allclose(client.requests[1][0]['state_history'][:, 14], [.1, .2, .3])


def test_invalid_observation_does_not_mutate_history_before_retry():
    policy = HttpTaskPolicy('unused', 0, client=RecordingClient())
    bad = observation(0)
    bad['base_pose'][2] = np.nan
    with pytest.raises(ValueError, match='base_pose'):
        policy.observe(bad)
    assert not policy._measured_states and not policy._measured_poses
    policy.observe(observation(0))
    missing = observation(1)
    missing.pop('base_pose')
    with pytest.raises(ValueError, match='every control step'):
        policy.observe(missing)
    assert len(policy._measured_states) == len(policy._measured_poses) == 1
    policy.observe(observation(1))
    state = policy._state(observation(1))
    assert len(state['state_history']) == len(state['base_pose_history']) == 2
    np.testing.assert_allclose(state['state_history'][:, 14], [0, .1])


def test_fullbody_history_at_e20_includes_every_real_leg_joint():
    class E20Client(RecordingClient):
        def query_action(self, images, instruction, state, history):
            self.requests.append((state, history))
            action = np.zeros((33, 36), np.float32)
            action[:,31] = .74
            return action
    client = E20Client()
    agent = HumanoidPolicyAgent(HttpTaskPolicy('unused', 0, client=client), Controller())
    for t in range(67):
        obs = observation(t)
        obs['joint_qpos'] += np.arange(43, dtype=np.float32) / 100
        agent.get_action(obs, 'x')
        obs['joint_qpos'][:] = -100
    chunks = [s['body_q_history'] for s,_ in client.requests]
    assert [x.shape for x in chunks] == [(1,29),(33,29),(33,29)]
    np.testing.assert_allclose(np.concatenate(chunks), np.arange(67)[:,None]/10 + np.arange(29)[None,:]/100, atol=1e-6)
    agent.reset()
    agent.get_action(observation(0), 'x')
    assert client.requests[-1][0]['body_q_history'].shape == (1,29)
