"""ToolBench adapter: state layout, the filled history, the 36-D command and the session protocol (no simulator)."""

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from kimodo_vla.benchmarks.server import NUMPY_JSON, make_server
from kimodo_vla.benchmarks.simple import hands
from kimodo_vla.benchmarks.toolbench.adapter import ToolBenchAdapter
from kimodo_vla.benchmarks.toolbench.policy import make_policy, routes
from kimodo_vla.core.types import G1_JOINT_INDEX, ActionChunk, yaw_matrix, yaw_of


def state(arm=0.1, closure=(1.0, 0.0)):
    values = np.zeros(32, np.float32)
    left = hands.CONTINUOUS_LEFT_CLOSE * closure[0]
    values[0:7] = np.concatenate([left[:3], left[5:7], left[3:5]])  # the command order: thumb, middle, index
    values[7:14] = hands.CONTINUOUS_RIGHT_CLOSE * closure[1]
    values[14:28] = arm + 0.01 * np.arange(14)
    values[28:31] = [0.01, 0.02, 0.03]
    values[31] = 0.74
    return values


def request(values, reset=False, session="a"):
    return {"image": {"rgb_head_stereo_left": np.zeros((360, 640, 3), np.uint8)}, "instruction": "use the hammer",
            "state": {"states": values[None]}, "history": {"reset": reset}, "session": session}


class TurningPlanner:
    """Walks forward and turns; records the observations it was given."""

    history_steps = 1000

    def __init__(self, frames=5):
        self.frames = frames
        self.seen = []

    def reset(self, seed=None):
        self.seed = seed
        self.seen = []

    def plan(self, history):
        self.seen.append(list(history))
        n, start = self.frames, yaw_of(history[-1].root_rot)
        yaws = start + 0.02 * np.arange(1, n + 1)
        joints = np.tile(np.linspace(-0.3, 0.3, 29, dtype=np.float32), (n, 1))
        return ActionChunk(np.tile([[0.006, 0.002]], (n, 1)).astype(np.float32), np.full(n, 0.72, np.float32),
                           np.stack([yaw_matrix(y) for y in yaws]), joints, np.full((n, 2), 0.5, np.float32), 50.0)


def test_state_layout():
    obs = ToolBenchAdapter().measured(state(), "x", None)
    assert obs.joint_pos[G1_JOINT_INDEX["left_shoulder_pitch_joint"]] == pytest.approx(0.1)
    assert obs.joint_pos[G1_JOINT_INDEX["right_wrist_yaw_joint"]] == pytest.approx(0.23)
    assert obs.joint_pos[G1_JOINT_INDEX["waist_roll_joint"]] == pytest.approx(0.01)
    assert obs.joint_pos[G1_JOINT_INDEX["waist_yaw_joint"]] == pytest.approx(0.03)
    assert obs.joint_mask.sum() == 17 and not obs.joint_mask[G1_JOINT_INDEX["left_knee_joint"]]
    np.testing.assert_allclose(obs.hand, [1.0, 0.0], atol=1e-6)


def test_history_is_filled_with_the_sent_targets_and_heading():
    planner = TurningPlanner(frames=5)
    policy = make_policy(planner)
    policy.reset()
    first = policy(request(state(), reset=True))
    second = policy(request(state(arm=0.2)))
    assert first.shape == second.shape == (5, 36)
    history = planner.seen[-1]
    assert [o.step for o in history] == list(range(6))  # 1 measured + 4 filled + 1 measured
    np.testing.assert_allclose(history[1].joint_pos, np.linspace(-0.3, 0.3, 29), atol=1e-6)
    # The newest observation is measured, facing the last heading we commanded.
    assert history[-1].joint_pos[G1_JOINT_INDEX["left_shoulder_pitch_joint"]] == pytest.approx(0.2)
    assert float(yaw_of(history[-1].root_rot)) == pytest.approx(0.1, abs=1e-6)
    assert float(second[0, 35]) == pytest.approx(0.12, abs=1e-6)


def test_command_layout():
    policy = make_policy(TurningPlanner(frames=3))
    policy.reset()
    command = policy(request(state(), reset=True))
    joints = np.linspace(-0.3, 0.3, 29, dtype=np.float32)
    np.testing.assert_allclose(command[0, 14], joints[G1_JOINT_INDEX["left_shoulder_pitch_joint"]], atol=1e-6)
    np.testing.assert_allclose(command[0, 27], joints[G1_JOINT_INDEX["right_wrist_yaw_joint"]], atol=1e-6)
    np.testing.assert_allclose(command[0, 28:31], joints[[G1_JOINT_INDEX[n] for n in ("waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint")]], atol=1e-6)
    np.testing.assert_allclose(command[0, 31:35], [0.72, 0.3, 0.1, 1.0], atol=1e-5)
    left = hands.CONTINUOUS_LEFT_CLOSE * 0.5
    np.testing.assert_allclose(command[0, 0:7], np.concatenate([left[:3], left[5:7], left[3:5]]), atol=1e-6)
    np.testing.assert_allclose(command[0, 7:14], hands.CONTINUOUS_RIGHT_CLOSE * 0.5, atol=1e-6)


@pytest.fixture
def base_url():
    server = make_server(routes(make_policy(TurningPlanner()), checkpoint="ckpt", base_seed=100), "127.0.0.1", 0, NUMPY_JSON)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def call(url, body=None):
    data = None if body is None else NUMPY_JSON.encode(body)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10) as response:
            return response.status, NUMPY_JSON.decode(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_sessions(base_url):
    status, body = call(f"{base_url}/act", request(state(), reset=True, session="a"))
    assert status == 200 and body["action"].shape == (5, 36) and body["action"].dtype == np.float32
    assert call(f"{base_url}/info")[1]["server_state"]["active_session"] == "a"
    assert call(f"{base_url}/act", request(state(), reset=True, session="b"))[0] == 409
    assert call(f"{base_url}/release", {"session": "a"})[1] == {"released": True}
    assert call(f"{base_url}/act", request(state(), reset=True, session="b"))[0] == 200
