"""The chunk server over real HTTP: routing, both wire encodings, errors."""

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from kimodo_vla.benchmarks.server import NUMPY_JSON, PLAIN_JSON, ChunkPolicy, make_server
from kimodo_vla.core.agent import Agent
from kimodo_vla.core.types import ActionChunk, Observation


class EchoPlanner:
    """Plans a chunk whose joint 0 counts the observations seen so far."""

    history_steps = 1000

    def reset(self, seed=None):
        self.seed = seed

    def plan(self, history):
        n = 3
        joints = np.full((n, 29), len(history), np.float32)
        return ActionChunk(np.zeros((n, 2), np.float32), np.full(n, 0.7, np.float32),
                           np.repeat(np.eye(3, dtype=np.float32)[None], n, 0), joints, np.zeros((n, 2), np.float32), 50.0)


class CountAdapter:
    control_fps = 50.0

    def reset(self):
        pass

    def to_observations(self, request):
        return [Observation(images={}, root_pos=np.zeros(3, np.float32), root_rot=np.eye(3, dtype=np.float32),
                            joint_pos=np.zeros(29, np.float32), joint_mask=np.ones(29, bool),
                            hand=np.zeros(2, np.float32), instruction="x") for _ in range(int(request["steps"]))]

    def to_response(self, chunk, request):
        # An array for the NumPy-aware encoding, a list for plain JSON.
        return {"action": chunk.joint_pos[:, 0] if request.get("numpy") else chunk.joint_pos[:, 0].tolist()}


@pytest.fixture
def serve():
    servers = []

    def start(routes, codec):
        server = make_server(routes, "127.0.0.1", 0, codec)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_port}"

    yield start
    for server in servers:
        server.shutdown()


def post(url, body: bytes):
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


@pytest.mark.parametrize("codec", [PLAIN_JSON, NUMPY_JSON])
def test_requests_accumulate_history_and_reset_clears_it(serve, codec):
    policy = ChunkPolicy(Agent(EchoPlanner()), CountAdapter())
    routes = {
        ("POST", "/reset"): lambda request: (policy.reset(request.get("seed")), (200, {"ok": True}))[1],
        ("POST", "/act"): lambda request: (200, policy(request)),
    }
    base = serve(routes, codec)
    post(f"{base}/reset", codec.encode({"seed": 3}))
    numpy = codec is NUMPY_JSON
    answers = [codec.decode(post(f"{base}/act", codec.encode({"steps": n, "numpy": numpy}))[1])["action"] for n in (1, 4)]
    assert all(isinstance(a, np.ndarray) == numpy for a in answers)
    assert [int(np.asarray(a)[0]) for a in answers] == [1, 5]
    assert policy.agent.planner.seed == 3


def test_errors_and_unknown_routes(serve):
    def fail(request):
        raise ValueError("bad state")

    base = serve({("POST", "/act"): fail, ("GET", "/health"): lambda _: (200, {"status": "ok"})}, PLAIN_JSON)
    status, body = post(f"{base}/act", b"{}")
    assert status == 500 and json.loads(body)["error"] == "ValueError: bad state"
    assert post(f"{base}/nowhere", b"{}")[0] == 404
    with urllib.request.urlopen(f"{base}/health/", timeout=10) as response:
        assert json.loads(response.read()) == {"status": "ok"}
