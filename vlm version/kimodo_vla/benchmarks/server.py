"""Serving a policy to benchmarks whose simulator runs elsewhere and asks for whole chunks.

HumanoidArena and HumanoidToolBench (and SIMPLE's ``/act``) call the policy
over HTTP and want a whole action chunk per request, built from every control
step since their previous request. ``ChunkPolicy`` is the benchmark-agnostic
part (Agent + a ``RequestAdapter``, one request at a time); ``make_server``
puts it behind HTTP. A benchmark supplies its adapter, its routes, and its wire
encoding when that is not plain JSON.
"""

from __future__ import annotations

import base64
import json
import threading
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping, Protocol

import numpy as np
from numpy.lib.format import descr_to_dtype, dtype_to_descr

from kimodo_vla.core.agent import Agent
from kimodo_vla.core.types import ActionChunk, Observation


class RequestAdapter(Protocol):
    """Translate one benchmark's requests and responses."""

    control_fps: float

    def reset(self) -> None: ...

    def to_observations(self, request: Mapping[str, Any]) -> list[Observation]:
        """Every control step since the previous request, oldest first; the newest carries the images."""
        ...

    def to_response(self, chunk: ActionChunk, request: Mapping[str, Any]) -> Any:
        """The whole planned chunk (at ``control_fps``) in the benchmark's format."""
        ...


class ChunkPolicy:
    """``policy(request) -> response``: record the request's steps, plan once, translate the whole chunk."""

    def __init__(self, agent: Agent, adapter: RequestAdapter):
        self.agent = agent
        self.adapter = adapter
        self.lock = threading.Lock()

    def reset(self, seed: int | None = None) -> None:
        with self.lock:
            self.adapter.reset()
            self.agent.reset(seed)

    def __call__(self, request: Mapping[str, Any]) -> Any:
        with self.lock:
            for obs in self.adapter.to_observations(request):
                self.agent.observe(obs)
            chunk = self.agent.replan()
            if abs(chunk.fps - self.adapter.control_fps) > 1e-6:
                raise ValueError(f"plan is at {chunk.fps} fps, the benchmark runs at {self.adapter.control_fps}")
            return self.adapter.to_response(chunk, request)


@dataclass(frozen=True)
class Codec:
    decode: Callable[[bytes], Any]
    encode: Callable[[Any], bytes]


def _numpy_to_json(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.ascontiguousarray(value)
        return {
            "__numpy__": base64.b64encode(array.tobytes()).decode("ascii"),
            "dtype": dtype_to_descr(array.dtype),
            "shape": list(array.shape),
        }
    if isinstance(value, dict):
        return {str(k): _numpy_to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_numpy_to_json(v) for v in value]
    return value


def _numpy_from_json(value: Any) -> Any:
    if isinstance(value, dict) and "__numpy__" in value:
        array = np.frombuffer(base64.b64decode(value["__numpy__"]), descr_to_dtype(value["dtype"]))
        shape = tuple(int(v) for v in value.get("shape", ()))
        return array.reshape(shape) if shape else array[0]
    if isinstance(value, dict):
        return {k: _numpy_from_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_numpy_from_json(v) for v in value]
    return value


PLAIN_JSON = Codec(
    decode=lambda body: json.loads(body or b"{}"),
    encode=lambda value: json.dumps(value).encode("utf-8"),
)
# SIMPLE's and HumanoidToolBench's wire format: arrays as {"__numpy__": base64, "dtype", "shape"}.
NUMPY_JSON = Codec(
    decode=lambda body: _numpy_from_json(json.loads(body or b"{}")),
    encode=lambda value: json.dumps(_numpy_to_json(value), allow_nan=False).encode("utf-8"),
)

Route = Callable[[Any], tuple[int, Any]]  # decoded request body (None for GET) -> (status, response)


def make_server(routes: Mapping[tuple[str, str], Route], host: str, port: int, codec: Codec = PLAIN_JSON) -> ThreadingHTTPServer:
    """An HTTP server for ``{(method, path): route}``; ``serve_forever()`` runs it.

    Paths are matched without trailing slashes (``""`` is the root). A route that
    raises answers 500 with ``{"error": "<type>: <message>"}``; the traceback stays
    on the server's terminal.
    """
    table = {(method.upper(), path.rstrip("/")): route for (method, path), route in routes.items()}

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, value: Any) -> None:
            body = codec.encode(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle(self, method: str) -> None:
            route = table.get((method, self.path.split("?")[0].rstrip("/")))
            if route is None:
                self._reply(404, {"error": "not found"})
                return
            try:
                request = None
                if method == "POST":
                    request = codec.decode(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                status, response = route(request)
            except Exception as exc:  # the client gets one line, the server keeps the traceback
                traceback.print_exc()
                self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._reply(status, response)

        def do_GET(self) -> None:  # noqa: N802
            self._handle("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._handle("POST")

        def log_message(self, format: str, *args) -> None:
            print(f"[{self.log_date_time_string()}] {format % args}", flush=True)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
