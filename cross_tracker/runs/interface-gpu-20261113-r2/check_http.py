"""Exercise the real saved policy over loopback HTTP with aligned demo observations."""
from pathlib import Path
import base64
import gc
import hashlib
import json
import sys
import threading
import time
import urllib.request
import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from kimodo_vla.benchmarks.arena.policy import make_kimodo_policy, routes
from kimodo_vla.benchmarks.server import make_server, PLAIN_JSON

checkpoint = ROOT / "runs/pilot-sonic-20261113/training/checkpoint_200"
prepared = ROOT / "data/cache/prepared-v1/train/HOI_pp_box/sonic/episode_000001"
receipt = json.loads(prepared.with_suffix(".json").read_text())
assert receipt["source_mapping"]["source_frame_start"] == 0
frames = np.load(str(prepared) + ".frames.npy", mmap_mode="r")
assert frames.ndim == 5 and frames.shape[1] == 1
raw = ROOT / "data/cache/inputs-v1/audit/dataset/HOI_pp_box/sonic_refpose_v3_1/data"
table = pq.read_table(sorted(raw.rglob("*.parquet")), filters=[("episode_index", "=", 1)])
table = table.sort_by("frame_index")
states = np.array(table["observation.state"].to_pylist(), dtype=np.float32)
assert states.shape[1] == 64 and len(states) >= 51
instruction = receipt["source_mapping"]["instruction"]
record = json.loads((RUN / "run.json").read_text())
record.update(status="running", training_started=False, formal_training_started=False)
(RUN / "run.json").write_text(json.dumps(record, indent=2))
results = []
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

for controller in ("sonic", "twist2"):
    policy = make_kimodo_policy(str(checkpoint), str(ROOT / "data/cache/text"), controller=controller)
    server = make_server(routes(policy), "127.0.0.1", 0, PLAIN_JSON)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(route, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/{route}", data=data,
                                     headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=120) as response:
            assert response.status == 200
            return json.load(response)

    try:
        assert request("health")["status"] == "ok"
        request("reset", {"seed": 17})
        outputs, times, payloads = [], [], []
        for start, stop, image_index in ((0, 1, 0), (1, 26, 15), (26, 51, 30)):
            image = np.asarray(frames[image_index, 0])
            assert image.ndim == 3 and image.shape[-1] in (3, 4)
            payload = {"task": instruction, "observation": {
                "state_history": states[start:stop].tolist(), "images": {"front": {
                    "shape": list(image.shape), "dtype": str(image.dtype),
                    "data_b64": base64.b64encode(image.tobytes()).decode(),
                }},
            }}
            begin = time.monotonic()
            action = np.array(request("infer", payload)["action_chunk"])
            times.append(time.monotonic() - begin)
            assert action.shape == (25, 40) and np.isfinite(action).all(), action.shape
            outputs.append(action)
            payloads.append(payload)
        request("reset", {"seed": 17})
        repeated = np.array(request("infer", payloads[0])["action_chunk"])
        difference = float(np.abs(repeated - outputs[0]).max())
        assert difference < 1e-5, difference
        np.save(RUN / f"{controller}-actions.npy", np.stack(outputs))
        results.append({"controller": controller, "requests": 3, "action_shape": [25, 40],
                        "inference_seconds": times, "reset_max_difference": difference})
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        del server, policy
        gc.collect()
        torch.cuda.empty_cache()

with (checkpoint / "training_state.pt").open("rb") as stream:
    checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
report = {"passed": True, "checkpoint_sha256": checkpoint_hash, "checks": results,
          "source": "HOI_pp_box/sonic/000001", "meaning": "real GPU HTTP inference, not simulator task success"}
(RUN / "result.json").write_text(json.dumps(report, indent=2))
record["status"] = "completed"
(RUN / "run.json").write_text(json.dumps(record, indent=2))
print(json.dumps(report), flush=True)
