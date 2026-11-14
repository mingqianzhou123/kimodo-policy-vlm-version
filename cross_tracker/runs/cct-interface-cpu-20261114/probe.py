"""Real CCT checkpoint -> planner -> Arena HTTP interface, on held-out demonstration observations."""
import base64
import gc
import hashlib
import json
from pathlib import Path
import sys
import threading
import urllib.request
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent)]
from cct_eval import load_planner
from repaired_data import read_segment, checksum
from kimodo_vla.benchmarks.arena.policy import ROOT_BOUNDARY, make_policy, routes
from kimodo_vla.benchmarks.server import PLAIN_JSON, make_server

torch.set_num_threads(8)
checkpoint = ROOT / "runs/cct-cpu-model-20261114/checkpoint"
task_name = "Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody"
text_cache = ROOT / "data/cache/cct-text-eval"
text_cache.mkdir(exist_ok=False)
original = torch.load(ROOT / "data/cache/text/HOI_pp_box.pt", weights_only=True)
torch.save(dict(original, task_name=task_name), text_cache / "HOI_pp_box.pt")
assert torch.equal(torch.load(text_cache / "HOI_pp_box.pt", weights_only=True)["embedding"], original["embedding"])
manifest = json.loads((ROOT / "data/manifests/balanced-v1.json").read_text())
assets = json.loads((ROOT / "data/manifests/assets-v1.json").read_text())
roots = json.loads((ROOT / "runs/box-screen-setup-20261114/paths.json").read_text())
report = {"passed": False, "device": "cpu", "diffusion_steps": 1, "formal_training": False,
          "checkpoint_sha256": checksum(checkpoint / "training_state.pt"), "trackers": {}}
(RUN / "run.json").write_text(json.dumps(dict(status="running", **report), indent=2))

for tracker in ["sonic", "twist2"]:
    dataset = next(d for d in manifest["datasets"] if d["dataset"] == f"HOI_pp_box/{tracker}_refpose_v3_1")
    episode = next(e for e in dataset["episodes"] if e["split"] == "validation")
    state, _, mapping = read_segment(dataset, episode, roots, assets)
    frames = np.load(ROOT / f"data/cache/prepared-v1/validation/HOI_pp_box/{tracker}/episode_{episode['episode']:06d}.frames.npy", mmap_mode="r")
    assert len(state) > 125 and len(frames) > 75
    def request(left, right, index):
        image = np.asarray(frames[index, 0])
        return {"task": task_name, "observation": {"state_history": state[left:right].tolist(),
                "images": {"front": {"shape": list(image.shape), "dtype": "uint8",
                                    "data_b64": base64.b64encode(image.tobytes()).decode()}}}}
    planner = load_planner(checkpoint, tracker, text_cache, device="cpu", control_fps=50,
                           execution_frames=15, diffusion_steps=1, rtc_overlap_frames=0,
                           root_boundary=ROOT_BOUNDARY[tracker])
    policy = make_policy(planner)
    server = make_server(routes(policy), "127.0.0.1", 0, PLAIN_JSON)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def post(route, payload):
        req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/{route}",
                                     data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)
    try:
        post("reset", {"seed": 1000})
        first = np.asarray(post("infer", request(0, 101, 60))["action_chunk"])
        second = np.asarray(post("infer", request(101, 126, 75))["action_chunk"])
        post("reset", {"seed": 1000})
        repeated = np.asarray(post("infer", request(0, 101, 60))["action_chunk"])
        assert first.shape == second.shape == (25, 40)
        assert np.isfinite(first).all() and np.isfinite(second).all()
        delta = float(np.max(np.abs(first - repeated)))
        assert delta == 0, "Reset is not reproducible"
        report["trackers"][tracker] = {"validation_episode": episode["episode"], "shape": list(first.shape),
                                       "continuous_requests": 2, "reset_max_difference": delta}
    finally:
        server.shutdown(); server.server_close(); thread.join()
    del planner, policy, server
    gc.collect()
report.update(passed=True, meaning="Real HTTP/model interface only; no simulator rollout or task-success claim")
(RUN / "result.json").write_text(json.dumps(report, indent=2)+"\n")
(RUN / "run.json").write_text(json.dumps(dict(status="completed", **report), indent=2)+"\n")
print(json.dumps(report), flush=True)
