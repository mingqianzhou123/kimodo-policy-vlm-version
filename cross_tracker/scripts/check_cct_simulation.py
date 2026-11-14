"""Check a trained CCT checkpoint in 50 real box steps on an idle RTX host."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.request
import sys
import numpy as np
import av

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("controller", choices=("sonic", "twist2"))
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--sim-gpu", type=int, default=0)
parser.add_argument("--policy-gpu", type=int, default=1)
parser.add_argument("--run-id", required=True)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "runs/sim-setup-20261113"
parent = (ROOT / "runs" / args.run_id).resolve()
assert parent.parent == ROOT / "runs" and args.sim_gpu != args.policy_gpu
RUN = parent / args.controller
checkpoint = args.checkpoint.resolve()
expected = json.loads((ROOT / "runs/cct-ddp-pilot-20261114/result.json").read_text())
with (checkpoint / "training_state.pt").open("rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == expected["checkpoint_sha256"]
assert expected["passed"] and expected["world_size"] == 8
assert json.loads((ROOT / "runs/cct-gpu-checks-20261114/result.json").read_text())["passed"]
name = subprocess.check_output(["nvidia-smi", f"--id={args.sim_gpu}",
    "--query-gpu=name", "--format=csv,noheader"], text=True)
assert "RTX" in name or "L40" in name or "A40" in name, "Use a verified RTX-capable host"
RUN.mkdir(parents=True, exist_ok=False)
for device in (args.sim_gpu, args.policy_gpu):
    deadline = time.monotonic() + 15
    while True:
        gpu = subprocess.check_output(["nvidia-smi", f"--id={device}",
            "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        memory, utilization = map(int, gpu.split(","))
        if memory < 100 and utilization == 0:
            break
        assert time.monotonic() < deadline, (device, gpu)
        time.sleep(1)
source = ROOT / "data/cache/sources/HumanoidArena-44a72229f11bab47a145e12db39ed891b677e783"
arena = source / "isaaclab_twist2_g1"
task_id, config_name = "PickPlace-Box", "pickplace_box"
with socket.socket() as port_socket:
    port_socket.bind(("127.0.0.1", 0))
    port = port_socket.getsockname()[1]
url = f"http://127.0.0.1:{port}"
server_command = [sys.executable, "-u",
    str(ROOT / "scripts/serve_cct.py"), "--checkpoint", str(checkpoint),
    "--text-cache", str(ROOT / "data/cache/cct-text-eval"), "--controller", args.controller,
    "--port", str(port)]
env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4"}
server_env = {**env, "CUDA_VISIBLE_DEVICES": str(args.policy_gpu), "HF_HUB_OFFLINE": "1",
              "PYTHONPATH": str(ROOT.parent) + ":" + str(ROOT / "data/cache/python")}
runtime = ROOT / "data/cache/sim-runtime"
sim_env = {**env, "CUDA_VISIBLE_DEVICES": str(args.sim_gpu), "PYTHONPATH": str(arena),
           "ACCEPT_EULA": "Y", "OMNI_KIT_ACCEPT_EULA": "YES"}
sim_env["CROSS_TRACKER_HISTORY_TRACE"] = str(RUN / "state-history.jsonl")
sim_env["CYCLONEDDS_HOME"] = str(ROOT / "data/cache/cyclonedds-0.10.2")
sim_env["LD_LIBRARY_PATH"] = str(ROOT / "data/cache/system-libs/usr/lib/x86_64-linux-gnu") + ":" + env.get("LD_LIBRARY_PATH", "")
for key, folder in [("XDG_CACHE_HOME", "cache"), ("XDG_DATA_HOME", "data"),
                    ("XDG_CONFIG_HOME", "config"), ("CUDA_CACHE_PATH", "cuda")]:
    sim_env[key] = str(runtime / folder)
sim_command = [str(ROOT / "data/cache/envs/isaaclab-2.2/bin/python"), "-u",
    str(SETUP / "arena_sim_bootstrap.py"),
    str(arena / f"script/eval_scripts/{args.controller}/sim_eval_vla.py"),
    "--headless", "--device", "cuda:0", "--kit_args", f"--allow-root --portable-root {runtime}",
    "--task", f"Isaac-Move-{task_id}-G129-Dex3-Wholebody",
    "--env_config_yaml", str(arena / f"tasks/common_env_config/{config_name}_{args.controller}.yaml"),
    "--seed", "1000", "--episode_seed", "1000", "--max_steps", "50",
    "--lerobot_server_url", url, "--lerobot_server_timeout", "120",
    "--result_json", str(RUN / "episode.json"), "--verbose_startup",
    "--success_video_dir", str(RUN / "videos"), "--failure_video_dir", str(RUN / "videos"),
    "--recording_save_dir", str(RUN / "recordings"), "--step_log_every_n", "10"]
if args.controller == "sonic":
    for name in ("encoder", "decoder"):
        sim_command += [f"--sonic_{name}_path", str(ROOT / f"data/cache/sim-assets/sonic/model_{name}.onnx")]
else:
    sim_command += ["--model_path", str(source / "TWIST2/assets/ckpts/twist2_1017_20k.onnx")]
record = {"stage": "diagnostic", "controller": args.controller, "task": "box", "seed": 1000, "diffusion_steps": 10,
          "checkpoint_sha256": expected["checkpoint_sha256"],
          "context_manifest_sha256": hashlib.sha256((ROOT / "data/cache/cct-box-v1/manifest.json").read_bytes()).hexdigest(),
          "max_steps": 50, "server_command": server_command, "sim_command": sim_command,
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "status": "running", "formal_training_started": False, "started_unix": time.time()}
(RUN / "run.json").write_text(json.dumps(record, indent=2))
(RUN / "arena_history.py").write_bytes((ROOT / "src/arena_history.py").read_bytes())
for source_file in (Path(__file__), SETUP / "arena_sim_bootstrap.py", ROOT / "scripts/serve_cct.py"):
    (RUN / source_file.name).write_bytes(source_file.read_bytes())
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with (RUN / "server.log").open("x") as server_log, (RUN / "simulation.log").open("x") as sim_log:
    server = subprocess.Popen(server_command, env=server_env, cwd=ROOT.parent,
                              stdout=server_log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 120
        while True:
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Policy server did not become ready")
            try:
                with opener.open(url + "/health", timeout=2) as response:
                    assert json.load(response)["status"] == "ok"
                break
            except OSError:
                time.sleep(1)
        completed = subprocess.run(sim_command, env=sim_env, cwd=arena,
                                   stdout=sim_log, stderr=subprocess.STDOUT, timeout=1200)
        record["exit_code"] = completed.returncode
        assert completed.returncode == 0
        episode = json.loads((RUN / "episode.json").read_text())
        assert not episode.get("error") and episode["episode_steps"] == 50, episode
        inference_requests = (RUN / "server.log").read_text().count('"POST /infer HTTP/1.1" 200')
        assert inference_requests >= 2, "Need continuous successful model inference during simulation"
        assert '"POST /infer HTTP/1.1" 500' not in (RUN / "server.log").read_text()
        packets = [json.loads(line) for line in (RUN / "state-history.jsonl").read_text().splitlines()]
        assert len(packets) == inference_requests
        times = np.asarray([t for packet in packets for t in packet["times"]])
        np.testing.assert_allclose(np.diff(times), .02, atol=1e-6, rtol=0)
        assert all(np.isfinite(packet["states"]).all() for packet in packets)
        video = Path(episode["video_path"])
        with av.open(str(video)) as container:
            frames = sum(1 for _ in container.decode(video=0))
        assert frames == 51, frames
        record.update(video_frames=frames, video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                      history_packet_sizes=[len(p["times"]) for p in packets],
                      episode_object_seed=episode["episode_object_seed"], task_success=episode["success"], passed=True)
        record["successful_inference_requests"] = inference_requests
        record["episode_steps"] = episode["episode_steps"]
        record["failure_reason"] = episode["failure_reason"]
        record.update(status="completed", meaning="short closed-loop diagnostic, not task mastery")
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        record["finished_unix"] = time.time()
        (RUN / "run.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
