"""Run 50 diagnostic task steps with the saved pilot and one controller."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("controller", choices=("sonic", "twist2"))
parser.add_argument("--task", choices=("box", "boxing", "navigation"), default="box")
parser.add_argument("--run-id", default="sim-closed-loop-20261113")
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[2]
SETUP = Path(__file__).resolve().parent
RUN = ROOT / "runs" / args.run_id / (args.controller if args.task == "box" else f"{args.controller}-{args.task}")
RUN.mkdir(parents=True, exist_ok=False)
assert json.loads((SETUP / "smoke-result-r7.json").read_text())["passed"]
for device in (0, 1):
    gpu = subprocess.check_output(["nvidia-smi", f"--id={device}",
        "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    memory, utilization = map(int, gpu.split(","))
    assert memory < 100 and utilization == 0, (device, gpu)
source = ROOT / "data/cache/sources/HumanoidArena-44a72229f11bab47a145e12db39ed891b677e783"
arena = source / "isaaclab_twist2_g1"
tasks = {
    "box": ("PickPlace-Box", "pickplace_box"),
    "boxing": ("Boxing-Bag", "boxing_bag"),
    "navigation": ("SmallWarehouse-VisionNavigation", "small_warehouse_vision_navigation"),
}
task_id, config_name = tasks[args.task]
checkpoint = ROOT / "runs/pilot-sonic-20261113/training/checkpoint_200"
with socket.socket() as port_socket:
    port_socket.bind(("127.0.0.1", 0))
    port = port_socket.getsockname()[1]
url = f"http://127.0.0.1:{port}"
server_command = [str(ROOT.parent / ".venv/bin/python"), "-u",
    str(ROOT.parent / "scripts/serve_arena.py"), "--checkpoint", str(checkpoint),
    "--text-cache", str(ROOT / "data/cache/text-eval"), "--controller", args.controller,
    "--port", str(port)]
env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4"}
server_env = {**env, "CUDA_VISIBLE_DEVICES": "1", "HF_HUB_OFFLINE": "1",
              "PYTHONPATH": str(ROOT.parent) + ":" + str(ROOT / "data/cache/python")}
runtime = ROOT / "data/cache/sim-runtime"
sim_env = {**env, "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(arena),
           "ACCEPT_EULA": "Y", "OMNI_KIT_ACCEPT_EULA": "YES"}
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
    "--seed", "17", "--episode_seed", "17", "--max_steps", "50",
    "--lerobot_server_url", url, "--lerobot_server_timeout", "120",
    "--result_json", str(RUN / "episode.json"), "--verbose_startup",
    "--success_video_dir", str(RUN / "videos"), "--failure_video_dir", str(RUN / "videos"),
    "--recording_save_dir", str(RUN / "recordings"), "--step_log_every_n", "10"]
if args.controller == "sonic":
    for name in ("encoder", "decoder"):
        sim_command += [f"--sonic_{name}_path", str(ROOT / f"data/cache/sim-assets/sonic/model_{name}.onnx")]
else:
    sim_command += ["--model_path", str(source / "TWIST2/assets/ckpts/twist2_1017_20k.onnx")]
record = {"stage": "diagnostic", "controller": args.controller, "task": args.task, "seed": 17,
          "max_steps": 50, "server_command": server_command, "sim_command": sim_command,
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "status": "running", "formal_training_started": False, "started_unix": time.time()}
(RUN / "run.json").write_text(json.dumps(record, indent=2))
for name in (Path(__file__).name, "arena_sim_bootstrap.py"):
    (RUN / name).write_text((SETUP / name).read_text())
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
        assert not episode.get("error") and episode["episode_steps"] > 0, episode
        inference_requests = (RUN / "server.log").read_text().count('"POST /infer HTTP/1.1" 200')
        assert inference_requests > 0, "No successful model inference during simulation"
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
