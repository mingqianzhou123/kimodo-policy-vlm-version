"""Evaluate one fixed final baseline on one tracker; each invocation has a new run ID."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
SETUP = ROOT / "runs/eval-setup-20261116"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("sonic", "twist2", "sonic-twist2"))
    parser.add_argument("controller", choices=("sonic", "twist2"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sim-gpu", type=int, default=1)
    parser.add_argument("--policy-gpu", type=int, default=2)
    parser.add_argument("--diagnostic", action="store_true", help="One 50-step episode; never counted as formal")
    args = parser.parse_args()
    plan = json.loads((SETUP / "evaluation-plan.json").read_text())
    model_id = f"box-{args.model}-s42-20261114"
    model = next(m for m in plan["models_previously_verified_complete"] if m["run_id"] == model_id)
    checkpoint = CACHE / "eval-checkpoints" / model_id / f"checkpoint_{model['step']}"
    for name, item in model["files"].items():
        assert sha(checkpoint / name) == item["sha256"], name
    run = (ROOT / "runs" / args.run_id).resolve()
    assert run.parent == ROOT / "runs" and args.sim_gpu != args.policy_gpu
    for device in (args.sim_gpu, args.policy_gpu):
        values = subprocess.check_output(["nvidia-smi", f"--id={device}",
            "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        memory, utilization = map(int, values.split(","))
        assert memory < 100 and utilization == 0, (device, values)
    run.mkdir(exist_ok=False)
    seeds = [1000] if args.diagnostic else plan["scene_seeds"]
    steps = 50 if args.diagnostic else plan["max_steps"]
    episodes = [{"seed": seed, "episode_seed": seed, "repeat_idx": 0, "episode_index": i,
        "result_json": str(run / f"episode-{seed}.json"), "max_steps": steps,
        "success_video_dir": str(run / "videos"), "failure_video_dir": str(run / "videos"),
        "recording_save_dir": str(run / "recordings"), "video_fps": 30,
        "post_termination_record_steps": 0, "model_label": model_id,
        "eval_model_path": str(checkpoint)} for i, seed in enumerate(seeds)]
    (run / "episodes.json").write_text(json.dumps({"episodes": episodes}, indent=2))
    source = CACHE / "sources/HumanoidArena-44a72229f11bab47a145e12db39ed891b677e783"
    arena = source / "isaaclab_twist2_g1"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    server_command = [sys.executable, "-u", str(ROOT.parent / "scripts/serve_arena.py"),
        "--checkpoint", str(checkpoint), "--text-cache", str(CACHE / "text-eval"),
        "--controller", args.controller, "--port", str(port), "--dtype", "fp32",
        "--execution-frames", "15", "--diffusion-steps", "10", "--rtc-overlap-frames", "0", "--timing", "exact"]
    env = {k: v for k, v in os.environ.items()
           if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")}
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4")
    server_env = {**env, "CUDA_VISIBLE_DEVICES": str(args.policy_gpu), "HF_HUB_OFFLINE": "1",
                  "PYTHONPATH": str(ROOT.parent)}
    runtime = CACHE / "sim-runtime" / args.run_id
    sim_env = {**env, "CUDA_VISIBLE_DEVICES": str(args.sim_gpu), "PYTHONPATH": str(arena),
               "ACCEPT_EULA": "Y", "OMNI_KIT_ACCEPT_EULA": "YES",
               "CYCLONEDDS_HOME": str(CACHE / "cyclonedds-0.10.2"),
               "CROSS_TRACKER_HISTORY_TRACE": str(run / "state-history.jsonl")}
    sim_env["LD_LIBRARY_PATH"] = str(CACHE / "system-libs/usr/lib/x86_64-linux-gnu") + ":" + env.get("LD_LIBRARY_PATH", "")
    for key, name in (("XDG_CACHE_HOME", "cache"), ("XDG_DATA_HOME", "data"),
                      ("XDG_CONFIG_HOME", "config"), ("CUDA_CACHE_PATH", "cuda")):
        sim_env[key] = str(runtime / name)
    bootstrap = ROOT / "runs/sim-setup-20261113/arena_sim_bootstrap.py"
    sim_command = [str(CACHE / "envs/isaaclab-2.2/bin/python"), "-u", str(bootstrap),
        str(arena / f"script/eval_scripts/{args.controller}/sim_eval_vla.py"),
        "--headless", "--device", "cuda:0", "--kit_args", f"--allow-root --portable-root {runtime}",
        "--task", "Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody",
        "--env_config_yaml", str(arena / f"tasks/common_env_config/pickplace_box_{args.controller}.yaml"),
        "--seed", "1000", "--episode_seed", "1000", "--max_steps", str(steps),
        "--episode_batch_json", str(run / "episodes.json"), "--lerobot_server_url", url,
        "--lerobot_server_timeout", "120", "--step_log_every_n", "100"]
    if args.controller == "sonic":
        for kind in ("encoder", "decoder"):
            sim_command += [f"--sonic_{kind}_path", str(CACHE / f"sim-assets/sonic/model_{kind}.onnx")]
    else:
        sim_command += ["--model_path", str(source / "TWIST2/assets/ckpts/twist2_1017_20k.onnx")]
    record = {"stage": "diagnostic" if args.diagnostic else "evaluation", "model": model_id,
        "controller": args.controller, "seeds": seeds, "max_steps": steps,
        "checkpoint_files": model["files"], "server_command": server_command, "sim_command": sim_command,
        "started_unix": time.time(), "status": "running", "valid_for_success_rate": False,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    for path in (Path(__file__), bootstrap, ROOT / "src/arena_history.py", ROOT.parent / "scripts/serve_arena.py"):
        (run / path.name).write_bytes(path.read_bytes())
    (run / "run.json").write_text(json.dumps(record, indent=2))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with (run / "server.log").open("x") as server_log, (run / "simulation.log").open("x") as sim_log:
        server = subprocess.Popen(server_command, env=server_env, cwd=ROOT.parent, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
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
                stdout=sim_log, stderr=subprocess.STDOUT, timeout=1800 if args.diagnostic else 21600)
            record["simulator_exit_code"] = completed.returncode
            completed.check_returncode()
            log = (run / "server.log").read_text()
            assert not re.search(r'POST /(?:infer|reset) HTTP/1.1" [45]\d\d', log), "HTTP failures"
            assert log.count('"POST /infer HTTP/1.1" 200') >= 2 * len(seeds)
            assert log.count('"POST /reset HTTP/1.1" 200') >= len(seeds)
            results = []
            import av
            for spec in episodes:
                episode = json.loads(Path(spec["result_json"]).read_text())
                assert not episode.get("error"), episode
                assert episode["episode_seed"] == spec["episode_seed"]
                # Arena mixes the first reset counter into the supplied episode seed.
                assert episode["episode_object_seed_source"] == "env_seed"
                assert episode["episode_object_seed"] == (spec["episode_seed"] ^ 0x9E3779B185EBCA87)
                assert isinstance(episode["success"], bool) and 0 < episode["episode_steps"] <= steps
                if args.diagnostic:
                    assert episode["episode_steps"] == 50
                video = Path(episode["video_path"])
                with av.open(str(video)) as container:
                    frames = sum(1 for _ in container.decode(video=0))
                assert frames >= episode["episode_steps"], (frames, episode["episode_steps"])
                results.append({"seed": episode["episode_seed"], "success": episode["success"],
                    "object_seed": episode["episode_object_seed"],
                    "failure_reason": episode["failure_reason"], "steps": episode["episode_steps"],
                    "video_sha256": sha(video), "video_frames": frames})
            record.update(status="completed", episodes=results, valid_for_success_rate=not args.diagnostic)
            if not args.diagnostic:
                record["success_rate"] = sum(e["success"] for e in results) / len(results)
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
            (run / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
