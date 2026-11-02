"""Run verified scratch training, then the original 60-episode DoubleDesk evaluation.

--detach keeps the experiment running after the terminal/session disconnects.
The status receipt and logs are written under outputs/, including failures.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from groot_experiment import atomic_json, check_preflight, fingerprint
from run_config import REPO, load_yaml


def training_command(config_path, run_dir, gpus):
    command = [sys.executable, "-u"]
    if len(gpus) > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                    f"--nproc_per_node={len(gpus)}"]
    return command + [str(REPO / "scripts/train.py"), "--config", str(config_path), "--run-dir", str(run_dir)]


def keepalive_receipt(script):
    """Keep the existing Supervisor configuration; never stop its monitor."""
    subprocess.run(["/usr/bin/python3", str(script), "ensure"], check=True, timeout=30, capture_output=True)
    result = subprocess.run(["/usr/bin/python3", str(script), "status"], check=True,
                            timeout=30, capture_output=True, text=True)
    if "RUNNING" not in result.stdout or "已启用" not in result.stdout:
        raise RuntimeError(f"xiancun is not enabled and monitored: {result.stdout}")
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "supervised": True, "status": result.stdout}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/arena_doubledesk_groot.yaml")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--gpus", help="comma-separated physical GPU IDs for DDP; --gpu remains the evaluation GPU")
    parser.add_argument("--distributed-validation", type=Path, default=REPO / "outputs/groot_ddp_preflight.json")
    parser.add_argument("--keepalive", type=Path, help="existing xiancun script, verified before and after the experiment")
    parser.add_argument("--smoke-validation", type=Path, default=REPO / "outputs/groot_sim_smoke_v2/validation.json")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    gpus = [int(v) for v in args.gpus.split(",")] if args.gpus else [args.gpu]
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError("Expected distinct nonnegative GPU IDs")
    if config.main.batch_size % (config.main.micro_batch_size * len(gpus)):
        raise ValueError("Global batch must be divisible by micro batch times GPU count")
    check_preflight(config)
    if len(gpus) > 1:
        ddp = json.loads(args.distributed_validation.read_text())
        if ddp.get("passed") is not True or ddp.get("fingerprint") != fingerprint(config) or ddp.get("gpus") != gpus:
            raise RuntimeError("Distributed stage transition and resume checks must pass for these GPUs/config")
    smoke = json.loads(args.smoke_validation.read_text())
    if smoke.get("valid_inference") is not True or smoke.get("successful_inferences", 0) < 1:
        raise RuntimeError("The real simulator smoke has not passed")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = (args.run_dir or REPO / config.main.save_root / f"scratch_{stamp}").resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite a training run: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    outputs = REPO / "outputs"
    outputs.mkdir(exist_ok=True)
    status_file = outputs / f"groot_{run_dir.name}_status.json"
    if args.detach:
        log = outputs / f"groot_{run_dir.name}_pipeline.log"
        command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
                   "--run-dir", str(run_dir), "--gpu", str(args.gpu),
                   "--smoke-validation", str(args.smoke_validation.resolve())]
        if args.gpus:
            command += ["--gpus", args.gpus, "--distributed-validation", str(args.distributed_validation.resolve())]
        if args.keepalive:
            command += ["--keepalive", str(args.keepalive.resolve())]
        with log.open("x") as handle:
            process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
        print(json.dumps({"pid": process.pid, "run_dir": str(run_dir), "status": str(status_file),
                          "pipeline_log": str(log)}, indent=2))
        return
    training = training_command(args.config.resolve(), run_dir, gpus)
    evaluation_dir = run_dir / "doubledesk_formal"
    evaluation = [sys.executable, "-u", str(REPO / "scripts/eval_arena_doubledesk.py"),
                  "--checkpoint", str(run_dir / f"checkpoint_{config.main.max_steps}"),
                  "--out", str(evaluation_dir), "--gpu", str(args.gpu)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
               PYTHONPATH=str(REPO) + os.pathsep + env.get("PYTHONPATH", ""),
               NO_ALBUMENTATIONS_UPDATE="1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    preflight = json.loads(Path(config.main.preflight_report).read_text())
    atomic_json(outputs / f"groot_{run_dir.name}_preflight.json", preflight)
    status = {"pid": os.getpid(), "gpu": args.gpu, "gpus": gpus, "world_size": len(gpus), "run_dir": str(run_dir),
              "started_at": datetime.now(timezone.utc).isoformat(), "phase": "training",
              "training_command": training, "evaluation_command": evaluation,
              "preflight_passed": True, "simulator_smoke_passed": True,
              "max_steps": config.main.max_steps, "global_batch": config.main.batch_size,
              "micro_batch": config.main.micro_batch_size, "scratch_no_419h": True}
    if args.keepalive:
        status["keepalive_before"] = keepalive_receipt(args.keepalive)
    atomic_json(status_file, status)
    try:
        subprocess.run(training, cwd=REPO, env=env, check=True)
        # An intentional checkpoint-and-stop must not launch final evaluation.
        if not (run_dir / f"checkpoint_{config.main.max_steps}/training_state.pt").is_file():
            status["phase"] = "stopped_with_checkpoint"
            return
        if args.keepalive:
            status["keepalive_after_training"] = keepalive_receipt(args.keepalive)
        status["phase"] = "evaluation"
        atomic_json(status_file, status)
        subprocess.run(evaluation, cwd=REPO, env=env, check=True)
        measured = json.loads((evaluation_dir / "summary.json").read_text())
        summary = {"measured_success_rate": measured["overall_success_rate"],
                   "total_successes": measured["total_successes"], "total_episodes": measured["total_episodes"],
                   "paper_reference_success_rate": 0.4, "matched_local_single_task_baseline": None,
                   "delta_vs_paper_reference": measured["overall_success_rate"] - 0.4}
        atomic_json(run_dir / "experiment_summary.json", summary)
        status["phase"] = "complete"
    except Exception as exc:
        status["phase"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["traceback"] = traceback.format_exc()
        raise
    finally:
        if args.keepalive:
            try:
                status["keepalive_after"] = keepalive_receipt(args.keepalive)
            except Exception as exc:
                status["keepalive_error"] = str(exc)
                status["phase"] = "keepalive_recovery_failed"
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(status_file, status)


if __name__ == "__main__":
    main()
