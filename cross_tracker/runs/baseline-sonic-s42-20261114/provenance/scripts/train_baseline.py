"""Start one registered baseline with a frozen protocol and the shared Kimodo trainer."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "scripts"), str(ROOT.parent)]


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run = (ROOT / "runs" / args.run_id).resolve()
    assert run.parent == ROOT / "runs", "Invalid run directory"
    record, protocol = read(run / "run.json"), read(run / "config.json")
    assert record["stage"] == "baseline" and record["status"] == "planned"
    assert protocol["training_ready"] and record["seed"] in protocol["train_seeds"]
    assert sorted(record["sources"]) in [sorted(g) for g in protocol["source_groups"]]
    assert int(os.environ.get("WORLD_SIZE", 1)) == 1, "Use the frozen single-GPU recipe"
    for folder in ("scripts", "src", "configs"):
        for path in (ROOT / folder).iterdir():
            if path.suffix not in (".py", ".json", ".yaml") or path.name == "paths.local.json":
                continue
            assert path.read_bytes() == (run / "provenance" / folder / path.name).read_bytes(), path
    assert (run / "manifest.json").read_bytes() == (ROOT / "data/manifests/balanced-v1.json").read_bytes()
    assert subprocess.check_output(["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"], text=True).strip() == record["dependency"]["head"]
    assert subprocess.check_output(["git", "-C", str(ROOT.parent), "diff", "HEAD", "--binary"]) == (run / "provenance/dependency.patch").read_bytes()
    coverage = read(ROOT / "runs/readiness-20261113-r2/dataset-coverage.json")
    assert coverage["exact_frozen_coverage"] and coverage["episodes"] == 744
    assert read(ROOT / "runs/readiness-20261113-r2/full-data-check.json")["prepared_checks"] == 744
    assert read(ROOT / "runs/overfit-20261113/result.json")["passed"]
    assert read(ROOT / "runs/pilot-sonic-20261113/run.json")["status"] == "completed"
    assert read(ROOT / "runs/sim-setup-20261113/closed-loop-audit.json")["passed"]
    archive = read(ROOT / "runs/archive-audit-20261114/result.json")
    assert archive["status"] == "completed" and archive["xz_eof_verified"]
    from fetch_assets import matches
    relative = protocol["initial_checkpoint"]["path"]
    checkpoint = ROOT / "data/cache/weights" / relative
    weights = read(ROOT / "data/manifests/weights-v1.json")["files"]
    for name in ("config.json", "training_state.pt"):
        assert matches(checkpoint / name, weights[f"{relative}/{name}"]), name
    source_count = len(record["sources"])
    multiplier = protocol["dual_source_step_multiplier"] if source_count == 2 else 1
    steps = protocol["single_source_steps"] * multiplier
    score_steps = [n * multiplier for n in protocol["score_checkpoints_single"]]
    config = ROOT / "configs/kimodo_baseline.yaml"
    from run_config import load_yaml
    resolved = load_yaml(config)
    assert resolved.main.batch_size == protocol["batch_size"]
    assert resolved.main.micro_batch_size == protocol["micro_batch_size"]
    assert resolved.main.max_steps == protocol["single_source_steps"]
    assert list(resolved.main.score_checkpoints) == protocol["score_checkpoints_single"]
    argv = [str(ROOT.parent / "scripts/train.py"), "--config", str(config),
            "--run-dir", str(run / "training"), "--init-checkpoint", str(checkpoint),
            "--task", *[f"{t}/{s}" for t in protocol["tasks"] for s in record["sources"]],
            "--device", "cuda", f"main.seed={record['seed']}", f"main.max_steps={steps}",
            f"main.checkpoint_every={2000 * multiplier}",
            f"main.final_checkpoint_spacing={2000 * multiplier}",
            "main.score_checkpoints=" + json.dumps(score_steps, separators=(",", ":")),
            f"training.scheduler.warmup_steps={1000 * multiplier}"]
    gpu = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
        "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    uuid, memory, utilization = [v.strip() for v in gpu.split(",")]
    assert int(memory) < 100 and int(utilization) == 0, f"GPU is busy: {gpu}"
    preflight = {"passed": True, "argv": argv, "gpu": gpu, "steps": steps,
                 "score_checkpoints": score_steps, "python": sys.version,
                 "platform": platform.platform(), "checked_utc": datetime.now(timezone.utc).isoformat()}
    (run / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    if args.check_only:
        print(json.dumps(preflight), flush=True)
        return
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4")
    os.chdir(ROOT.parent)
    import train as trainer
    from sampling import TrainingDataset, BalancedSampler
    trainer.KimodoWindowDataset = TrainingDataset
    trainer.EpisodeUniformSampler = BalancedSampler
    sys.argv = argv
    record.update(status="running", training_started=False, actual_trainer_argv=argv,
                  runtime_environment=preflight, pid=os.getpid(),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  baseline_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
    (run / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    try:
        trainer.main()
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        metrics = run / "training/metrics.jsonl"
        record["training_started"] = metrics.exists() and metrics.stat().st_size > 0
        record["ended_utc"] = datetime.now(timezone.utc).isoformat()
        (run / "run.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
