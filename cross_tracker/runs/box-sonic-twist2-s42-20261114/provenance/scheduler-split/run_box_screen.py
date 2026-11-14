"""Run exactly one registered box baseline in its own eight-GPU allocation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "runs/box-screen-setup-20261114"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def preflight(run_id):
    plan = read(SETUP / "plan.json")
    assert run_id in plan["run_ids"], "Unknown run; no other runs may be launched"
    for relative, expected in read(SETUP / "dependency-files.json").items():
        assert digest(ROOT.parent / relative) == expected, relative
    for relative, expected in read(SETUP / "deployment-files.json").items():
        assert digest(ROOT / relative) == expected, relative
    assets = read(SETUP / "assets-verified.json")
    assert assets["passed"] and assets["prepared_episodes"] == 108
    assert assets["manifest_sha256"] == digest(ROOT / "data/manifests/balanced-v1.json")
    # Recheck the actual inputs, not just an old success flag.
    for relative, expected in assets["files"].items():
        assert digest(ROOT / relative) == expected, relative
    run = ROOT / "runs" / run_id
    record = read(run / "run.json")
    assert record["platform"] == "sensecore" and record["seed"] == 42
    assert record["status"] in ("planned", "queued", "completed")
    for relative, expected in record["sha256"].items():
        assert digest(run / relative) == expected, (run_id, relative)
    return run_id


def command(run, protocol, sources):
    count = len(sources)
    steps = protocol["single_source_steps"] * count
    checkpoint = ROOT / "data/cache/weights" / protocol["initial_checkpoint"]["path"]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
            str(ROOT / "scripts/train_worker.py"), "--config", str(ROOT / "configs/box_screen.yaml"),
            "--run-dir", str(run / "training"), "--init-checkpoint", str(checkpoint),
            "--task", *[f"HOI_pp_box/{source}" for source in sources],
            "--device", "cuda",
            f"main.max_steps={steps}", f"main.checkpoint_every={steps // 2}",
            f"main.score_checkpoints=[{steps}]", f"training.scheduler.warmup_steps={333 * count}"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Exactly one registered experiment; required")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run_id = preflight(args.run_id)
    if args.check_only:
        print(json.dumps({"passed": True, "run_id": run_id}), flush=True)
        return
    import torch
    assert torch.cuda.device_count() == 8, "This submission requires exactly eight allocated GPUs"
    gpu_names = [torch.cuda.get_device_name(i) for i in range(8)]
    assert all("A800" in name for name in gpu_names), gpu_names
    os.chdir(ROOT.parent)
    environment = dict(os.environ, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    run = ROOT / "runs" / run_id
    record, protocol = read(run / "run.json"), read(run / "config.json")
    if record["status"] == "completed":
        return
    argv = command(run, protocol, record["sources"])
    record.update(status="running", actual_command=argv, started_utc=datetime.now(timezone.utc).isoformat(),
                  runtime_environment={"python": sys.version, "torch": torch.__version__, "gpus": gpu_names})
    write(run / "run.json", record)
    try:
        with (run / "launcher.log").open("x") as log:
            subprocess.run(argv, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(line) for line in (run / "training/metrics.jsonl").read_text().splitlines()]
        target = protocol["single_source_steps"] * len(record["sources"])
        assert rows[-1]["step"] == target and all(math.isfinite(row["loss"]) for row in rows)
        assert (run / f"training/checkpoint_{target}/training_state.pt").is_file()
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        metrics = run / "training/metrics.jsonl"
        record.update(training_started=metrics.exists() and metrics.stat().st_size > 0,
                      ended_utc=datetime.now(timezone.utc).isoformat())
        write(run / "run.json", record)
    print(f"COMPLETED {run_id}", flush=True)


if __name__ == "__main__":
    main()
