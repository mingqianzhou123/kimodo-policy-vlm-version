"""One independent scratch job: fixed-window check, DDP pilot, fresh formal run."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from backup_weights import backup
from run_box_screen import digest, read, write

ROOT = Path(__file__).resolve().parents[1]


def command(run, sources, pilot=False):
    steps = 10000 * len(sources)
    args = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
            str(ROOT / "scripts/train_worker.py"), "--config", str(ROOT / "configs/box_screen.yaml"),
            "--run-dir", str(run / "training"), "--task", *[f"HOI_pp_box/{s}" for s in sources],
            "--device", "cuda", f"main.max_steps={steps}", f"training.scheduler.warmup_steps={333*len(sources)}"]
    return args + (["main.stop_after=200", "main.log_every=10", "main.checkpoint_every=200", "main.score_checkpoints=[200]"]
                   if pilot else [f"main.checkpoint_every={steps//2}", f"main.score_checkpoints=[{steps}]"])


def preflight(run):
    assert run.parent == ROOT / "runs"
    plan = read(run / "plan.json")
    assert plan["initialization"] == "kimodo_default_without_pt419h"
    assert plan["sources"] in [["sonic"], ["twist2"], ["sonic", "twist2"]]
    assert plan["steps"] == 10000 * len(plan["sources"])
    assert read(run / "config.json")["initial_checkpoint"] is None
    for rel, sha in read(run / "source-files.json").items():
        assert digest(ROOT.parent / rel) == sha, rel
    for name, sha in plan["frozen_files"].items():
        assert digest(run / name) == sha, name
    assets = read(ROOT / "runs/box-screen-setup-20261114/assets-verified.json")
    assert assets["passed"] and assets["prepared_episodes"] == 108
    assert digest(ROOT / "data/manifests/balanced-v1.json") == assets["manifest_sha256"]
    for rel, sha in assets["files"].items():
        assert digest(ROOT / rel) == sha, rel
    for name in (plan["overfit"], plan["pilot"]):
        p = (ROOT / "runs" / name).resolve()
        assert p.parent == ROOT / "runs" and not p.exists(), name
    assert not (run / "training").exists()
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run = (ROOT / "runs" / args.run_id).resolve()
    plan = preflight(run)
    if args.check_only:
        print(json.dumps(dict(passed=True, training_started=False, run_id=run.name)))
        return
    import torch
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    assert len(gpus) == 8 and all("A800" in g for g in gpus), gpus
    record = read(run / "run.json")
    assert record["status"] in ("planned", "queued")
    env = dict(os.environ, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    record.update(status="checking", started_utc=datetime.now(timezone.utc).isoformat(), gpus=gpus)
    write(run / "run.json", record)

    def execute(name, argv, environment):
        with (run / f"{name}.log").open("x") as log:
            subprocess.run(argv, cwd=ROOT.parent, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)

    def verify(folder, steps):
        config = read(folder / "training/config.json")
        assert "initialization" not in config, "Policy pretraining unexpectedly loaded"
        assert config["main"]["batch_size"] == 8 and config["main"]["seed"] == 42
        rows = [json.loads(line) for line in (folder / "training/metrics.jsonl").read_text().splitlines()]
        assert rows[-1]["step"] == steps
        assert all(math.isfinite(row[k]) for row in rows for k in ("loss", "grad_control", "grad_hand"))
        checkpoint = folder / f"training/checkpoint_{steps}"
        payload = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
        assert payload["global_step"] == steps and all(torch.isfinite(v).all() for v in payload["model"].values())
        del payload
        return checkpoint

    try:
        execute("overfit", [sys.executable, str(ROOT / "scripts/check_scratch_model.py"), plan["overfit"],
                            "--sources", *plan["sources"]], dict(env, CUDA_VISIBLE_DEVICES="0"))
        assert read(ROOT / "runs" / plan["overfit"] / "result.json")["passed"]
        pilot = ROOT / "runs" / plan["pilot"]
        pilot.mkdir(exist_ok=False)
        write(pilot / "run.json", dict(stage="pilot", status="running", seed=42, formal_training=False))
        execute("pilot", command(pilot, plan["sources"], pilot=True), env)
        checkpoint = verify(pilot, 200)
        write(pilot / "run.json", dict(stage="pilot", status="completed", seed=42, formal_training=False))
        record["pilot_backup"] = backup(checkpoint, ROOT.parent.parent / "cross_tracker_backups/weights" / pilot.name / checkpoint.name)
        record.update(status="running", checks_passed=True, actual_command=command(run, plan["sources"]))
        write(run / "run.json", record)
        execute("formal", record["actual_command"], env)  # Fresh process/seed; never initialize from the pilot.
        checkpoint = verify(run, plan["steps"])
        record["backup"] = backup(checkpoint, ROOT.parent.parent / "cross_tracker_backups/weights" / run.name / checkpoint.name)
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        metrics = run / "training/metrics.jsonl"
        record.update(training_started=metrics.exists() and metrics.stat().st_size > 0,
                      ended_utc=datetime.now(timezone.utc).isoformat())
        write(run / "run.json", record)


if __name__ == "__main__":
    main()
