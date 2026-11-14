"""Run a registered Kimodo pilot with repaired data and balanced sampling."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "scripts"), str(ROOT.parent)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/cache/prepared-v1/train")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    run = (ROOT / "runs" / args.run_id).resolve()
    if run.parent != ROOT / "runs" or not (run / "run.json").is_file():
        parser.error("Register a unique run first with register_run.py")
    record = json.loads((run / "run.json").read_text())
    if record["stage"] not in ("diagnostic", "pilot") or record["status"] != "planned":
        parser.error("This entry only starts a fresh diagnostic/pilot, never a formal baseline")
    if not 1 <= args.steps <= 2000 or int(os.environ.get("WORLD_SIZE", 1)) != 1:
        parser.error("Pilot verification currently supports one GPU and at most 2000 updates")
    if not args.data_root.resolve().is_relative_to(ROOT / "data/cache"):
        parser.error("Prepared data must stay in cross_tracker/data/cache")
    protocol = json.loads((run / "config.json").read_text())
    tasks = args.tasks or list(protocol["tasks"])
    if not set(tasks) <= set(protocol["tasks"]):
        parser.error("Task is not in the frozen protocol")
    # Refuse silently running code/data changed since registration.
    for folder in ("scripts", "src"):
        for path in (ROOT / folder).glob("*.py"):
            archived = run / "provenance" / folder / path.name
            if not archived.exists() or archived.read_bytes() != path.read_bytes():
                parser.error("Cross-tracker code changed after registration; register a new run")
    if (run / "manifest.json").read_bytes() != (ROOT / "data/manifests/balanced-v1.json").read_bytes():
        parser.error("The frozen data manifest changed after registration")
    from fetch_assets import matches
    weights = json.loads((ROOT / "data/manifests/weights-v1.json").read_text())["files"]
    relative = protocol["initial_checkpoint"]["path"]
    checkpoint = ROOT / "data/cache/weights" / relative
    for name in ("config.json", "training_state.pt"):
        path = checkpoint / name
        if not path.is_file() or not matches(path, weights[f"{relative}/{name}"]):
            parser.error(f"Pinned pretrained checkpoint missing or incomplete: {path}")
    import train as trainer
    from sampling import TrainingDataset, BalancedSampler
    # Process-local adapters retain the shared optimizer, model and training loop.
    trainer.KimodoWindowDataset = TrainingDataset
    trainer.EpisodeUniformSampler = BalancedSampler
    config = ROOT / "configs/kimodo_pilot.yaml"
    shutil.copyfile(config, run / "provenance/kimodo_pilot.yaml")
    sys.argv = [str(Path(trainer.__file__)), "--config", str(config),
                "--run-dir", str(run / "training"), "--init-checkpoint", str(checkpoint),
                "--task", *[f"{task}/{source}" for task in tasks for source in record["sources"]],
                "--device", "cuda", f"main.data_root={args.data_root.resolve()}",
                f"main.seed={record['seed']}", f"main.max_steps={args.steps}"]
    record.update(status="running", training_started=False, actual_trainer_argv=sys.argv,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  pilot_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
    (run / "run.json").write_text(json.dumps(record, indent=2)+"\n")
    try:
        trainer.main()
        record["status"] = "completed"
    except BaseException:
        record["status"] = "failed"
        raise
    finally:
        metrics = run / "training/metrics.jsonl"
        record["training_started"] = metrics.exists() and metrics.stat().st_size > 0
        record["ended_utc"] = datetime.now(timezone.utc).isoformat()
        (run / "run.json").write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    main()
