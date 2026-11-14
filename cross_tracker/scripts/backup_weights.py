"""Copy completed cross-tracker checkpoints outside runs, verifying source and destination."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def backup(source, target):
    names = ("training_state.pt", "config.json")
    if not all((source / name).is_file() for name in names):
        raise ValueError(f"Incomplete checkpoint: {source}")
    before = {name: digest(source / name) for name in names}
    if target.exists():
        receipt = json.loads((target / "backup.json").read_text())
        for name in names:
            expected = receipt["files"][name]["sha256"]
            if before[name] != expected or digest(target / name) != expected:
                raise ValueError(f"Existing backup differs: {target}/{name}")
        return {"path": str(target), "status": "already_verified"}
    temporary = target.with_name(target.name + ".partial")
    temporary.mkdir(parents=True, exist_ok=False)
    for name in names:
        shutil.copyfile(source / name, temporary / name)
        if digest(temporary / name) != before[name] or digest(source / name) != before[name]:
            raise ValueError(f"Checkpoint changed during backup: {source}")
        if (source / name).stat().st_ino == (temporary / name).stat().st_ino:
            raise ValueError("Backup must be an independent copy")
    report = {"source": str(source), "verified_utc": datetime.now(timezone.utc).isoformat(),
              "files": {name: {"sha256": before[name], "bytes": (source / name).stat().st_size} for name in names}}
    (temporary / "backup.json").write_text(json.dumps(report, indent=2)+"\n")
    temporary.rename(target)
    return {"path": str(target), "status": "copied_and_verified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True, help="Independent storage directory, never inside runs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = args.destination.resolve()
    if destination.is_relative_to(root):
        parser.error("Use a backup location outside cross_tracker")
    results, pending = [], []
    for run in sorted((root / "runs").iterdir()):
        if not run.is_dir() or not (run / "run.json").is_file():
            continue
        for source in sorted((run / "training").glob("checkpoint_*")):
            if not (source / "config.json").is_file():
                pending.append(str(source))
                continue
            results.append(backup(source, destination / run.name / source.name))
    print(json.dumps({"verified_utc": datetime.now(timezone.utc).isoformat(),
                      "completed": results, "incomplete_skipped": pending,
                      "resume_snapshots": "not copied; mutable files require a separate stable snapshot"}, indent=2))


if __name__ == "__main__":
    main()
