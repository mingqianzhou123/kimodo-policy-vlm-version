"""Register an experiment and provenance without starting its command."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def capture_repo(repo, destination):
    """Snapshot tracked code and its diff, without copying data/checkpoints."""
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])
    head = git("rev-parse", "HEAD").decode().strip()
    status = git("status", "--short").decode()
    # Git archive preserves the committed sources. The binary patch preserves tracked changes.
    subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", "-o",
                    str(destination / "dependency.tar"), "HEAD"], check=True)
    (destination / "dependency.patch").write_bytes(git("diff", "HEAD", "--binary"))
    (destination / "dependency-status.txt").write_text(status)
    return {"path": str(repo), "head": head, "status": status,
            "untracked_files_not_archived": True}


def register(args):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{2,95}", args.run_id):
        raise ValueError("Use a short run ID containing letters, digits, '_' or '-'")
    config, manifest = json.loads(args.config.read_text()), json.loads(args.manifest.read_text())
    diagnostic_sources = (args.stage == "diagnostic" and len(args.sources) == len(set(args.sources))
                          and set(args.sources) <= set(config.get("trackers", args.sources)))
    if not diagnostic_sources and sorted(args.sources) not in [sorted(group) for group in config["source_groups"]]:
        raise ValueError("Source group is not in the protocol")
    data_keys = ("tasks", "trackers", "subset_seed", "validation_fraction",
                 "action_chunk_frames", "observation_history_frames")
    if any(manifest["protocol"].get(key) != config.get(key) for key in data_keys):
        raise ValueError("Data settings differ from the frozen manifest; freeze a new version first")
    if not args.command:
        raise ValueError("Record the intended command, even for a planned run")
    run = ROOT / "runs" / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    try:
        for name in ("checkpoints", "eval", "provenance"):
            (run / name).mkdir()
        shutil.copyfile(args.config, run / "config.json")
        shutil.copyfile(args.manifest, run / "manifest.json")
        for name in ("scripts", "tests", "src"):
            if not (ROOT / name).exists():
                continue
            shutil.copytree(ROOT / name, run / "provenance" / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        dependency = capture_repo(args.code_repo.resolve(), run / "provenance")
        hashes = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in run.rglob("*") if p.is_file()}
        record = {"run_id": args.run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "stage": args.stage, "purpose": args.purpose, "platform": args.platform,
                  "seed": args.seed, "sources": args.sources, "command": args.command,
                  "status": "planned", "training_started": False, "dependency": dependency,
                  "sha256": hashes, "runtime_environment": None, "parent_run": args.parent_run}
        (run / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    except Exception:
        # Preserve the failed registration for inspection; never silently reuse its ID.
        (run / "REGISTRATION_FAILED").touch()
        raise
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--stage", required=True, choices=["diagnostic", "pilot", "baseline", "cct", "method"])
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--platform", required=True, choices=["local", "baidu", "sensecore"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--code-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/balanced-v1.json")
    parser.add_argument("--parent-run")
    parser.add_argument("--command", nargs=argparse.REMAINDER, required=True)
    print(register(parser.parse_args()))
