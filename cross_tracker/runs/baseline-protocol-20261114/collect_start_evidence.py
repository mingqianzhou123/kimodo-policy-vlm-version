"""Validate actual startup and bundle small evidence files, never model weights."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/baseline-sonic-s42-20261114-r2"
protocol = json.loads((RUN / "config.json").read_text())
config = json.loads((RUN / "training/config.json").read_text())
record = json.loads((RUN / "run.json").read_text())
rows = [json.loads(line) for line in (RUN / "training/metrics.jsonl").read_text().splitlines()]
assert record["status"] == "running" and record["seed"] == 42 and record["sources"] == ["sonic"]
assert rows[-1]["step"] >= 100
assert all(math.isfinite(value) for row in rows for value in row.values() if isinstance(value, (int, float)))
assert config["main"]["max_steps"] == protocol["single_source_steps"] == 30000
assert config["main"]["batch_size"] == 8 and config["main"]["micro_batch_size"] == 2
assert config["main"]["score_checkpoints"] == [26000, 28000, 30000]
assert config["main"]["seed"] == 42
assert sorted(config["main"]["task"]) == sorted(f"{t}/sonic" for t in protocol["tasks"])
assert config["initialization"]["checkpoint"].endswith(protocol["initial_checkpoint"]["path"])
assert "198 demos" in (RUN / "training/train.log").read_text()
process = json.loads((RUN / "process.json").read_text())
gpu_rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                    "--format=csv,noheader,nounits"], text=True).splitlines()
owned_gpu = [row for row in gpu_rows if row.split(",")[0].strip() == str(process["pid"])]
assert len(owned_gpu) == 1
assert record["runtime_environment"]["gpu"].split(",")[0] in owned_gpu[0]
record.update(training_started=True, verified_step=rows[-1]["step"])
(RUN / "run.json").write_text(json.dumps(record, indent=2) + "\n")
process.update(status="running", verified_step=rows[-1]["step"])
(RUN / "process.json").write_text(json.dumps(process, indent=2) + "\n")
result = {"passed": True, "checked_utc": datetime.now(timezone.utc).isoformat(),
          "stage": "baseline", "training_complete": False, "last_metric": rows[-1],
          "first_metric": rows[0], "training_episodes": 198, "gpu_process": owned_gpu[0]}
(RUN / "startup-check.json").write_text(json.dumps(result, indent=2) + "\n")
plan_path = ROOT / "runs/baseline-protocol-20261114/plan.json"
plan = json.loads(plan_path.read_text())
plan[0].update(status="running", run_id=RUN.name)
plan_path.write_text(json.dumps(plan, indent=2) + "\n")
archive_path = ROOT / "baseline-start-evidence-20261114.zip"
manifest = {}
with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as bundle:
    for folder in (ROOT / "runs/archive-audit-20261114", ROOT / "runs/baseline-protocol-20261114", ROOT / "runs/baseline-sonic-s42-20261114", RUN):
        for path in sorted(folder.rglob("*")):
            if path.suffix not in (".json", ".jsonl", ".py", ".yaml", ".txt", ".log", ".patch", ".tar"):
                continue
            if not path.is_file() or path.stat().st_size > 50_000_000:
                continue
            name, data = str(path.relative_to(ROOT)), path.read_bytes()
            bundle.writestr(name, data)
            manifest[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    bundle.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
print(json.dumps({"startup": result, "bundle": str(archive_path), "files": len(manifest),
                  "bytes": archive_path.stat().st_size,
                  "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()}, indent=2), flush=True)
