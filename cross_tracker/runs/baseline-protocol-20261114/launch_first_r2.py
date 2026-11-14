"""Wait for this archive audit, check the frozen first run, then launch it once."""
from pathlib import Path
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/baseline-sonic-s42-20261114-r2"
audit = ROOT / "runs/archive-audit-20261114/result.json"
deadline = time.monotonic() + 7200
while not audit.exists():
    assert time.monotonic() < deadline, "Archive audit did not finish within two hours"
    time.sleep(10)
assert json.loads(audit.read_text())["status"] == "completed", "Archive audit failed"
python = str(ROOT.parent / ".venv/bin/python")
command = [python, "-u", str(ROOT / "scripts/train_baseline.py"), RUN.name, "--gpu", "0"]
env = {**os.environ, "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"}
subprocess.run(command + ["--check-only"], cwd=ROOT.parent, env=env, check=True)
with (RUN / "environment.txt").open("x") as output:
    subprocess.run([python, "-c", "import importlib.metadata as m; print(chr(10).join(sorted(d.metadata['Name'] + '==' + d.version for d in m.distributions())))"], stdout=output, check=True)
with (RUN / "train.log").open("x") as output:
    process = subprocess.Popen(command, cwd=ROOT.parent, env=env, stdin=subprocess.DEVNULL,
                               stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
receipt = {"pid": process.pid, "command": command, "started_unix": time.time(),
           "status": "launched_pending_first_metric", "gpu": 0}
(RUN / "process.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
