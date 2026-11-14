"""After CPU feature preparation exits, run remaining checks serially within the 16 GiB CCI."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
status = {"started_utc": datetime.now(timezone.utc).isoformat(), "formal_training_started": False, "stages": []}
try:
    process = Path("/proc/1401380")
    deadline = time.monotonic() + 3600
    while process.exists() and b"prepare_cct.py" in (process / "cmdline").read_bytes():
        if time.monotonic() > deadline:
            raise TimeoutError("Feature preparation exceeded follow-up wait budget")
        time.sleep(10)
    manifest = ROOT / "data/cache/cct-box-v1/manifest.json"
    assert json.loads(manifest.read_text())["complete"], "Context preparation did not complete"
    shutil.copyfile(manifest, RUN / "context-manifest.json")
    steps = [
        ("text-audit-attempt3", [sys.executable, str(ROOT / "scripts/audit_cct_text.py")]),
        ("cpu-model", [sys.executable, str(ROOT / "scripts/check_cct_model.py"), "cct-cpu-model-20261114", "--device", "cpu"]),
    ]
    for name, command in steps:
        with (RUN / (name + ".log")).open("x") as log:
            result = subprocess.run(command, cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT)
        status["stages"].append({"name": name, "command": command, "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}")
    status["passed"] = True
except Exception as error:
    status.update(passed=False, error=str(error))
finally:
    status["ended_utc"] = datetime.now(timezone.utc).isoformat()
    (RUN / "cpu-followup-result-r2.json").write_text(json.dumps(status, indent=2)+"\n")
    print(json.dumps(status), flush=True)
