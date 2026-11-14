"""Install the complete resolved Linux dependency set with explicit NVIDIA wheel URLs."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
python = ROOT / "data/cache/envs/isaaclab-2.2/bin/python"
requirements = RUN / "sim-explicit.txt"
env = {**os.environ, "UV_CACHE_DIR": str(ROOT / "data/cache/uv"),
       "UV_HTTP_TIMEOUT": "30", "UV_HTTP_RETRIES": "3", "UV_CONCURRENT_DOWNLOADS": "4"}
env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1,pypi.tuna.tsinghua.edu.cn"
command = ["uv", "pip", "install", "--python", str(python), "--no-deps",
           "--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple", "-r", str(requirements)]
record = json.loads((RUN / "run.json").read_text())
record.update(status="installing_simulator", simulation_executed=False, formal_training_started=False)
(RUN / "run.json").write_text(json.dumps(record, indent=2))
start = time.time()
with (RUN / "install-r5-isaacsim.log").open("x") as log:
    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
receipt = {"command": command, "exit_code": result.returncode,
           "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
           "started_unix": start, "finished_unix": time.time()}
(RUN / "install-r5-isaacsim.json").write_text(json.dumps(receipt, indent=2))
if result.returncode == 0:
    with (RUN / "sim-dependency-check.txt").open("x") as log:
        check = subprocess.run(["uv", "pip", "check", "--python", str(python)],
                               stdout=log, stderr=subprocess.STDOUT, env=env)
    receipt["dependency_check_exit_code"] = check.returncode
    (RUN / "install-r5-isaacsim.json").write_text(json.dumps(receipt, indent=2))
record["status"] = "sim_packages_installed" if result.returncode == 0 else "failed"
(RUN / "run.json").write_text(json.dumps(record, indent=2))
print(record["status"], receipt, flush=True)
raise SystemExit(result.returncode)
