"""Install the pinned simulator using verified local NVIDIA wheels."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
assert json.loads((RUN / "sim-wheels-import.json").read_text())["verified_wheels"] == 26
python = ROOT / "data/cache/envs/isaaclab-2.2/bin/python"
requirements = RUN / "sim-local-wheels.txt"
lines = []
for line in (RUN / "sim-explicit.txt").read_text().splitlines():
    if " @ https://pypi.nvidia.com/" in line:
        url = line.split(" @ ", 1)[1].split("#", 1)[0]
        line = str(ROOT / "data/cache/sim-wheels" / url.rsplit("/", 1)[1])
    lines.append(line)
requirements.write_text("\n".join(lines) + "\n")
env = {**os.environ, "UV_CACHE_DIR": str(ROOT / "data/cache/uv"),
       "UV_HTTP_TIMEOUT": "30", "UV_HTTP_RETRIES": "3"}
env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1,pypi.tuna.tsinghua.edu.cn"
command = ["uv", "pip", "install", "--python", str(python), "--no-deps",
           "--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple", "-r", str(requirements)]
start = time.time()
with (RUN / "install-r6-isaacsim.log").open("x") as log:
    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
receipt = {"command": command, "exit_code": result.returncode,
           "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
           "started_unix": start, "finished_unix": time.time()}
if result.returncode == 0:
    with (RUN / "sim-dependency-check-r6.txt").open("x") as log:
        check = subprocess.run(["uv", "pip", "check", "--python", str(python)],
                               stdout=log, stderr=subprocess.STDOUT, env=env)
    receipt["dependency_check_exit_code"] = check.returncode
(RUN / "install-r6-isaacsim.json").write_text(json.dumps(receipt, indent=2))
record = json.loads((RUN / "run.json").read_text())
record.update(status="sim_packages_installed" if result.returncode == 0 else "failed",
              simulation_executed=False, formal_training_started=False)
if result.returncode == 0:
    record.pop("failed_stage", None)
(RUN / "run.json").write_text(json.dumps(record, indent=2))
print(record["status"], receipt, flush=True)
raise SystemExit(result.returncode)
