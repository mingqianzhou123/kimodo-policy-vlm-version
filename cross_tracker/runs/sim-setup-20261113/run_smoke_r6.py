"""Launch a bounded GPU-0 simulator diagnostic after installation succeeds."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
assert json.loads((RUN / "install-r3-lab-arena.json").read_text())["exit_code"] == 0
subprocess.run(["uv", "pip", "check", "--python", str(ROOT / "data/cache/envs/isaaclab-2.2/bin/python")], check=True)
gpu = subprocess.check_output([
    "nvidia-smi", "--id=0", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits",
], text=True).strip()
memory, utilization = map(int, gpu.split(","))
assert memory < 100 and utilization == 0, f"GPU 0 is occupied: {gpu}"
env = dict(os.environ)
env["CUDA_VISIBLE_DEVICES"] = "0"
env["LD_LIBRARY_PATH"] = str(ROOT / "data/cache/system-libs/usr/lib/x86_64-linux-gnu") + ":" + env.get("LD_LIBRARY_PATH", "")
env.update(ACCEPT_EULA="Y", OMNI_KIT_ACCEPT_EULA="YES",
           PYTHONUNBUFFERED="1")
for key, name in [("XDG_CACHE_HOME", "cache"), ("XDG_DATA_HOME", "data"),
                  ("XDG_CONFIG_HOME", "config"), ("CUDA_CACHE_PATH", "cuda")]:
    path = ROOT / "data/cache/sim-runtime" / name
    path.mkdir(parents=True, exist_ok=True)
    env[key] = str(path)
python = ROOT / "data/cache/envs/isaaclab-2.2/bin/python"
script = RUN / "smoke_sim_r6.py"
start = time.time()
with (RUN / "smoke-launch-r6.log").open("x") as log:
    try:
        result = subprocess.run([str(python), str(script)], env=env,
                                stdout=log, stderr=subprocess.STDOUT, timeout=900)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = 124
result_path = RUN / "smoke-result-r6.json"
passed = result_path.exists() and json.loads(result_path.read_text()).get("passed") is True
receipt = {"exit_code": code, "physics_and_rgb_passed": passed, "started_unix": start, "finished_unix": time.time(),
           "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
           "gpu_before": gpu, "formal_training_started": False}
(RUN / "smoke-launch-r6.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
raise SystemExit(code if code else (0 if passed else 1))
