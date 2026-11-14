"""Restore Sim 5 compatibility after unconstrained optional Lab dependencies drifted."""
from pathlib import Path
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
python = str(ROOT / "data/cache/envs/isaaclab-2.2/bin/python")
env = {**os.environ, "UV_CACHE_DIR": str(ROOT / "data/cache/uv")}
env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1,pypi.tuna.tsinghua.edu.cn"
pins = ["packaging==23.0", "click==8.1.7", "idna==3.10", "wheel==0.44.0",
        "setuptools==80.9.0", "transformers==4.51.3", "huggingface-hub==0.33.1", "httpx==0.28.1"]
command = ["uv", "pip", "install", "--python", python,
           "--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple", *pins]
with (RUN / "dependency-repair.log").open("x") as log:
    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    # httpx 2 had installed httpx2; after restoring httpx 0.28 this orphan conflicts with Sim's idna.
    subprocess.run([python, "-c", "import importlib.metadata as m; "
        "assert not [(d.metadata['Name'],r) for d in m.distributions() "
        "for r in (d.requires or []) if r.lower().startswith('httpx2')], 'httpx2 still required'"],
        env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(["uv", "pip", "uninstall", "--python", python, "httpx2"],
                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    check = subprocess.run(["uv", "pip", "check", "--python", python], env=env,
                           stdout=log, stderr=subprocess.STDOUT)
with (RUN / "installed-packages-final.txt").open("x") as log:
    subprocess.run(["uv", "pip", "freeze", "--python", python], env=env,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
(RUN / "dependency-repair.json").write_text(json.dumps({
    "command": command, "dependency_check_exit_code": check.returncode,
    "finished_unix": time.time(),
}, indent=2))
print("FINAL_DEPENDENCY_CHECK", check.returncode, flush=True)
raise SystemExit(check.returncode)
