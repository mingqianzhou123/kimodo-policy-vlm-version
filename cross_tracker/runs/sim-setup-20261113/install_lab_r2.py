"""Install pinned local Lab modules and Arena dependencies into the simulator venv."""
from pathlib import Path
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
python = ROOT / "data/cache/envs/isaaclab-2.2/bin/python"
lab = ROOT / "data/cache/sources/IsaacLab-2.2.0"
arena = ROOT / "data/cache/sources/HumanoidArena-44a72229f11bab47a145e12db39ed891b677e783"
assert json.loads((RUN / "install-r6-isaacsim.json").read_text())["exit_code"] == 0
constraints = RUN / "sim-constraints-r2.txt"
constraints.write_text("torch==2.7.0+cu128\ntorchvision==0.22.0+cu128\n"
                       "torchaudio==2.7.0+cu128\nnumpy==1.26.0\nisaacsim==5.0.0\nsetuptools==80.9.0\n")
env = {**os.environ, "UV_CACHE_DIR": str(ROOT / "data/cache/uv"),
       "UV_HTTP_TIMEOUT": "30", "UV_HTTP_RETRIES": "3"}
env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1,pypi.tuna.tsinghua.edu.cn"
base = ["uv", "pip", "install", "--python", str(python),
        "--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple"]
modules = [str(lab / "source" / name) for name in
           ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl")]
commands = [
    ("build-tools", base + ["toml", "wheel", "setuptools==80.9.0"]),
    ("lab-arena", base + ["--no-build-isolation", "--constraint", str(constraints),
                         "-r", str(arena / "isaaclab_twist2_g1/requirements.txt")]
                  + [arg for path in modules for arg in ("-e", path)]),
]
for name, command in commands:
    start = time.time()
    with (RUN / f"install-r2-{name}.log").open("x") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    (RUN / f"install-r2-{name}.json").write_text(json.dumps({
        "command": command, "exit_code": result.returncode,
        "started_unix": start, "finished_unix": time.time(),
    }, indent=2))
    print(name, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
for name, command in [
    ("installed-packages-r2.txt", ["uv", "pip", "freeze", "--python", str(python)]),
    ("dependency-check-r2.txt", ["uv", "pip", "check", "--python", str(python)]),
]:
    with (RUN / name).open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)
print("LAB_PACKAGES_INSTALLED; simulation not launched", flush=True)
