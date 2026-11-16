"""Install pinned Lab/DDS adapters after the base simulator packages have finished."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tarfile
import time
import zipfile

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
CACHE = ROOT / "data/cache"
OLD = ROOT / "runs/sim-setup-20261113"
env = {k: v for k, v in os.environ.items()
       if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")}
env.update(UV_CACHE_DIR=str(CACHE / "uv"), UV_HTTP_TIMEOUT="60", UV_HTTP_RETRIES="3",
           CYCLONEDDS_HOME=str(CACHE / "cyclonedds-0.10.2"))
python = CACHE / "envs/isaaclab-2.2/bin/python"
records = []


def run(name, command):
    with (RUN / f"{name}.log").open("x") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    records.append({"name": name, "command": list(map(str, command)), "exit_code": result.returncode})
    (RUN / "sim-adapters.json").write_text(json.dumps(records, indent=2) + "\n")
    print(name, result.returncode, flush=True)
    result.check_returncode()


if __name__ == "__main__":
    # No package mutations while the original installer still owns this environment.
    deadline = time.monotonic() + 7200
    pid = json.loads((RUN / "sim-install-start.json").read_text())["pid"]
    while Path(f"/proc/{pid}/cmdline").exists():
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        if b"uv\x00pip\x00install" not in command:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Base simulator installation still running")
        time.sleep(10)
    run("sim-base-check", [python, "-c", "import importlib.metadata as m; assert m.version('isaacsim')=='5.0.0.0'; assert m.version('torch')=='2.7.0+cu128'"])
    sources = CACHE / "sources"
    bundle = CACHE / "sim-dds-sources.zip"
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == "90b9172adb48c54679d135e638dc70b59944f1fd35b7b442e4d7b4e161fa2958"
    store = CACHE / "setup/dds"
    store.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(bundle) as archive:
        assert all(Path(n).name == n for n in archive.namelist())
        archive.extractall(store)
    for name in ("unitree-sdk-source.json", "cyclonedds-source.json"):
        item = json.loads((OLD / name).read_text())
        archive = store / item["archive"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == item["sha256"]
        with tarfile.open(archive) as content:
            content.extractall(sources, filter="data")
    dds = sources / "cyclonedds-9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd"
    run("dds-configure", ["cmake", "-S", dds, "-B", dds / "build",
        f"-DCMAKE_INSTALL_PREFIX={env['CYCLONEDDS_HOME']}", "-DBUILD_TESTING=OFF", "-DBUILD_EXAMPLES=OFF"])
    run("dds-build", ["cmake", "--build", dds / "build", "--target", "install", "-j", "8"])
    base = ["uv", "pip", "install", "--python", python, "--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple"]
    run("sim-build-tools", base + ["setuptools==80.9.0", "wheel==0.44.0", "toml"])
    modules = [sources / "IsaacLab-2.2.0/source" / name
               for name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl")]
    modules.append(sources / "unitree_sdk2_python-814556d15970dd2ecf1c9984e845ca02ab07e206")
    constraints = RUN / "sim-adapter-constraints.txt"
    constraints.write_text((OLD / "sim-constraints-r3.txt").read_text() + "warp-lang==1.9.0\nonnxruntime-gpu==1.22.0\n")
    run("sim-lab-sdk", base + ["--no-build-isolation", "-c", constraints,
        "warp-lang==1.9.0", "onnxruntime-gpu==1.22.0", "onnx", "rerun-sdk==0.20.1",
        "pyzmq==27.0.0", "logging_mp==0.1.5", "pynput==1.8.1", "coverage", "redis"]
        + [arg for path in modules for arg in ("-e", path)])
    run("sim-dependency-check", ["uv", "pip", "check", "--python", python])
    print("SIM_ADAPTERS_INSTALLED; actual GPU simulation still required", flush=True)
