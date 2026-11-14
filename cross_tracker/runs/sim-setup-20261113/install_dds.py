"""Build pinned DDS and install the official Unitree SDK inside the simulation environment."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
bundle = ROOT / "sim-dds-sources.zip"
assert hashlib.sha256(bundle.read_bytes()).hexdigest() == "90b9172adb48c54679d135e638dc70b59944f1fd35b7b442e4d7b4e161fa2958"
store = ROOT / "data/cache/setup/dds"
store.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(bundle) as archive:
    assert all(Path(n).name == n for n in archive.namelist())
    archive.extractall(store)
sources = ROOT / "data/cache/sources"
for name in ("unitree-sdk-source.json", "cyclonedds-source.json"):
    item = json.loads((RUN / name).read_text())
    path = store / item["archive"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    with tarfile.open(path) as archive:
        archive.extractall(sources, filter="data")
bundle.rename(store / bundle.name)
dds = sources / "cyclonedds-9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd"
sdk = sources / "unitree_sdk2_python-814556d15970dd2ecf1c9984e845ca02ab07e206"
prefix = ROOT / "data/cache/cyclonedds-0.10.2"
python = ROOT / "data/cache/envs/isaaclab-2.2/bin/python"
env = {**os.environ, "CYCLONEDDS_HOME": str(prefix), "UV_CACHE_DIR": str(ROOT / "data/cache/uv")}
env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1,pypi.tuna.tsinghua.edu.cn"
commands = [
    ["cmake", "-S", str(dds), "-B", str(dds / "build"), f"-DCMAKE_INSTALL_PREFIX={prefix}",
     "-DBUILD_TESTING=OFF", "-DBUILD_EXAMPLES=OFF"],
    ["cmake", "--build", str(dds / "build"), "--target", "install", "-j", "8"],
    ["uv", "pip", "install", "--python", str(python), "--no-build-isolation",
     "--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple",
     "--constraint", str(RUN / "sim-constraints-r3.txt"), "-e", str(sdk)],
    ["uv", "pip", "check", "--python", str(python)],
]
with (RUN / "dds-install.log").open("x") as log:
    for command in commands:
        print("COMMAND", command, file=log, flush=True)
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
(RUN / "dds-install.json").write_text(json.dumps({"commands": commands, "passed": True}, indent=2))
print("DDS_SDK_INSTALLED", flush=True)
