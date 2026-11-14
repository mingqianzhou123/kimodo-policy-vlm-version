"""Verify and import official wheels relayed through the development IDE."""
from pathlib import Path
import hashlib
import json
import shutil
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
meta = json.loads((RUN / "sim-wheels-transfer.json").read_text())
store = ROOT / "data/cache/setup/sim-wheels-transfer"
store.mkdir(parents=True, exist_ok=True)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


deadline = time.monotonic() + 7200
for item in meta["chunks"]:
    path = ROOT / item["name"]
    while not path.exists() or path.stat().st_size != item["bytes"]:
        if time.monotonic() > deadline:
            raise TimeoutError(path)
        time.sleep(10)
    assert digest(path) == item["sha256"], path
archive = store / meta["archive"]
with archive.open("xb") as output:
    for item in meta["chunks"]:
        with (ROOT / item["name"]).open("rb") as stream:
            shutil.copyfileobj(stream, output, 16 * 1024**2)
assert digest(archive) == meta["sha256"]
destination = ROOT / "data/cache/sim-wheels"
destination.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as bundle:
    assert len(bundle.namelist()) == meta["wheel_count"]
    assert all(Path(name).name == name and name.endswith(".whl") for name in bundle.namelist())
    bundle.extractall(destination)
pins = json.loads((RUN / "nvidia-wheel-pins.json").read_text())
for item in pins.values():
    url, expected = item["url"].split("#sha256=")
    path = destination / url.rsplit("/", 1)[1]
    assert path.stat().st_size == item["bytes"] and digest(path) == expected, path
for item in meta["chunks"]:
    (ROOT / item["name"]).rename(store / item["name"])
(RUN / "sim-wheels-import.json").write_text(json.dumps({
    "verified_wheels": len(pins), "archive_sha256": meta["sha256"],
    "finished_unix": time.time(),
}, indent=2))
print("VERIFIED_SIM_WHEELS", len(pins), flush=True)
