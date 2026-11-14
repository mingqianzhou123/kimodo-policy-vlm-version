"""Download official pinned NVIDIA wheels locally for verified transfer if server HTTPS stalls."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import urllib.parse
import urllib.request

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
destination = ROOT / "data/cache/sim-wheels"
destination.mkdir(exist_ok=True)
pins = json.loads((RUN / "nvidia-wheel-pins.json").read_text())


def fetch(item):
    name, pin = item
    url, fragment = pin["url"].split("#")
    expected = fragment.removeprefix("sha256=")
    path = destination / urllib.parse.unquote(url.rsplit("/", 1)[-1])
    if not path.exists():
        temporary = path.with_suffix(".partial")
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as output:
            while chunk := source.read(8 * 1024**2):
                output.write(chunk)
        with temporary.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name
        temporary.rename(path)
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name
    print("VERIFIED", name, path.stat().st_size, flush=True)
    return name, {"file": path.name, "sha256": expected, "bytes": path.stat().st_size}


with ThreadPoolExecutor(max_workers=4) as pool:
    receipt = dict(pool.map(fetch, pins.items()))
(RUN / "sim-wheels-local.json").write_text(json.dumps(receipt, indent=2))
print("ALL_VENDOR_WHEELS_VERIFIED", len(receipt), flush=True)
