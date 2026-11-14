"""Fetch bounded HTTP ranges concurrently; full hash is checked by audit_archive.py."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
CACHE = ROOT / "data/cache/setup/modelscope-main"
PARTS = CACHE / "parts"
PARTS.mkdir(exist_ok=True)
SIZE, CHUNK = 29160726392, 256 * 1024 * 1024
URL = "https://modelscope.cn/datasets/Twang2026/HumanoidArenaV3.1/resolve/fb35e9a5be95d4c9b7fefd1913c2ca558bb3cb4c/HumanoidArena_datasets_v3_1.zip"


def fetch(index):
    start, end = index * CHUNK, min(SIZE, (index + 1) * CHUNK) - 1
    path = PARTS / f"{index:03d}.part"
    header = PARTS / f"{index:03d}.headers"
    def valid():
        if not path.exists() or not header.exists() or path.stat().st_size != end - start + 1:
            return False
        ranges = re.findall(r"content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)", header.read_text(), re.I)
        return ranges and tuple(map(int, ranges[-1])) == (start, end, SIZE)
    if not valid():
        with (PARTS / f"{index:03d}.log").open("w") as log:
            subprocess.run(["curl", "--noproxy", "*", "-fLsS", "--retry", "5", "--connect-timeout", "20",
                            "--max-time", "900", "--speed-time", "90", "--speed-limit", "100000", "--range", f"{start}-{end}", "-D", str(header),
                            "-o", str(path), URL], stdout=log, stderr=subprocess.STDOUT, check=True)
        assert valid(), f"Range mismatch: {index}"
    print(f"part {index:03d} verified byte range, {path.stat().st_size} bytes", flush=True)
    return path


started = time.time()
with ThreadPoolExecutor(max_workers=32) as pool:
    parts = list(pool.map(fetch, range((SIZE + CHUNK - 1) // CHUNK)))
target = CACHE / "HumanoidArena_datasets_v3_1.tar.xz"
assert not target.exists(), "Preserve any earlier partial download before assembling"
with target.open("xb") as output:
    for path in parts:
        with path.open("rb") as source:
            shutil.copyfileobj(source, output, 8 * 1024 * 1024)
assert target.stat().st_size == SIZE
(RUN / "download-result.json").write_text(json.dumps({"bytes": SIZE, "parts": len(parts),
    "seconds": time.time() - started, "status": "downloaded_pending_full_hash"}, indent=2))
subprocess.run(["python3", str(RUN / "audit_archive.py")], check=True)
