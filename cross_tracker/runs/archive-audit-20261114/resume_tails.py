"""Resume verified-range prefixes without discarding bytes after a curl timeout."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import shutil
import subprocess

RUN = Path(__file__).resolve().parent
PARTS = RUN.parents[1] / "data/cache/setup/modelscope-main/parts"
SIZE, CHUNK = 29160726392, 268435456
URL = "https://modelscope.cn/datasets/Twang2026/HumanoidArenaV3.1/resolve/fb35e9a5be95d4c9b7fefd1913c2ca558bb3cb4c/HumanoidArena_datasets_v3_1.zip"


def resume(index):
    path = PARTS / f"{index:03d}.part"
    start, end = index * CHUNK, min(SIZE, (index + 1) * CHUNK) - 1
    prefix = path.stat().st_size
    if prefix == end - start + 1:
        return
    headers = (PARTS / f"{index:03d}.headers").read_text()
    ranges = re.findall(r"content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)", headers, re.I)
    assert tuple(map(int, ranges[-1])) == (start, end, SIZE)
    tail, header = PARTS / f"{index:03d}.tail", PARTS / f"{index:03d}.tail.headers"
    print(f"Resuming part {index} after {prefix} bytes", flush=True)
    subprocess.run(["curl", "--noproxy", "*", "-fLsS", "--retry", "3", "--connect-timeout", "20",
                    "--max-time", "900", "--range", f"{start + prefix}-{end}",
                    "-D", str(header), "-o", str(tail), URL], check=True)
    ranges = re.findall(r"content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)", header.read_text(), re.I)
    assert tuple(map(int, ranges[-1])) == (start + prefix, end, SIZE)
    assert tail.stat().st_size == end - start + 1 - prefix and path.stat().st_size == prefix
    with path.open("ab") as output, tail.open("rb") as source:
        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    print(f"Part {index} complete", flush=True)


with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(resume, range(109)))
subprocess.run(["python", "-u", str(RUN / "download_parts.py")], check=True)
