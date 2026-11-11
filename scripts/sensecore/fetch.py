"""Download files listed in a manifest from hf-mirror.com, without the HF listing API.

    python fetch.py manifest.tsv [--first N] [--workers 4]

manifest.tsv rows: <W|D>\t<path in repo>\t<size>, built on a machine that can reach huggingface.co
(W = YunhengWang/kimodo-policy @ aed6ac7, D = dataset WilliamWang16/HumanoidArena_dataset_v3_1 @ ca04c2d).
A file whose local size matches is skipped; partial files resume with HTTP Range; each file is retried
with back-off (hf-mirror throttles with 403/503/521). Prints FETCH_OK or FETCH_MISSING <n> at the end.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

H = os.environ["KVLA_HOME"]  # set by env.sh
REPOS = {
    "W": ("https://hf-mirror.com/YunhengWang/kimodo-policy/resolve/aed6ac74d774f869eccd3010e0352424ee3bd248/", f"{H}/weights/kimodo-policy"),
    "D": ("https://hf-mirror.com/datasets/WilliamWang16/HumanoidArena_dataset_v3_1/resolve/ca04c2d53598d640bb3580edb5c6af78d59ea6b1/", f"{H}/data/HumanoidArena_dataset_v3_1"),
}


def fetch(code: str, path: str, size: int) -> tuple[str, bool]:
    base, root = REPOS[code]
    dest = os.path.join(root, path)
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return path, True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    url = base + urllib.parse.quote(path)
    for attempt in range(30):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if have > size:
                os.remove(part); have = 0
            if have < size or size == 0:
                req = urllib.request.Request(url, headers={"User-Agent": "kvla-fetch/1.0", **({"Range": f"bytes={have}-"} if have else {})})
                with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have and r.status == 206 else "wb") as fh:
                    while chunk := r.read(1 << 20):
                        fh.write(chunk)
            if os.path.getsize(part) == size:
                os.replace(part, dest)
                return path, True
        except Exception as exc:  # noqa: BLE001 - mirror errors are retried
            print(f"retry {attempt + 1} {path}: {exc}", flush=True)
        time.sleep(min(300, 10 * (attempt + 1)))
    return path, False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--first", type=int, default=0, help="only the first N rows")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    rows = [line.rstrip("\n").split("\t") for line in open(a.manifest) if line.strip()]
    if a.first:
        rows = rows[: a.first]
    total = sum(int(r[2]) for r in rows)
    done = missing = 0
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as pool:
        futures = {pool.submit(fetch, c, path, int(s)): int(s) for c, path, s in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            path, ok = fut.result()
            done += futures[fut] if ok else 0
            missing += not ok
            if i % 20 == 0 or not ok:
                print(f"{i}/{len(rows)} files, {done / 1e9:.1f}/{total / 1e9:.1f} GB, {time.time() - t0:.0f}s"
                      + ("" if ok else f" FAILED {path}"), flush=True)
    print("FETCH_OK" if not missing else f"FETCH_MISSING {missing}", flush=True)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
