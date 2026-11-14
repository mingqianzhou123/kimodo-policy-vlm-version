"""Download pinned videos or weights, verifying public LFS SHA256 or Git blob SHA1."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from repaired_data import checksum


def matches(path, record):
    if path.stat().st_size != record["bytes"]:
        return False
    if "sha256" in record:
        return checksum(path) == record["sha256"]
    content = path.read_bytes()
    return hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest() == record["git_blob_sha1"]


def fetch(item, output, reuse, endpoint):
    rel, record = item
    dest = output / rel
    if dest.exists():
        if not matches(dest, record):
            raise ValueError(f"Existing asset has wrong hash: {dest}")
        return {"path": rel, "status": "verified"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    for root in reuse:
        source = root / rel
        if source.is_file() and matches(source, record):
            dest.symlink_to(source.resolve())
            return {"path": rel, "status": "reused"}
    url = record["url"].replace("https://huggingface.co", endpoint)
    partial = dest.with_name(dest.name + ".partial")
    if partial.exists() and partial.stat().st_size == record["bytes"]:
        if not matches(partial, record):
            raise ValueError(f"Full partial file has wrong hash; inspect before retry: {partial}")
        partial.rename(dest)
        return {"path": rel, "status": "recovered"}
    # Short range requests survive the server's recurring long-transfer disconnects.
    # Retry budget resets after each successful range, not after an entire 5 GB shard.
    with requests.Session() as session:
        while not partial.exists() or partial.stat().st_size < record["bytes"]:
            for attempt in range(8):
                offset = partial.stat().st_size if partial.exists() else 0
                end = min(offset + 16*1024*1024, record["bytes"]) - 1
                try:
                    with session.get(url, headers={"Range": f"bytes={offset}-{end}"},
                                     stream=True, timeout=(20, 60)) as response:
                        response.raise_for_status()
                        resume = response.status_code == 206
                        expected = f"bytes {offset}-{end}/{record['bytes']}"
                        if resume and response.headers.get("Content-Range") != expected:
                            raise ValueError("Server returned an incorrect range")
                        with partial.open("ab" if resume else "wb") as stream:
                            for block in response.iter_content(1024*1024):
                                stream.write(block)
                    size = partial.stat().st_size
                    if size <= offset or size > record["bytes"]:
                        raise ValueError("Download made no progress or exceeded expected size")
                    print(json.dumps({"path": rel, "bytes": size, "total": record["bytes"]}), flush=True)
                    break
                except (requests.RequestException, ValueError) as error:
                    # Avoid copying signed redirect URLs into routine progress logs.
                    print(json.dumps({"path": rel, "attempt": attempt+1,
                                      "error_type": type(error).__name__}), flush=True)
                    if attempt == 7:
                        raise
                    time.sleep(min(2**attempt, 15))
    if not matches(partial, record):
        raise ValueError(f"Downloaded asset checksum mismatch: {rel}")
    partial.rename(dest)
    return {"path": rel, "status": "downloaded"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["videos", "weights"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse", type=Path, nargs="*", default=[])
    parser.add_argument("--endpoint", choices=["https://huggingface.co", "https://hf-mirror.com"], default="https://huggingface.co")
    args = parser.parse_args()
    args.output = args.output or ROOT / "data/cache" / args.kind
    if not args.output.resolve().is_relative_to(ROOT / "data/cache"):
        parser.error("Downloads must stay in cross_tracker/data/cache")
    manifest = "assets-v1.json" if args.kind == "videos" else "weights-v1.json"
    assets = json.loads((ROOT / "data/manifests" / manifest).read_text())
    records = assets["videos"] if args.kind == "videos" else assets["files"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(lambda item: fetch(item, args.output, args.reuse, args.endpoint), records.items()):
            print(json.dumps(result), flush=True)
