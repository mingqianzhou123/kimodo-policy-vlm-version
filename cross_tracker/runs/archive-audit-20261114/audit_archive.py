"""Verify the pinned ModelScope archive, then stream every tar entry without extraction."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import lzma
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
ARCHIVE = ROOT / "data/cache/setup/modelscope-main/HumanoidArena_datasets_v3_1.tar.xz"
EXPECTED = "a92723e304046fb6073a6cca05eb21f6d5e88b0dea782496a5f33b54f3719782"
report = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
          "expected_bytes": 29160726392, "expected_sha256": EXPECTED,
          "revision": "fb35e9a5be95d4c9b7fefd1913c2ca558bb3cb4c",
          "source": "https://modelscope.cn/datasets/Twang2026/HumanoidArenaV3.1"}
try:
    assert ARCHIVE.stat().st_size == report["expected_bytes"], "Incomplete download"
    digest = hashlib.sha256()
    with ARCHIVE.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    report["sha256"] = digest.hexdigest()
    assert report["sha256"] == EXPECTED, "Archive hash mismatch"
    print("Full archive SHA256 verified", flush=True)
    counts, matches, nested, mimic = Counter(), [], [], []
    count = 0
    with (RUN / "entries.jsonl").open("x") as listing, lzma.open(ARCHIVE, "rb") as raw:
        with tarfile.open(fileobj=raw, mode="r|") as archive:
            for item in archive:
                name = item.name.lower()
                entry = {"name": item.name, "size": item.size, "type": item.type.decode("ascii")}
                listing.write(json.dumps(entry) + "\n")
                count += 1
                if item.isfile():
                    counts[Path(name).suffix] += 1
                if "xua2csee" in name or name.endswith((".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".bin")):
                    matches.append(entry)
                if item.isfile() and name.endswith((".zip", ".xz", ".gz", ".tar", ".7z", ".rar")):
                    nested.append(entry)
                if "mimic" in name:
                    mimic.append(entry)
                if count % 100 == 0:
                    print(f"Scanned {count} entries: {item.name}", flush=True)
        # tar end markers may precede XZ's checksum/footer. Read through EOF to verify those too.
        while raw.read(8 * 1024 * 1024):
            pass
    report.update(status="completed", entries=count, suffix_counts=dict(counts),
                  weight_candidates=matches, nested_archives=nested, mimic_paths=mimic,
                  exact_weight_found=any(Path(x["name"]).name == "policy-xua2csee-4000.onnx" for x in matches),
                  xz_eof_verified=True, listing_sha256=hashlib.sha256((RUN / "entries.jsonl").read_bytes()).hexdigest())
except BaseException as error:
    report.update(status="failed", error=repr(error))
    raise
finally:
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (RUN / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
