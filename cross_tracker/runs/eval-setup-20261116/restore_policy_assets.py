"""Restore only the pinned motion/vision assets needed for cached-text inference."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "data/manifests/shared-weights-v1.json").read_text())
output = ROOT / "data/cache/shared-weights"
env = {k: v for k, v in os.environ.items()
       if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")}


def verified(path, item):
    if not path.is_file() or path.stat().st_size != item["bytes"]:
        return False
    if "sha256" in item:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == item["git_blob_sha1"]


if __name__ == "__main__":
    records = []
    for name, item in manifest["files"].items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not verified(target, item):
            raise ValueError(f"Preserving mismatched existing asset: {target}")
        if not target.exists():
            partial = target.with_name(target.name + ".partial")
            if not verified(partial, item):
                url = (f"https://huggingface.co/{manifest['repository']}/resolve/"
                       f"{manifest['revision']}/{name}")
                subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                                "--retry", "5", "--connect-timeout", "20", "--max-time", "7200",
                                "--continue-at", "-", "--output", str(partial), url],
                               env=env, check=True)
            if not verified(partial, item):
                raise ValueError(f"Downloaded asset failed verification: {partial}")
            partial.rename(target)
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append({"path": name, "bytes": item["bytes"], "sha256": digest})
        (RUN / "policy-assets.json").write_text(json.dumps(records, indent=2) + "\n")
        print("VERIFIED", name, flush=True)
    print("POLICY_ASSETS_COMPLETE; no GPU or training launched", flush=True)
