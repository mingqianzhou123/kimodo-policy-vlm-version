"""Import the pinned public model bundle; keep incomplete earlier downloads separate."""
from pathlib import Path
import hashlib
import json
import zipfile

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
CACHE = ROOT / "data/cache"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
    receipt = json.loads((RUN / "policy-transfer.json").read_text())
    archive = CACHE / receipt["archive"]
    assert archive.stat().st_size == receipt["bytes"] and sha(archive) == receipt["sha256"]
    stage = CACHE / "import-policy-20261116"
    stage.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in bundle.namelist())
        bundle.extractall(stage)
    manifest = json.loads((ROOT / "data/manifests/shared-weights-v1.json").read_text())
    records = []
    for name, item in manifest["files"].items():
        path = stage / "shared-weights" / name
        assert path.stat().st_size == item["bytes"], name
        digest = sha(path)
        if "sha256" in item:
            assert digest == item["sha256"], name
        else:
            data = path.read_bytes()
            assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == item["git_blob_sha1"], name
        records.append({"path": name, "bytes": item["bytes"], "sha256": digest})
    destination = CACHE / "shared-weights"
    if destination.exists():
        saved = CACHE / "shared-weights-incomplete-hf-20261116"
        assert not saved.exists()
        destination.rename(saved)
    (stage / "shared-weights").rename(destination)
    link = ROOT.parent / "checkpoints"
    assert not link.exists() and not link.is_symlink()
    link.symlink_to(destination / "checkpoints", target_is_directory=True)
    (RUN / "policy-assets.json").write_text(json.dumps(records, indent=2) + "\n")
    print("POLICY_ASSETS_IMPORTED_AND_VERIFIED", len(records), flush=True)
