"""Import verified public simulator inputs and reapply the recorded material fixes."""
from pathlib import Path
import hashlib
import json
import os
import zipfile

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
CACHE = ROOT / "data/cache"
ASSETS = CACHE / "sim-assets"
ARCHIVES = {
    "humanoidarena-assets.zip": ("arena/assets", "d0a0c6be2e47ac9a3b161ec9ca11a9d67908ef755fe0fd96445404938376447b"),
    "sonic-weights.zip": (".", "1c3fbdfc8c575366f1795e45aead003fa8ac61e83ee25155ac735cc7ea881cfb"),
    "table-materials.zip": ("table-materials", "514af6314b7f5f581eb6da4dcaa95a30407fa4d786bff198250a356991ca3ab5"),
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    records = []
    # Validate all archives before creating any extracted files.
    for name, (_, digest) in ARCHIVES.items():
        assert sha(CACHE / name) == digest, name
    assert not ASSETS.exists(), "Preserve any previous import; inspect before resuming"
    for name, (relative, digest) in ARCHIVES.items():
        with zipfile.ZipFile(CACHE / name) as archive:
            assert all(not Path(n).is_absolute() and ".." not in Path(n).parts
                       for n in archive.namelist())
            archive.extractall(ASSETS / relative)
        records.append({"archive": name, "sha256": digest})
    materials = ASSETS / "table-materials"
    for item in json.loads((materials / "manifest.json").read_text()):
        assert sha(materials / item["path"]) == item["sha256"]
    original_dir = CACHE / "setup/eval-20261116-material-originals"
    original_dir.mkdir(parents=True, exist_ok=False)
    assets = ASSETS / "arena/assets"
    replacements = []

    def replace(path, original, fixed):
        relative = path.relative_to(assets)
        saved = original_dir / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(original)
        path.write_bytes(fixed)
        replacements.append({"path": str(relative), "before_sha256": sha(saved),
                             "after_sha256": sha(path)})

    for variant in ("", "small_warehouse_box", "small_warehouse_boxing", "small_warehouse_vision_navigation"):
        path = assets / "objects/small_warehouse" / variant / "SubUSDs/materials/M_TrafficCone.mdl"
        original = path.read_bytes()
        assert "export material OmniUe4Base(" in path.with_name("OmniUe4Base_1.mdl").read_text()
        assert original.count(b"::OmniUe4Base_1(") == 1
        replace(path, original, original.replace(b"::OmniUe4Base_1(", b"::OmniUe4Base("))
    base = (b"http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/"
            b"Isaac/Props/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/materials/")
    for path in assets.rglob("*.usd"):
        original = path.read_bytes()
        if base not in original:
            continue
        assert original.startswith(b"#usda ")
        fixed = original
        for mdl in materials.glob("*.mdl"):
            fixed = fixed.replace(base + mdl.name.encode(), os.path.relpath(mdl, path.parent).encode())
        assert base not in fixed
        replace(path, original, fixed)
    arena = CACHE / "sources/HumanoidArena-44a72229f11bab47a145e12db39ed891b677e783/isaaclab_twist2_g1"
    link = arena / "assets"
    assert not link.exists() and not link.is_symlink()
    link.symlink_to(assets, target_is_directory=True)
    (RUN / "sim-inputs.json").write_text(json.dumps({"archives": records, "material_fixes": replacements}, indent=2) + "\n")
    print("SIM_INPUTS_IMPORTED; runtime not yet checked", flush=True)


if __name__ == "__main__":
    main()
