"""Package selected numerical inputs and metadata; videos get a pinned download list."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from repaired_data import checksum
import pyarrow.parquet as pq
from freeze_data import save_frozen


def package(paths, manifest, output):
    audit, repair = Path(paths["audit"]), Path(paths["repair"])
    tree = {r["path"]: r for r in json.loads((audit / "hf_tree.json").read_text()) if r["type"] == "file"}
    assets = {"dataset_revision": manifest["dataset_revision"], "metadata": {}, "videos": {}}
    files = {}
    for d in manifest["datasets"]:
        for key in ("source", "overlay"):
            ref = d[key]
            file = Path(paths[ref["root"]]) / ref["path"]
            if checksum(file) != ref["sha256"]:
                raise ValueError(f"Changed input: {file}")
            files[f"{ref['root']}/{ref['path']}"] = file
        directory = audit / "dataset" / d["dataset"]
        rels = ["meta/info.json", "meta/tasks.parquet", "meta/episodes/chunk-000/file-000.parquet"]
        assets["metadata"][d["dataset"]] = {rel: checksum(directory / rel) for rel in rels}
        for rel in rels:
            files[f"audit/dataset/{d['dataset']}/{rel}"] = directory / rel
        info = json.loads((directory / "meta/info.json").read_text())
        selected = set(d["train"] + d["validation"])
        prefix = "videos/observation.images.front/"
        for e in pq.read_table(directory / rels[-1]).to_pylist():
            if e["episode_index"] not in selected:
                continue
            rel = d["dataset"] + "/" + info["video_path"].format(video_key="observation.images.front",
                chunk_index=e[prefix+"chunk_index"], file_index=e[prefix+"file_index"])
            entry = tree[rel]
            assets["videos"][rel] = {"bytes": entry["size"], "sha256": entry["lfs"]["oid"],
                "url": f"https://huggingface.co/datasets/WilliamWang16/HumanoidArena_dataset_v3_1/resolve/{manifest['dataset_revision']}/{rel}"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, file in files.items():
            archive.write(file, name)
        archive.writestr("SHA256.json", json.dumps({name: checksum(file) for name, file in files.items()}, indent=2))
    save_frozen(ROOT / "data/manifests/assets-v1.json", assets)
    print(json.dumps({"archive": str(output), "bytes": output.stat().st_size, "sha256": checksum(output),
                      "files": len(files), "video_files": len(assets["videos"]),
                      "video_bytes": sum(v["bytes"] for v in assets["videos"].values())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/cache/inputs-v1.zip")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / "data/cache"):
        parser.error("Output must stay under cross_tracker/data/cache")
    package(json.loads(args.paths.read_text()), json.loads((ROOT / "data/manifests/balanced-v1.json").read_text()), args.output)
