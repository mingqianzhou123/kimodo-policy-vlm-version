"""Build repaired Kimodo episodes on CPU, separated by train/validation and source."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent)]
import torch
from kimodo_vla.data.episodes import save_episode
from repaired_data import checksum, prepare_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "data/cache/prepared-v1")
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--episode", type=int)
    parser.add_argument("--limit", type=int, help="Per dataset and split; diagnostic only")
    parser.add_argument("--split", choices=["train", "validation", "both"], default="both")
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / "data/cache"):
        parser.error("Prepared files must stay in cross_tracker/data/cache")
    torch.set_num_threads(1)
    manifest_path = ROOT / "data/manifests/balanced-v1.json"
    manifest = json.loads(manifest_path.read_text())
    assets = json.loads((ROOT / "data/manifests/assets-v1.json").read_text())
    roots = json.loads(args.paths.read_text())
    jobs = []
    for d in manifest["datasets"]:
        if args.datasets and d["dataset"] not in args.datasets:
            continue
        for split in ("train", "validation"):
            if args.split not in (split, "both"):
                continue
            selected = [e for e in d["episodes"] if e["split"] == split
                        and (args.episode is None or e["episode"] == args.episode)]
            jobs.extend((d, e) for e in selected[:args.limit])
    if not jobs:
        parser.error("No selected episodes")
    for d, e in jobs:
        started = time.time()
        task, tracker = d["dataset"].split("/")
        folder = args.out / e["split"] / task / tracker.removesuffix("_refpose_v3_1")
        stem = folder / f"episode_{e['episode']:06d}"
        receipt = stem.with_suffix(".json")
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["manifest_sha256"] != checksum(manifest_path):
                raise ValueError("Prepared episode uses a different split")
            if not all(checksum(stem.with_suffix(ext)) == digest for ext, digest in saved["files"].items()):
                raise ValueError(f"Prepared file changed: {stem}")
            print(f"VERIFIED {stem}", flush=True)
            continue
        if stem.with_suffix(".pt").exists() or stem.with_suffix(".frames.npy").exists():
            raise FileExistsError(f"Incomplete previous preparation needs inspection: {stem}")
        record, frames = prepare_record(d, e, roots, assets, args.video_root)
        record["manifest_sha256"] = checksum(manifest_path)
        save_episode(stem, record, frames)
        result = {"manifest_sha256": checksum(manifest_path), "model_fps": 30,
                  "model_frames": len(frames), "windows": len(frames)-49,
                  "source_mapping": record["source_mapping"], "split": e["split"],
                  "files": {ext: checksum(stem.with_suffix(ext)) for ext in (".pt", ".frames.npy")}}
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        print(f"PREPARED {stem}: {len(frames)} model frames, {time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
