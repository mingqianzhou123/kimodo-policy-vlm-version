"""Freeze balanced episode lists from verified repair artifacts; never train."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def split_episodes(ids, count, seed, dataset, fraction):
    """Stable selection, independent of training seed and input ordering."""
    if len(ids) != len(set(ids)) or len(ids) < count or not 0 < fraction < 1:
        raise ValueError(f"Invalid candidate pool/split: {dataset}")
    def rank(stage, eid):
        return hashlib.sha256(f"{seed}:{stage}:{dataset}:{eid}".encode()).hexdigest()
    chosen = sorted(ids, key=lambda eid: rank("select", eid))[:count]
    chosen.sort(key=lambda eid: rank("split", eid))
    n_val = math.ceil(count * fraction)
    if n_val >= count:
        raise ValueError("Empty training split")
    return {"train": sorted(chosen[n_val:]), "validation": sorted(chosen[:n_val])}


def freeze(paths, config):
    audit, repair = Path(paths["audit"]), Path(paths["repair"])
    inputs = {name: repair / name for name in ("dataset_manifest.json", "episode_repairs.json")}
    datasets = json.loads(inputs["dataset_manifest.json"].read_text())
    decisions = json.loads(inputs["episode_repairs.json"].read_text())
    records = []
    for task, count in config["tasks"].items():
        for tracker in config["trackers"]:
            dataset = f"{task}/{tracker}_refpose_v3_1"
            entry = next(row for row in datasets if row["dataset"] == dataset)
            source = audit / "dataset" / dataset / "data/chunk-000/file-000.parquet"
            overlay = repair / entry["overlay"]
            for path, expected in ((source, entry["source_sha256"]), (overlay, entry["overlay_sha256"])):
                if sha256(path) != expected:
                    raise ValueError(f"Checksum mismatch: {path}")
            keys = ["index", "episode_index", "frame_index", "timestamp"]
            original = pq.read_table(source, columns=keys)
            fixed = pq.read_table(overlay)
            if any(not original[key].equals(fixed[key]) for key in keys):
                raise ValueError(f"Misaligned overlay: {dataset}")
            mask = fixed["numeric_eligible"].to_numpy()
            episode = fixed["episode_index"].to_numpy()
            candidates = [row for row in decisions if row["dataset"] == dataset and row["status"] == "numeric_candidate"]
            splits = split_episodes([row["episode"] for row in candidates], count,
                                    config["subset_seed"], dataset, config["validation_fraction"])
            selected = set(splits["train"] + splits["validation"])
            ranges = []
            for row in candidates:
                if row["episode"] not in selected:
                    continue
                left, right = row["original_row_start"], row["original_row_stop"]
                valid = np.flatnonzero(mask[left:right]) + left
                # Current repair removes prefixes or whole episodes. Refuse future holes.
                if not len(valid) or not np.array_equal(valid, np.arange(valid[0], valid[-1] + 1)):
                    raise ValueError(f"Non-contiguous eligible segment: {dataset}/{row['episode']}")
                if not np.all(episode[left:right] == row["episode"]):
                    raise ValueError("Episode range mismatch")
                begin = int(valid[0]) + config["observation_history_frames"]
                end = int(valid[-1]) + 2 - config["action_chunk_frames"]
                if end <= begin:
                    raise ValueError("Episode has no complete safe window")
                ranges.append({"episode": row["episode"], "eligible_rows": [int(valid[0]), int(valid[-1]) + 1],
                               "window_start_range": [begin, end], "split": "train" if row["episode"] in splits["train"] else "validation"})
            records.append({"dataset": dataset, "candidate_episodes": len(candidates), **splits,
                            "source": {"root": "audit", "path": str(source.relative_to(audit)), "sha256": entry["source_sha256"]},
                            "overlay": {"root": "repair", "path": entry["overlay"], "sha256": entry["overlay_sha256"]},
                            "episodes": sorted(ranges, key=lambda row: row["episode"])})
    return {"protocol": config, "dataset_revision": json.loads((audit / "hf_dataset.json").read_text())["sha"],
            "repair_inputs_sha256": {name: sha256(path) for name, path in inputs.items()},
            "success_certified": False, "training_ready": False, "datasets": records}


def save_frozen(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Refusing to change frozen manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/manifests/balanced-v1.json")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / "data/manifests"):
        parser.error("Manifests must stay under cross_tracker/data/manifests")
    manifest = freeze(json.loads(args.paths.read_text()), json.loads(args.config.read_text()))
    save_frozen(args.output, manifest)
    print(json.dumps({"manifest": str(args.output), "sha256": sha256(args.output),
                      "train": sum(len(d["train"]) for d in manifest["datasets"]),
                      "validation": sum(len(d["validation"]) for d in manifest["datasets"])}))
