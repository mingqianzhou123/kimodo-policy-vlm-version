"""Read 43 aligned 50 Hz transitions per training episode, with prepared RGB indices."""
import json
from pathlib import Path
import numpy as np
import torch
from repaired_data import checksum, read_segment

ROOT = Path(__file__).resolve().parents[1]


def context_inputs(roots):
    path = ROOT / "data/manifests/balanced-v1.json"
    manifest = json.loads(path.read_text())
    assets = json.loads((ROOT / "data/manifests/assets-v1.json").read_text())
    for dataset in manifest["datasets"]:
        name = dataset["dataset"].removesuffix("_refpose_v3_1")
        if name not in ("HOI_pp_box/sonic", "HOI_pp_box/twist2"):
            continue
        for episode in dataset["episodes"]:
            if episode["split"] != "train":
                continue
            episode_id = f"{name}/{episode['episode']:06d}"
            stem = ROOT / "data/cache/prepared-v1/train" / name / f"episode_{episode['episode']:06d}"
            receipt = json.loads(stem.with_suffix(".json").read_text())
            if receipt["manifest_sha256"] != checksum(path):
                raise ValueError("Prepared split changed")
            for ext, digest in receipt["files"].items():
                if checksum(stem.with_suffix(ext)) != digest:
                    raise ValueError(f"Prepared content changed: {stem}{ext}")
            state, action, mapping = read_segment(dataset, episode, roots, assets)
            frames = np.load(stem.with_suffix(".frames.npy"), mmap_mode="r")
            raw = np.minimum(np.round(np.arange(len(frames)) * 50 / 30).astype(int), len(state)-1)
            valid = np.flatnonzero(raw < len(state)-1)
            indices = valid[np.linspace(0, len(valid)-1, 43).round().astype(int)]
            rows = raw[indices]
            if len(np.unique(indices)) != 43:
                raise ValueError("Not enough distinct context images")
            transitions = np.concatenate((state[rows, :35], action[rows], state[rows+1, :35]), axis=1)
            if transitions.shape != (43, 110) or not np.isfinite(transitions).all():
                raise ValueError("Invalid commanded/achieved transitions")
            images = np.array(frames[indices])
            if images.shape != (43, 1, 224, 224, 3):
                raise ValueError(f"Unexpected prepared images: {images.shape}")
            record = dict(episode_id=episode_id, split="train", transitions=torch.from_numpy(transitions),
                          prepared_indices=indices.tolist(), source_rows=(rows+episode["eligible_rows"][0]).tolist(),
                          next_source_rows=(rows+episode["eligible_rows"][0]+1).tolist(),
                          source_mapping=mapping, prepared_sha256=receipt["files"])
            yield record, torch.from_numpy(images[:, 0]).permute(0, 3, 1, 2).contiguous()
