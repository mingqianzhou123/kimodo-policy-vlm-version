"""Disjoint, same-controller context from the frozen box training split."""
import hashlib
import json
import os
from pathlib import Path
import torch
from sampling import TrainingDataset

ROOT = Path(__file__).resolve().parents[1]


def select_context(episodes, target, generator):
    record = episodes[target]
    pool = [i for i, e in enumerate(episodes)
            if e["task"] == record["task"] and e["tracker"] == record["tracker"]
            and e["episode_id"] != record["episode_id"] and e["split"] == "train"]
    if len(pool) < 3:
        raise ValueError("Need three distinct training context trajectories besides the target")
    return [pool[i] for i in torch.randperm(len(pool), generator=generator)[:3].tolist()]


def combine_context(records, metadata, mean, std):
    counts = [43, 43, 42]  # 128 total, covering the whole span of each trajectory.
    streams, slots = {"visual": [], "transitions": []}, []
    for slot, (record, count) in enumerate(zip(records, counts)):
        if len(record["transitions"]) != 43 or len(record["visual"]) != 43:
            raise ValueError("Expected 43 verified context samples per episode")
        indices = torch.linspace(0, 42, count).round().long()
        for key in streams:
            streams[key].append(record[key][indices])
        slots.append(torch.full((count,), slot, dtype=torch.long))
    result = {k: torch.cat(v) for k, v in streams.items()}
    result["transitions"] = (result["transitions"].float() - mean) / std
    result.update(trajectory_ids=torch.cat(slots), metadata=metadata)
    return result


class CCTDataset(TrainingDataset):
    def __init__(self, roots, **kwargs):
        super().__init__(roots, **kwargs)
        cache = ROOT / "data/cache/cct-box-v1"
        report = json.loads((cache / "manifest.json").read_text())
        expected = hashlib.sha256((ROOT / "data/manifests/balanced-v1.json").read_bytes()).hexdigest()
        if report["manifest_sha256"] != expected or not report["complete"]:
            raise ValueError("Unverified context cache")
        if {e["episode_id"] for e in self.episodes} != set(report["episodes"]):
            raise ValueError("CCT requires exactly the frozen 43+43 box training episodes")
        for relative, digest in report["files"].items():
            with (cache / relative).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"Context cache changed: {relative}")
        self.contexts = []
        for e in self.episodes:
            item = report["episodes"][e["episode_id"]]
            record = torch.load(cache / item["file"], weights_only=True, map_location="cpu")
            if record["episode_id"] != e["episode_id"] or record["split"] != "train":
                raise ValueError("Context trajectory identity/split mismatch")
            self.contexts.append(record)
        self.metadata = torch.load(cache / "metadata.pt", weights_only=True, map_location="cpu")
        self.mean, self.std = torch.tensor(report["mean"]), torch.tensor(report["std"])
        self.generator = torch.Generator().manual_seed(42 * 1000003 + int(os.environ.get("RANK", 0)))

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        target, _ = self.index[index]
        ids = select_context(self.episodes, target, self.generator)
        source = self.episodes[target]["tracker"]
        sample["egoview"] = {"image": sample["egoview"], "context": combine_context(
            [self.contexts[i] for i in ids], self.metadata[source], self.mean, self.std)}
        sample["context_episode_ids"] = [self.episodes[i]["episode_id"] for i in ids]
        return sample
