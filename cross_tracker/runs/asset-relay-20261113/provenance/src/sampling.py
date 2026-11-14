"""Reuse Kimodo windows, but balance task, source, episode, then valid window."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Sampler
from kimodo_vla.data.episodes import KimodoWindowDataset

ROOT = Path(__file__).resolve().parents[1]


class TrainingDataset(KimodoWindowDataset):
    def __init__(self, roots, **kwargs):
        super().__init__(roots, **kwargs)
        manifest_path = ROOT / "data/manifests/balanced-v1.json"
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        manifest = json.loads(manifest_path.read_text())
        allowed = {f"{d['dataset'].removesuffix('_refpose_v3_1')}/{e:06d}"
                   for d in manifest["datasets"] for e in d["train"]}
        ids = [e["episode_id"] for e in self.episodes]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate training episodes")
        for record, frames in zip(self.episodes, self.frame_paths):
            if record.get("split") != "train" or record["episode_id"] not in allowed:
                raise ValueError("Validation/unlisted episode reached the training loader")
            if record.get("manifest_sha256") != digest:
                raise ValueError("Wrong prepared-data manifest")
            receipt = frames.with_suffix("").with_suffix(".json")
            if not receipt.exists():
                raise ValueError("Incomplete prepared episode")
        if not self.index:
            raise ValueError("No complete prediction windows")


class BalancedSampler(Sampler):
    def __init__(self, dataset, num_replicas=1, rank=0, seed=0):
        self.counts = torch.bincount(torch.tensor([e for e, _ in dataset.index]), minlength=len(dataset.episodes))
        if (self.counts == 0).any():
            raise ValueError("An episode has no full prediction window; do not silently drop it")
        self.starts = self.counts.cumsum(0) - self.counts
        self.groups = defaultdict(lambda: defaultdict(list))
        for i, episode in enumerate(dataset.episodes):
            self.groups[episode["task"]][episode["tracker"]].append(i)
        source_sets = [set(group) for group in self.groups.values()]
        if any(sources != source_sets[0] for sources in source_sets):
            raise ValueError("Tasks do not contain the same sources")
        self.num_samples = len(dataset) // num_replicas
        self.seed = int(seed)*1_000_003 + rank

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        def pick(values):
            return values[int(torch.randint(len(values), (), generator=generator))]
        tasks = sorted(self.groups)
        for _ in range(self.num_samples):
            group = self.groups[pick(tasks)]
            episodes = group[pick(sorted(group))]
            episode = pick(episodes)
            offset = int(torch.randint(int(self.counts[episode]), (), generator=generator))
            yield int(self.starts[episode]) + offset
