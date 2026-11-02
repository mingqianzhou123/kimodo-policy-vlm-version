"""Use the audited original Arena loader without changing its sampling contract."""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
from functools import partial
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


def build_arena_dataset(config, text_embeddings):
    root = Path(config.main.original_repo).expanduser().resolve()
    if not (root / "data/datasetloader.py").is_file():
        raise FileNotFoundError(f"Original Kimodo Arena loader not found at {root}")
    if str(root) not in sys.path:
        sys.path.append(str(root))
    from data.datasetloader import MultiSourceG1Dataset

    dataset = MultiSourceG1Dataset(
        dataset_roots={"HumanoidArena": config.main.data_root},
        dataset_selection=config.main.get("arena_dataset_selection") or {"HumanoidArena": {"task": config.main.task, "backend": "sonic"}},
        sampling={"mode": "episode_uniform"},
        action_history=config.main.action_history,
        action_chunk=config.main.action_chunk,
        sample_stride=config.main.get("sample_stride", 1),
        target_fps=config.model.fps,
        sampling_seed=config.main.seed,
        episode_cache_size=config.main.get("episode_cache_size", 100),
        video_cache_size=config.main.get("video_cache_size", 32),
        training=True,
    )
    dataset.set_text_embeddings({task: text_embeddings[instruction]
                                 for task, instruction in dataset.task_instructions.items()})
    counts = Counter(episode.task_name for _, episode in dataset._all_episode_records)
    expected = config.main.get("expected_tasks")
    if expected and dict(counts) != dict(expected):
        raise ValueError(f"Arena task inventory differs from the experiment: {dict(counts)} != {dict(expected)}")
    return dataset


class ArenaOrdinalBatchSampler(Sampler):
    """Match Accelerate's contiguous per-rank batches of absolute ordinals.

    The original loader maps an ordinal deterministically to an episode/cut.
    It accepts ordinals beyond len(dataset); shuffling a finite index range
    would repeatedly reuse a small subset throughout a 500k-update run.
    """
    def __init__(self, start_step, max_steps, batch_size, micro_batch, rank, world):
        self.start_step, self.max_steps = int(start_step), int(max_steps)
        self.batch_size, self.micro_batch = int(batch_size), int(micro_batch)
        self.rank, self.world = int(rank), int(world)
        if self.batch_size % (self.micro_batch * self.world):
            raise ValueError("global batch must divide into rank micro batches")
        if not 0 <= self.rank < self.world or not 0 <= self.start_step <= self.max_steps:
            raise ValueError("invalid ordinal sampler rank or step range")
        self.accumulation = self.batch_size // (self.micro_batch * self.world)

    def __iter__(self):
        for step in range(self.start_step, self.max_steps):
            for micro in range(self.accumulation):
                first = step * self.batch_size + (micro * self.world + self.rank) * self.micro_batch
                yield list(range(first, first + self.micro_batch))

    def __len__(self):
        return (self.max_steps - self.start_step) * self.accumulation


def initialize_arena_worker(worker_id, *, seed, start_step, world, workers, rank):
    worker_seed = seed + start_step * world * max(1, workers) + rank * max(1, workers) + worker_id
    np.random.seed(worker_seed % (2**32))
    random.seed(worker_seed)
    torch.manual_seed(worker_seed % (2**63 - 1))
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def original_arena_batches(dataset, config, start_step, rank, world):
    workers = int(config.main.num_workers)
    return DataLoader(dataset, batch_sampler=ArenaOrdinalBatchSampler(start_step, config.main.max_steps,
        config.main.batch_size, config.main.micro_batch_size, rank, world), num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0, prefetch_factor=1 if workers else None,
        multiprocessing_context="spawn" if workers else None,
        worker_init_fn=partial(initialize_arena_worker, seed=int(config.main.seed), start_step=start_step,
                              world=world, workers=workers, rank=rank))
