"""Prepared training episodes and the window dataset that trains Kimodo-Policy on them.

A benchmark's preparation script writes one pair of files per demonstration:

* ``<name>.pt``: ``task``, ``instruction``, ``fps`` and, at model rate (``T`` frames),
  ``observed_motion [T, 417]`` with ``observed_mask [417]`` (what the policy is
  conditioned on), ``target_motion [T, 417]`` (what it should output) and
  ``hand [T, 2]`` (gripper commands). Motion features are unnormalized.
* ``<name>.frames.npy``: ``[T, H, W, 3]`` uint8, the camera image at every frame.

``KimodoWindowDataset`` cuts these into the samples ``KimodoPolicy`` trains on,
with the same keys and conventions as the original repository's loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

MOTION_DIM = 417
PLANAR_ROOT_FEATURES = (0, 2)  # smooth_root_pos x and z (Kimodo is y-up)


def save_episode(stem: Path, record: Mapping, frames: np.ndarray) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[0] != record["target_motion"].shape[0] or frames.shape[-1] != 3:
        raise ValueError(f"frames must be [T, H, W, 3] with T = {record['target_motion'].shape[0]}, got {frames.shape}")
    torch.save(dict(record), stem.with_suffix(".pt"))
    np.save(stem.with_suffix(".frames.npy"), frames)


def canonicalize_window_translation(gt_motion, condition_motion, condition_mask, gt_mask) -> None:
    """Move the first valid target root x/z of a window to the origin, in place.

    Official Kimodo clips are translated so they start above the origin; the
    clean target and any valid condition root features shift together.
    """
    first = int(torch.nonzero(gt_mask)[0])
    origin = gt_motion[first, list(PLANAR_ROOT_FEATURES)].clone()
    for offset, feature in enumerate(PLANAR_ROOT_FEATURES):
        gt_motion[gt_mask, feature] -= origin[offset]
        valid = gt_mask & condition_mask[:, feature]
        condition_motion[valid, feature] -= origin[offset]


class KimodoWindowDataset(Dataset):
    """Every (episode, cut) with a full future chunk; history is left-padded at the start."""

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        action_history: int = 100,
        action_chunk: int = 50,
        text_embeddings: Mapping[str, torch.Tensor] | None = None,
        tasks: Sequence[str] | None = None,
    ):
        self.history, self.chunk = int(action_history), int(action_chunk)
        self.text_embeddings = dict(text_embeddings or {})
        self.episodes: list[dict] = []
        self.frame_paths: list[Path] = []
        for path in sorted(p for root in roots for p in Path(root).rglob("*.pt")):
            record = torch.load(path, map_location="cpu", weights_only=True)
            if tasks is not None and record["task"] not in tasks:
                continue
            self.episodes.append(record)
            self.frame_paths.append(path.with_suffix(".frames.npy"))
        if not self.episodes:
            raise FileNotFoundError(f"no prepared episodes under {list(roots)}")
        self.index = [
            (e, cut)
            for e, record in enumerate(self.episodes)
            for cut in range(record["target_motion"].shape[0] - self.chunk + 1)
        ]
        self._frames: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _frame(self, e: int, cut: int) -> torch.Tensor:
        if e not in self._frames:  # memory-mapped per worker; only the touched frames are read
            self._frames[e] = np.load(self.frame_paths[e], mmap_mode="r")
        return torch.from_numpy(np.array(self._frames[e][cut])).permute(2, 0, 1).contiguous()

    def __getitem__(self, i: int) -> dict:
        e, cut = self.index[i]
        record = self.episodes[e]
        h, c = self.history, self.chunk
        start = max(0, cut - h)
        n_history = cut - start
        past = slice(h - n_history, h)
        future = slice(h, h + c)

        gt_motion = torch.zeros(h + c, MOTION_DIM)
        condition_motion = torch.zeros(h + c, MOTION_DIM)
        condition_mask = torch.zeros(h + c, MOTION_DIM, dtype=torch.bool)
        gt_hand = torch.zeros(h + c, 2)
        gt_mask = torch.zeros(h + c, dtype=torch.bool)

        gt_motion[past] = record["target_motion"][start:cut]
        condition_motion[past] = record["observed_motion"][start:cut]
        condition_mask[past] = record["observed_mask"]
        gt_hand[past] = record["hand"][start:cut]
        gt_mask[past] = True
        gt_motion[future] = record["target_motion"][cut : cut + c]
        gt_hand[future] = record["hand"][cut : cut + c]
        gt_mask[future] = True
        canonicalize_window_translation(gt_motion, condition_motion, condition_mask, gt_mask)

        sample = {
            "instruction": record["instruction"],
            "egoview": self._frame(e, cut),
            "gt_motion": gt_motion,
            "condition_motion": condition_motion,
            "condition_motion_mask": condition_mask,
            "gt_hand": gt_hand,
            "gt_hand_mask": gt_mask.unsqueeze(-1).expand(-1, 2).clone(),
            "gt_mask": gt_mask,
            "task_id": record["task"],
            "episode_id": record["episode_id"],
            "cut_index": cut,
        }
        embedding = self.text_embeddings.get(record["instruction"])
        if embedding is not None:
            sample["text_embedding"] = embedding
            sample["text_length"] = torch.tensor(embedding.shape[0], dtype=torch.long)
        return sample
