"""Prepared training episodes and the window dataset that trains Kimodo-Policy on them.

A benchmark's preparation script writes one pair of files per demonstration:

* ``<name>.pt``: ``task``, ``instruction``, ``fps``, ``cameras`` (names, in frame order)
  and, at model rate (``T`` frames),
  ``observed_motion [T, 417]`` with ``observed_mask [417]`` (what the policy is
  conditioned on), ``target_motion [T, 417]`` (what it should output) and
  ``hand [T, 2]`` (gripper commands). Motion features are unnormalized.
* ``<name>.frames.npy``: ``[T, cameras, H, W, 3]`` uint8, every camera's image at every frame.

``KimodoWindowDataset`` cuts these into the samples ``KimodoPolicy`` trains on,
with the same keys and conventions as the original repository's loader.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import hashlib
import json
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

MOTION_DIM = 417
PLANAR_ROOT_FEATURES = (0, 2)  # smooth_root_pos x and z (Kimodo is y-up)
WINDOW_SAMPLING_VERSION = "all_cuts_terminal_static_hold_v2"


@lru_cache(maxsize=1)
def _hold_representation():
    from kimodo_vla.motion.representation.kimodo_motionrep import KimodoMotionRep
    from kimodo_vla.skeleton.definitions import G1Skeleton34
    return KimodoMotionRep(G1Skeleton34(), fps=30, stats_path=None)


def terminal_hold_features(frame: torch.Tensor) -> torch.Tensor:
    """Repeat a terminal pose with consistent stationary derivative features.

    Recorded endpoint velocities copy the previous derivative. Repeating that
    entire feature row would label an unmoving hold as continued movement.
    """
    from kimodo_vla.motion.feet import foot_detect_from_pos_and_vel
    rep = _hold_representation()
    result = frame.clone()
    result[rep.slice_dict['velocities']] = 0
    positions = result[rep.slice_dict['local_joints_positions']].reshape(1, rep.skeleton.nbjoints, 3)
    # Local positions preserve absolute y; planar origin does not affect contact.
    result[rep.slice_dict['foot_contacts']] = foot_detect_from_pos_and_vel(
        positions, torch.zeros_like(positions), rep.skeleton, .15, .10)[0]
    return result


def save_episode(stem: Path, record: Mapping, frames: np.ndarray) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    expected = (record["target_motion"].shape[0], len(record["cameras"]))
    if frames.ndim != 5 or frames.shape[:2] != expected or frames.shape[-1] != 3:
        raise ValueError(f"frames must be [T, cameras, H, W, 3] with (T, cameras) = {expected}, got {frames.shape}")
    # Publish the record last: an interrupted first write cannot look complete.
    temporary = []
    try:
        for suffix, write in ((".frames.npy", lambda f: np.save(f, frames)),
                              (".pt", lambda f: torch.save(dict(record), f))):
            with tempfile.NamedTemporaryFile(dir=stem.parent, suffix=".tmp", delete=False) as f:
                temporary.append(Path(f.name))
                write(f)
        for path, suffix in zip(temporary, (".frames.npy", ".pt")):
            path.replace(stem.with_suffix(suffix))
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def validate_episode(record: Mapping, frame_path: Path) -> None:
    """Check tensor consistency and the entire image payload's declared size."""
    n = len(record["target_motion"])
    shapes = {"target_motion": (n, MOTION_DIM), "observed_motion": (n, MOTION_DIM),
              "observed_mask": (MOTION_DIM,), "hand": (n, 2)}
    if record.get("target_padding") == "pose_before_features_with_last_command_v1":
        shapes["target_motion_padded"] = (n + 49, MOTION_DIM)
        shapes["target_hand_padded"] = (n + 49, 2)
        if not torch.equal(record["target_motion"], record["target_motion_padded"][:n]):
            raise ValueError("padded target prefix differs from original target")
    if "observed_hand" in record:
        shapes["observed_hand"] = (n, 2)
    for key, shape in (("current_observed_motion", (n, MOTION_DIM)), ("current_observed_hand", (n, 2))):
        if key in record:
            shapes[key] = shape
    if "current_observed_mask" in record:
        shapes["current_observed_mask"] = (MOTION_DIM,)
    if not n:
        raise ValueError(f"{frame_path}: empty episode")
    for key, expected in shapes.items():
        value = record[key]
        if tuple(value.shape) != expected or not torch.isfinite(value).all():
            raise ValueError(f"{frame_path}: invalid {key}, expected finite {expected}")
    frames = np.load(frame_path, mmap_mode="r", allow_pickle=False)
    try:
        if (frames.ndim != 5 or frames.shape[:2] != (n, len(record["cameras"]))
                or frames.shape[-1] != 3 or min(frames.shape) < 1 or frames.dtype != np.uint8
                or frame_path.stat().st_size != frames.offset + frames.nbytes):
            raise ValueError(f"{frame_path}: invalid or incomplete image cache")
    finally:
        frames._mmap.close()


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


def base_history_mask() -> torch.Tensor:
    """History features that stay observed when the joints are masked: root heading and pelvis orientation."""
    from kimodo_vla.model.features import history_feature_mask
    from kimodo_vla.motion.representation.kimodo_motionrep import KimodoMotionRep
    from kimodo_vla.skeleton.definitions import G1Skeleton34

    representation = KimodoMotionRep(G1Skeleton34(), fps=30, stats_path=None)
    return history_feature_mask(representation, np.zeros(29, bool))


class KimodoWindowDataset(Dataset):
    """Every (episode, cut); future targets repeat the final frame at the end.

    ``egoview`` is ``[3, H, W]`` for one camera (what KimodoPolicy takes today) and
    ``[len(cameras), 3, H, W]`` for several.

    Training-time augmentation, as in BiGym 2.0's baseline recipe (off by default):
    ``image_shift`` shifts every camera's image independently by up to that many pixels
    (edge-replicated padding); with probability ``proprio_mask_prob`` per sample, the
    history keeps only its base features (root heading, pelvis orientation) and marks
    every joint unobserved, the counterpart of the recipe zeroing the non-base
    proprioception. The hand history is left alone: it holds past gripper commands.
    """

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        action_history: int = 100,
        action_chunk: int = 50,
        text_embeddings: Mapping[str, torch.Tensor] | None = None,
        tasks: Sequence[str] | None = None,
        cameras: Sequence[str] = ("head",),
        image_shift: int = 0,
        proprio_mask_prob: float = 0.0,
        required_metadata: Mapping[str, str] | None = None,
        condition_current_observation: bool = False,
    ):
        self.history, self.chunk = int(action_history), int(action_chunk)
        self.condition_current_observation = bool(condition_current_observation)
        if self.history < 0 or self.chunk < 1:
            raise ValueError("action_history must be nonnegative and action_chunk positive")
        self.image_shift, self.proprio_mask_prob = int(image_shift), float(proprio_mask_prob)
        self._base_mask = base_history_mask() if self.proprio_mask_prob > 0 else None
        self.text_embeddings = dict(text_embeddings or {})
        self.cameras = tuple(cameras)
        self.episodes: list[dict] = []
        self.frame_paths: list[Path] = []
        identities = []
        for path in sorted(p for root in roots for p in Path(root).rglob("*.pt")):
            record = torch.load(path, map_location="cpu", weights_only=True)
            if tasks is not None and record["task"] not in tasks:
                continue
            for key, expected in (required_metadata or {}).items():
                if record.get(key) != expected:
                    raise ValueError(f"{path}: {key}={record.get(key)!r}, expected {expected!r}; rebuild prepared data")
            frame_path = path.with_suffix(".frames.npy")
            validate_episode(record, frame_path)
            if self.condition_current_observation and not all(k in record for k in ("current_observed_motion", "current_observed_mask", "current_observed_hand")):
                raise ValueError(f"{path}: exact current observations are missing; rebuild prepared data")
            stat = frame_path.stat()
            identities.append((str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest(),
                               stat.st_size, stat.st_mtime_ns))
            record["camera_index"] = [record["cameras"].index(c) for c in self.cameras]
            self.episodes.append(record)
            self.frame_paths.append(frame_path)
        if not self.episodes:
            raise FileNotFoundError(f"no prepared episodes under {list(roots)}")
        self.index = [
            (e, cut)
            for e, record in enumerate(self.episodes)
            for cut in range(record["target_motion"].shape[0])
        ]
        self._frames: dict[int, np.ndarray] = {}
        identity = dict(version=WINDOW_SAMPLING_VERSION, episodes=identities,
                        history=self.history, chunk=self.chunk, cameras=self.cameras,
                        image_shift=self.image_shift, proprio_mask_prob=self.proprio_mask_prob,
                        augmentation_version="sampler_draw_seed_v1")
        identity["condition_current_observation"] = self.condition_current_observation
        self.identity = {"window_sampling_version": WINDOW_SAMPLING_VERSION,
                         "sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()}

    def __len__(self) -> int:
        return len(self.index)

    def _frame(self, e: int, cut: int, generator=None) -> torch.Tensor:
        if e not in self._frames:  # memory-mapped per worker; only the touched frames are read
            self._frames[e] = np.load(self.frame_paths[e], mmap_mode="r")
        images = torch.from_numpy(np.array(self._frames[e][cut, self.episodes[e]["camera_index"]]))
        images = images.permute(0, 3, 1, 2).contiguous()
        if self.image_shift:
            images = self._shift(images, generator)
        return images[0] if len(self.cameras) == 1 else images

    def _shift(self, images: torch.Tensor, generator=None) -> torch.Tensor:
        """Shift each ``[3, H, W]`` view by a random offset in ``[-image_shift, image_shift]``."""
        pad, (h, w) = self.image_shift, images.shape[-2:]
        padded = F.pad(images.float(), (pad, pad, pad, pad), mode="replicate")
        offsets = torch.randint(0, 2 * pad + 1, (images.shape[0], 2), generator=generator)
        return torch.stack([padded[v, :, dy : dy + h, dx : dx + w] for v, (dy, dx) in enumerate(offsets.tolist())]).to(images.dtype)

    def __getitem__(self, i: int) -> dict:
        generator = None
        if isinstance(i, tuple):
            i, seed = i
            generator = torch.Generator().manual_seed(seed)
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
        if self.proprio_mask_prob and float(torch.rand((), generator=generator)) < self.proprio_mask_prob:
            condition_mask[past] &= self._base_mask
        gt_hand[past] = record.get("observed_hand", record["hand"])[start:cut]
        gt_mask[past] = True
        # Include terminal cuts and short episodes. Repeated terminal targets
        # remain supervised: the policy must learn to hold the final pose.
        indices = torch.arange(cut, cut + c).clamp(max=len(record["target_motion"]) - 1)
        gt_motion[future] = record["target_motion"][indices]
        if cut + c > len(record['target_motion']):
            # The original final pose also has a repeated successor in this
            # window, so its forward derivative is zero too.
            terminal = indices == len(record['target_motion']) - 1
            gt_motion[h:h+c][terminal] = terminal_hold_features(record['target_motion'][-1])
        gt_hand[future] = record["hand"][indices]
        if "target_motion_padded" in record:
            if cut + c > len(record["target_motion_padded"]):
                raise ValueError("prepared pose padding does not cover the requested horizon")
            gt_motion[future] = record["target_motion_padded"][cut:cut+c]
            gt_hand[future] = record["target_hand_padded"][cut:cut+c]
        gt_mask[future] = True
        current_hand = None
        if self.condition_current_observation:
            # Prediction frame 0 is the current cut. Keep historical times and
            # future targets on their existing grid; constrain the cut itself.
            condition_motion[h] = record["current_observed_motion"][cut]
            condition_mask[h] = record["current_observed_mask"]
            gt_motion[h] = torch.where(condition_mask[h], condition_motion[h], gt_motion[h])
            current_hand = record["current_observed_hand"][cut].clone()
            gt_hand[h] = current_hand
        canonicalize_window_translation(gt_motion, condition_motion, condition_mask, gt_mask)

        sample = {
            "instruction": record["instruction"],
            "egoview": self._frame(e, cut, generator),
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
        if current_hand is not None:
            sample["current_hand_state"] = current_hand
        embedding = self.text_embeddings.get(record["instruction"])
        if embedding is not None:
            sample["text_embedding"] = embedding
            sample["text_length"] = torch.tensor(embedding.shape[0], dtype=torch.long)
        return sample
