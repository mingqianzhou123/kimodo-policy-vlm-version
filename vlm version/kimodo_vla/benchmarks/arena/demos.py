"""HumanoidArena demonstrations (``WilliamWang16/HumanoidArena_dataset_v3_1``) as training episodes.

The dataset has one LeRobot v3 dataset per task and tracker, ``<root>/<task>/<tracker>_refpose_v3_1/``
(trackers: ``sonic``, ``twist2``, ``mimic_lite``), recorded at 50 Hz. Columns used::

    observation.state (64)      the rows the simulator posts as ``state_history`` at evaluation:
                                root rot6d relative to the episode's first heading, 29 joint angles, 29 velocities
    action (40)                 the whole-body reference the tracker followed (``ActionChunk.from_array40``
                                layout): root xy displacement, height, root rot6d, 29 joints, two binary hands
    observation.images.front    480x640 video, the image the simulator sends with each request

As in the original repository (``data/humanoidarena_loader.py``): the target of frame ``t`` is
row ``t``'s action, and the hand history is the commanded hand (Arena's state has no hand
reading; at evaluation the hand history is the policy's own previous output). The history goes
through ``adapter.state64_observations``, the function the Arena server uses on each request.
Images are stored at the 224x224 the DINOv3 encoder resizes every image to (bilinear,
antialiased), so training sees the same pixels as an encoder fed the full 480x640 frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from kimodo_vla.benchmarks.arena.adapter import CAMERA, CONTROL_FPS, state64_observations
from kimodo_vla.core.types import Observation
from kimodo_vla.data.build import DemoMotion
from kimodo_vla.motion.g1_reference import HumanoidArenaActionDecoder, rot6d_row_to_matrix

SCHEMA = "unitree_g1_gmt_refpose_v3_1"
# The simulator's task id -> the dataset's task folder.
TASKS = {
    "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody": "HOI_double_desk",
    "Isaac-Move-Football-Single-G129-Dex3-Wholebody": "HOI_football",
    "Isaac-Move-ArtVIP-Livingroom-GrapCup-G129-Dex3-Wholebody": "HOI_grap_cup",
    "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby": "HOI_pp_box",  # sic, as in the dataset
    "Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody": "HSI_boxing",
    "Isaac-Move-Open-Door-G129-Dex3-Wholebody": "HSI_open_door",
    "Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody": "HSI_sit_sofa",
    "Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody": "HSI_vision_navi",
}
TRACKERS = ("sonic", "twist2", "mimic_lite")
# Not in the official evaluation protocol; the original repository leaves it out of multi-task training.
NOT_EVALUATED = ("HOI_grap_cup",)
IMAGE_SIZE = 224  # DINOv3's preprocessor_config.json: resize to 224x224, bilinear
VIDEO_KEY = "observation.images.front"


def dataset_dir(root: Path, task: str, tracker: str) -> Path:
    return Path(root) / task / f"{tracker}_refpose_v3_1"


@dataclass
class EpisodeInfo:
    index: int
    length: int
    data_path: Path
    video_path: Path
    video_from_timestamp: float


def read_dataset(directory: Path) -> tuple[str, list[EpisodeInfo]]:
    """``(instruction, episodes)`` of one task/tracker dataset, checked against the expected format."""
    import pyarrow.parquet as pq

    directory = Path(directory)
    info = json.loads((directory / "meta/info.json").read_text())
    schema = info.get("vla_protocol", {}).get("schema")
    if schema != SCHEMA or tuple(info["features"]["action"]["shape"]) != (40,) or float(info["fps"]) != CONTROL_FPS:
        raise ValueError(f"{directory}: expected {SCHEMA}, 40-D actions at {CONTROL_FPS:g} Hz")
    instructions = [str(t).strip() for t in pq.read_table(directory / "meta/tasks.parquet").column("task").to_pylist()]
    if len(instructions) != 1:
        raise ValueError(f"{directory}: expected one instruction, found {instructions}")
    episodes = []
    for path in sorted((directory / "meta/episodes").rglob("*.parquet")):
        rows = pq.read_table(path).to_pydict()
        for i, index in enumerate(rows["episode_index"]):
            if [TASKS.get(t) for t in rows["tasks"][i]] != [directory.parent.name]:
                raise ValueError(f"{directory}: episode {index} is task {rows['tasks'][i]}")
            episodes.append(EpisodeInfo(
                index=int(index),
                length=int(rows["length"][i]),
                data_path=directory / info["data_path"].format(
                    chunk_index=rows["data/chunk_index"][i], file_index=rows["data/file_index"][i]),
                video_path=directory / info["video_path"].format(
                    video_key=VIDEO_KEY, chunk_index=rows[f"videos/{VIDEO_KEY}/chunk_index"][i],
                    file_index=rows[f"videos/{VIDEO_KEY}/file_index"][i]),
                video_from_timestamp=float(rows[f"videos/{VIDEO_KEY}/from_timestamp"][i]),
            ))
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError(f"{directory}: {len(episodes)} episodes listed, info.json says {info['total_episodes']}")
    return instructions[0], sorted(episodes, key=lambda e: e.index)


def task_instructions(root: Path, tracker: str = "sonic") -> dict[str, str]:
    """Task folder -> instruction, for every task that has a ``tracker`` dataset under ``root``."""
    import pyarrow.parquet as pq

    out = {}
    for task in sorted(set(TASKS.values())):
        path = dataset_dir(root, task, tracker) / "meta/tasks.parquet"
        if path.is_file():
            out[task] = str(pq.read_table(path).column("task")[0]).strip()
    return out


def read_episode(episode: EpisodeInfo) -> tuple[np.ndarray, np.ndarray]:
    """``(state [T, 64], action [T, 40])`` float32, in frame order."""
    import pyarrow.parquet as pq

    table = pq.read_table(episode.data_path, columns=["observation.state", "action", "frame_index"],
                          filters=[("episode_index", "=", episode.index)])
    frames = np.asarray(table.column("frame_index").to_numpy())
    if not np.array_equal(frames, np.arange(episode.length)):
        raise ValueError(f"episode {episode.index}: rows are not frames 0..{episode.length - 1}")
    state = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
    action = np.stack(table.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
    if not (np.isfinite(state).all() and np.isfinite(action).all()):
        raise ValueError(f"episode {episode.index}: NaN or Inf in state or action")
    return state, action


def resize_image(images: np.ndarray) -> np.ndarray:
    """``[N, H, W, 3]`` uint8 -> ``[N, 224, 224, 3]``, the resize DINOv3's image processor applies.

    The processor rescales to [0, 1] and then resizes (bilinear, antialiased) in float; doing the
    same and rounding back to uint8 leaves it at most half a gray level from the full-size path.
    """
    from torchvision.transforms.v2 import InterpolationMode
    from torchvision.transforms.v2 import functional as F

    tensor = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 255.0
    resized = F.resize(tensor, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return (resized * 255.0).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()


class ArenaDemo:
    """One recorded Arena episode as a ``benchmarks.base.Demonstration``."""

    control_fps = CONTROL_FPS
    cameras = (CAMERA,)
    hand_mode = "binary"

    def __init__(self, state: np.ndarray, action: np.ndarray, instruction: str,
                 video: Path | None = None, video_from_timestamp: float = 0.0):
        if state.shape[1:] != (64,) or action.shape != (len(state), 40):
            raise ValueError(f"expected state [T, 64] and action [T, 40], got {state.shape} and {action.shape}")
        self.state, self.action, self.instruction = state, action, instruction
        self.video, self.video_from_timestamp = video, float(video_from_timestamp)

    def __len__(self) -> int:
        return len(self.state)

    def observations(self) -> list[Observation]:
        observations = state64_observations(self.state, self.instruction, None)
        for step, obs in enumerate(observations):
            obs.step = step
        return observations

    def commanded_motion(self) -> DemoMotion:
        """The reference the tracker followed; the root path is integrated as the original decoder does."""
        actions = torch.from_numpy(self.action)
        root_rot = rot6d_row_to_matrix(actions[:, 3:9])
        root_pos = HumanoidArenaActionDecoder._integrate_root(actions, root_rot)
        return DemoMotion(root_pos=root_pos.numpy(), root_rot=root_rot.numpy(),
                          joint_pos=self.action[:, 9:38].copy(), hand=self.action[:, 38:40].copy())

    def images(self, steps: np.ndarray) -> np.ndarray:
        """``[len(steps), 1, 224, 224, 3]``: the front camera at those control steps."""
        import av

        first = round(self.video_from_timestamp * CONTROL_FPS)
        wanted = {first + int(s) for s in steps}
        frames: dict[int, np.ndarray] = {}
        with av.open(str(self.video)) as container:
            stream = container.streams.video[0]
            if abs(float(stream.average_rate) - CONTROL_FPS) > 1e-6:
                raise ValueError(f"{self.video} runs at {stream.average_rate} fps, expected {CONTROL_FPS:g}")
            container.seek(int(self.video_from_timestamp / float(stream.time_base)), stream=stream, backward=True)
            for frame in container.decode(stream):
                index = round(float(frame.pts * stream.time_base) * CONTROL_FPS)
                if index in wanted:
                    frames[index] = resize_image(frame.to_ndarray(format="rgb24")[None])[0]  # keep memory small
                if index >= max(wanted):
                    break
        missing = wanted - set(frames)
        if missing:
            raise ValueError(f"{self.video} has no frames {sorted(missing)[:5]} (from {self.video_from_timestamp}s)")
        return np.stack([frames[first + int(s)] for s in steps])[:, None]

