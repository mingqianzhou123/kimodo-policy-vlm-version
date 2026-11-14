"""Read verified valid segments, then reuse Kimodo's existing feature conversion."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=2)
def checked_table(path, expected):
    if checksum(path) != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    return pq.read_table(path)


def read_segment(dataset, episode, roots, assets):
    """Return state/action and exact original video coordinates; do not decode yet."""
    tables = []
    for key in ("source", "overlay"):
        ref = dataset[key]
        tables.append(checked_table(str(Path(roots[ref["root"]]) / ref["path"]), ref["sha256"]))
    original, overlay = tables
    for key in ("index", "episode_index", "frame_index", "timestamp"):
        if not original[key].equals(overlay[key]):
            raise ValueError(f"Misaligned repair: {dataset['dataset']}/{key}")
    left, right = episode["eligible_rows"]
    if not 0 <= left < right <= len(original):
        raise ValueError("Invalid segment bounds")
    raw, fixed = original.slice(left, right-left), overlay.slice(left, right-left)
    if not fixed["numeric_eligible"].to_numpy().all():
        raise ValueError("Segment contains excluded rows")
    if not np.all(raw["episode_index"].to_numpy() == episode["episode"]):
        raise ValueError("Segment crosses an episode boundary")
    frames = raw["frame_index"].to_numpy()
    if not np.array_equal(frames, np.arange(frames[0], frames[0]+len(frames))):
        raise ValueError("Non-contiguous original frames")
    state = np.asarray(raw["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(fixed["candidate.action"].to_pylist(), dtype=np.float32)
    if state.shape != (len(raw), 64) or action.shape != (len(raw), 40):
        raise ValueError("Expected state64/action40")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Non-finite training data")
    if np.max(np.abs(action[0, :2])) > 1e-6:
        raise ValueError("Segment must begin at a repaired zero root displacement")
    directory = Path(roots["audit"]) / "dataset" / dataset["dataset"]
    for rel, expected in assets["metadata"][dataset["dataset"]].items():
        if checksum(directory / rel) != expected:
            raise ValueError(f"Metadata changed: {directory / rel}")
    info = json.loads((directory / "meta/info.json").read_text())
    if info["fps"] != 50:
        raise ValueError("Expected 50 Hz control data")
    metadata = pq.read_table(directory / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    meta = next(row for row in metadata if row["episode_index"] == episode["episode"])
    prefix = "videos/observation.images.front/"
    video = info["video_path"].format(video_key="observation.images.front",
                                    chunk_index=meta[prefix+"chunk_index"], file_index=meta[prefix+"file_index"])
    times = raw["timestamp"].to_numpy().astype(np.float64) + meta[prefix+"from_timestamp"]
    if not np.allclose(times, times[0] + np.arange(len(raw))/50, atol=2e-5, rtol=0):
        raise ValueError("Non-uniform video timestamp mapping")
    if times[-1] >= meta[prefix+"to_timestamp"] + 1e-5:
        raise ValueError("Video timestamp exceeds episode")
    task = pq.read_table(directory / "meta/tasks.parquet")["task"][0].as_py()
    return state, action, {"instruction": task, "video": dataset["dataset"] + "/" + video,
                           "video_offset_s": float(times[0]), "source_frame_start": int(frames[0]),
                           "source_frame_stop": int(frames[-1])+1}


def prepare_record(dataset, episode, roots, assets, video_root):
    from kimodo_vla.benchmarks.arena.demos import ArenaDemo
    from kimodo_vla.data.build import training_episode
    from kimodo_vla.model.features import KimodoFeatures

    state, action, mapping = read_segment(dataset, episode, roots, assets)
    video = Path(video_root) / mapping["video"]
    expected = assets["videos"][mapping["video"]]["sha256"]
    verify_video(str(video), expected)
    # A valid segment becomes a virtual episode: state/target share a new heading
    # and a 30 Hz grid; camera time keeps its original 50 Hz file offset.
    demo = ArenaDemo(state, action, mapping["instruction"], video, mapping["video_offset_s"])
    task, tracker = dataset["dataset"].split("/")
    tracker = tracker.removesuffix("_refpose_v3_1")
    record, images = training_episode(demo, KimodoFeatures(50), task=task,
                                     episode_id=f"{task}/{tracker}/{episode['episode']:06d}")
    record.update(tracker=tracker, split=episode["split"], source_mapping=mapping,
                  source_sha256=dataset["source"]["sha256"], overlay_sha256=dataset["overlay"]["sha256"])
    if len(images) < 50:
        raise ValueError("Segment shorter than Kimodo's 50 model-frame prediction")
    if not all(value.isfinite().all() for value in (record["observed_motion"], record["target_motion"], record["hand"])):
        raise ValueError("Non-finite Kimodo features")
    return record, images


@lru_cache(maxsize=32)
def verify_video(path, expected):
    if checksum(path) != expected:
        raise ValueError(f"Video checksum mismatch: {path}")
