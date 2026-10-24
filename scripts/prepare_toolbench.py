"""Turn HumanoidToolBench demonstrations (ToolBook) into Kimodo training episodes.

For every episode of every recording under --data-dir (a local copy of the
dataset's ``raw/`` at revision 8b2cd31e): the history and the operator's commands
as Kimodo features (``kimodo_vla.data.build.training_episode``) and the head
camera, resized to 224x224, at every model-rate frame. Output:
``<out>/<condition>/<recording>_<episode>.pt`` and ``.frames.npy`` (see
``kimodo_vla.data.episodes``). Existing episodes are skipped.

    python scripts/prepare_toolbench.py --data-dir <ToolBook>/raw --out <prepared dir> --workers 16

By default only successful episodes are used (``episode_success`` in
``meta/episodes.jsonl``); ``--all`` keeps the rest.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

_FEATURES = None


def _init_worker() -> None:
    import torch

    global _FEATURES
    torch.set_num_threads(1)  # many workers share the host; the default oversubscribes it
    from kimodo_vla.benchmarks.toolbench.adapter import CONTROL_FPS
    from kimodo_vla.model.features import KimodoFeatures

    _FEATURES = KimodoFeatures(CONTROL_FPS)


def recordings(data_dir: Path) -> list[Path]:
    """Every LeRobot recording below ``data_dir`` (a directory with ``meta/info.json``)."""
    return sorted(p.parent.parent for p in data_dir.rglob("meta/info.json"))


def prepare_one(job) -> str:
    recording, out_dir, episode, name = job
    from kimodo_vla.benchmarks.toolbench import demos
    from kimodo_vla.data.build import training_episode
    from kimodo_vla.data.episodes import save_episode

    start = time.time()
    recording = Path(recording)
    condition = recording.parent.name.removesuffix("-v0")
    stem = Path(out_dir) / condition / f"{name}_{episode:06d}"
    if stem.with_suffix(".frames.npy").exists():
        return f"{stem.name}: exists"
    video = recording / f"videos/chunk-{episode // 1000:03d}/observation.images.ego_view/episode_{episode:06d}.mp4"
    demo = demos.ToolBenchDemo(demos.read_recording(recording, episode), demos.instruction(recording.parent.name), video)
    record, frames = training_episode(demo, _FEATURES, task=condition, episode_id=stem.name)
    save_episode(stem, record, frames)
    return f"{condition}/{stem.name}: {len(frames)} frames in {time.time() - start:.0f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--all", action="store_true", help="also failed episodes")
    parser.add_argument("--limit", type=int, help="at most this many episodes per recording")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from kimodo_vla.benchmarks.toolbench import demos

    jobs = []
    for recording in recordings(args.data_dir):
        name = "_".join(recording.relative_to(args.data_dir).parts).replace("/", "_")
        kept = [e["episode_index"] for e in demos.episodes(recording) if args.all or e.get("episode_success")]
        jobs += [(str(recording), str(args.out), episode, name) for episode in kept[: args.limit]]
    print(f"{len(jobs)} episodes", flush=True)
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        for message in pool.imap_unordered(prepare_one, jobs, chunksize=1):
            print(message, flush=True)


if __name__ == "__main__":
    main()
