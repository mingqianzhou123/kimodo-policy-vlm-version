"""Turn HumanoidArena demonstrations into Kimodo training episodes.

For each selected task and tracker under --data-dir (a local copy of the Hugging Face dataset
``WilliamWang16/HumanoidArena_dataset_v3_1`` at revision ca04c2d5): the history and the
whole-body reference as Kimodo features (``kimodo_vla.data.build.training_episode``) and the
front camera, resized to the encoder's 224x224, at every model-rate frame. Output:
``<out>/<task>/<tracker>/episode_<index>.pt`` and ``.frames.npy`` (see
``kimodo_vla.data.episodes``), about 100 MB per episode. Existing episodes are skipped.

    python scripts/prepare_arena.py --data-dir <dataset> --out <prepared dir> --tasks HOI_double_desk --workers 8
    python scripts/prepare_arena.py --data-dir <dataset> --out <prepared dir> --tasks all --trackers sonic twist2

Training then names ``<task>/<tracker>`` folders (``scripts/train.py --task HOI_double_desk/sonic``).
The merged datasets in the download (``HumanoidArena_merged_datasets_v3_1``) are copies of the
per-task ones; train on several folders instead of preparing them twice.
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
    from kimodo_vla.benchmarks.arena.adapter import CONTROL_FPS
    from kimodo_vla.model.features import KimodoFeatures

    _FEATURES = KimodoFeatures(CONTROL_FPS)


def prepare_one(job) -> str:
    task, tracker, instruction, episode, out_dir = job
    from kimodo_vla.benchmarks.arena import demos
    from kimodo_vla.data.build import training_episode
    from kimodo_vla.data.episodes import save_episode

    start = time.time()
    stem = Path(out_dir) / task / tracker / f"episode_{episode.index:06d}"
    if stem.with_suffix(".frames.npy").exists():
        return f"{task}/{tracker}/{stem.name}: exists"
    state, action = demos.read_episode(episode)
    demo = demos.ArenaDemo(state, action, instruction, episode.video_path, episode.video_from_timestamp)
    record, frames = training_episode(demo, _FEATURES, task=task, episode_id=f"{task}/{tracker}/{stem.name}")
    record["tracker"] = tracker
    save_episode(stem, record, frames)
    return f"{task}/{tracker}/{stem.name}: {len(frames)} frames in {time.time() - start:.0f}s"


def main() -> None:
    from kimodo_vla.benchmarks.arena import demos

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True,
                        help=f"task folders, or 'all' / 'evaluated' (all but {', '.join(demos.NOT_EVALUATED)})")
    parser.add_argument("--trackers", nargs="+", default=["sonic"], choices=demos.TRACKERS)
    parser.add_argument("--limit", type=int, help="at most this many episodes per task and tracker (a quick test)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    every = sorted(set(demos.TASKS.values()))
    tasks = {"all": every, "evaluated": [t for t in every if t not in demos.NOT_EVALUATED]}.get(args.tasks[0], args.tasks)
    jobs = []
    for task in tasks:
        if task not in every:
            raise SystemExit(f"unknown task {task!r}; expected one of {every}")
        for tracker in args.trackers:
            directory = demos.dataset_dir(args.data_dir, task, tracker)
            if not directory.is_dir():
                print(f"{task}/{tracker}: not in the dataset, skipped", flush=True)
                continue
            instruction, episodes = demos.read_dataset(directory)
            jobs += [(task, tracker, instruction, e, str(args.out)) for e in episodes[: args.limit]]
            print(f"{task}/{tracker}: {len(episodes[: args.limit])} episodes, {instruction!r}", flush=True)
    print(f"{len(jobs)} episodes", flush=True)
    failed = 0
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        for message in pool.imap_unordered(_safe, jobs, chunksize=1):
            failed += message.startswith("FAILED")
            print(message, flush=True)
    print(f"done, {failed} failed" if failed else "done", flush=True)
    raise SystemExit(1 if failed else 0)


def _safe(job) -> str:
    try:
        return prepare_one(job)
    except Exception as error:  # noqa: BLE001 - report the episode and keep the others going
        return f"FAILED {job[0]}/{job[1]}/episode {job[3].index}: {type(error).__name__}: {error}"


if __name__ == "__main__":
    main()
