"""Turn BiGym 2.0 demonstrations into Kimodo training episodes.

For every demo: the observed history and commanded target as Kimodo features
(``benchmarks.bigym2.demos.training_episode``) and the recorded images of all three
cameras (official 84x84) at every model-rate frame. Output: ``<out>/<task>/<demo>.pt`` and
``.frames.npy`` (see ``kimodo_vla.data.episodes``). Existing episodes are skipped.

    python scripts/prepare_bigym2.py --data-dir <bigym-g1-native60> --out <prepared dir> --workers 16
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

_CACHE: dict = {}


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)  # many workers share the host; the default oversubscribes it


def _task_state(data_dir: Path, task: str):
    if task not in _CACHE:
        from bigym.loco import make
        from bigym.loco.demos.dataset import load_episodes, load_task_metadata

        from kimodo_vla.benchmarks.bigym2.adapter import BiGym2Adapter
        from kimodo_vla.model.features import KimodoFeatures

        for env, *_ in _CACHE.values():  # jobs are task-major: keep one task at a time
            env.close()
        _CACHE.clear()
        env = make(task, camera_keys=())
        adapter = BiGym2Adapter.from_env(env, task)
        metadata = load_task_metadata(data_dir / task)
        cameras = list(metadata["task"]["camera_keys"])
        episodes = load_episodes(data_dir / task, decode_images=False)
        _CACHE[task] = (env, adapter, KimodoFeatures(adapter.control_fps), metadata["action_stats"], cameras, episodes)
    return _CACHE[task]


def prepare_one(job) -> str:
    data_dir, out_dir, task, index = job
    from kimodo_vla.benchmarks.bigym2.demos import demo_frames, training_episode
    from kimodo_vla.data.episodes import save_episode

    start = time.time()
    env, adapter, kimodo, stats, cameras, episodes = _task_state(Path(data_dir), task)
    if index >= len(episodes):
        return f"{task}: no demo {index}"
    name, episode = episodes[index]
    stem = Path(out_dir) / task / Path(name).stem
    if stem.with_suffix(".frames.npy").exists():
        return f"{task}/{stem.name}: exists"
    record, steps = training_episode(env, adapter, kimodo, episode, stats)
    record.update(task=task, episode_id=stem.name, cameras=cameras)
    save_episode(stem, record, demo_frames(episode, steps))
    return f"{task}/{stem.name}: {len(steps)} frames in {time.time() - start:.0f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", help="default: every task folder under --data-dir")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    tasks = args.tasks or sorted(p.name for p in args.data_dir.iterdir() if (p / "metadata.json").exists())
    jobs = [(str(args.data_dir), str(args.out), task, i) for task in tasks for i in range(args.episodes)]
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        chunksize = max(1, len(jobs) // (args.workers * 4))
        for message in pool.imap_unordered(prepare_one, jobs, chunksize=chunksize):
            print(message, flush=True)


if __name__ == "__main__":
    main()
