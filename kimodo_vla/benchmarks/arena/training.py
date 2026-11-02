"""Use the audited original Arena loader without changing its sampling contract."""

from __future__ import annotations

import sys
from pathlib import Path


def build_arena_dataset(config, text_embeddings):
    root = Path(config.main.original_repo).expanduser().resolve()
    if not (root / "data/datasetloader.py").is_file():
        raise FileNotFoundError(f"Original Kimodo Arena loader not found at {root}")
    if str(root) not in sys.path:
        sys.path.append(str(root))
    from data.datasetloader import MultiSourceG1Dataset

    dataset = MultiSourceG1Dataset(
        dataset_roots={"HumanoidArena": config.main.data_root},
        dataset_selection={"HumanoidArena": {"task": config.main.task, "backend": "sonic"}},
        sampling={"mode": "episode_uniform"},
        action_history=config.main.action_history,
        action_chunk=config.main.action_chunk,
        target_fps=config.model.fps,
        sampling_seed=config.main.seed,
        episode_cache_size=config.main.get("episode_cache_size", 100),
        training=True,
    )
    dataset.set_text_embeddings({task: text_embeddings[instruction]
                                 for task, instruction in dataset.task_instructions.items()})
    return dataset
