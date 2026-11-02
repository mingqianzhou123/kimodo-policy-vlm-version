"""Compare the seven-task inventory and ordinal selections with original Kimodo."""
import argparse
from collections import Counter
from pathlib import Path

from groot_experiment import atomic_json, fingerprint
from run_config import REPO, load_yaml
from kimodo_vla.benchmarks.arena.training import build_arena_dataset
from kimodo_vla.model.planner import load_text_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'configs/arena_sonicx7_groot.yaml')
    parser.add_argument('--out', type=Path, default=REPO / 'outputs/sonicx7_preflight/data_audit.json')
    args = parser.parse_args()
    config = load_yaml(args.config)
    text = load_text_embeddings(config.main.text_cache)
    dataset = build_arena_dataset(config, text)
    reference_path = Path(config.main.original_repo) / 'reproduce_humanoidarena_sonicx7.yaml'
    reference_config = load_yaml(reference_path)
    # build_arena_dataset made the audited original loader available on sys.path.
    from data.datasetloader import MultiSourceG1Dataset
    reference = MultiSourceG1Dataset(
        dataset_roots={'HumanoidArena': config.main.data_root},
        dataset_selection=reference_config.main.dataset_selection,
        sampling=reference_config.main.sampling,
        action_history=reference_config.main.action_history,
        action_chunk=reference_config.main.action_chunk,
        sample_stride=reference_config.main.sample_stride,
        target_fps=reference_config.model.fps,
        sampling_seed=reference_config.main.seed,
        episode_cache_size=reference_config.main.episode_cache_size,
        training=True)
    ordinals = [0, 1, 31, 32, 127, 128, len(dataset)-1, len(dataset), 1000000,
                int(config.main.max_steps * config.main.batch_size)-1]
    identities = []
    for ordinal in ordinals:
        ours = dataset._sample_record(dataset._rng_for_index(ordinal))
        theirs = reference._sample_record(reference._rng_for_index(ordinal))
        # The original tuple contains source, episode, and cut index.
        def identity(record):
            adapter, episode, cut = record
            return [type(adapter).__name__, str(episode.cache_key), int(cut)]
        if identity(ours) != identity(theirs):
            raise AssertionError(f'Original sample differs at ordinal {ordinal}')
        identities.append({'ordinal': ordinal, 'identity': identity(ours)})
    counts = Counter(episode.task_name for _, episode in dataset._all_episode_records)
    if len(dataset) != len(reference) or len(dataset._all_episode_records) != 700:
        raise AssertionError('Seven-task dataset differs from original')
    atomic_json(args.out, {
        'passed': True, 'fingerprint': fingerprint(config), 'episodes': 700, 'windows': len(dataset),
        'task_counts': dict(counts), 'sampling': 'episode_uniform',
        'absolute_ordinal_range': [0, int(config.main.max_steps * config.main.batch_size)],
        'reference_sample_identity': True, 'all_text_cached': True, 'excluded': ['HOI_grap_cup'],
        'instructions': dataset.task_instructions, 'original_reference_config': str(reference_path),
        'checked_samples': identities})
    print(f'PASS: 700 episodes, {len(dataset)} windows, {len(identities)} original sample identities')


if __name__ == '__main__':
    main()
