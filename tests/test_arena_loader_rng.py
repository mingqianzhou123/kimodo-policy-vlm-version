"""Recreating a resumed data iterator must not advance training randomness."""
import torch
from omegaconf import OmegaConf

from kimodo_vla.benchmarks.arena.training import original_arena_batches


def test_resumed_iterator_preserves_training_rng_and_sample_position():
    dataset = torch.utils.data.TensorDataset(torch.arange(16))
    cfg = OmegaConf.create({'main': {'seed': 42, 'num_workers': 0,
        'isolated_loader_rng': True, 'max_steps': 4, 'batch_size': 4, 'micro_batch_size': 2}})
    before = torch.get_rng_state().clone()
    uninterrupted = iter(original_arena_batches(dataset, cfg, 0, 0, 2))
    assert next(uninterrupted)[0].tolist() == [0, 1]
    expected = next(uninterrupted)[0]
    resumed = iter(original_arena_batches(dataset, cfg, 1, 0, 2))
    assert torch.equal(next(resumed)[0], expected)
    assert torch.equal(torch.get_rng_state(), before)
