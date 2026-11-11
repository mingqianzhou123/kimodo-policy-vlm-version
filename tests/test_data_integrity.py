"""Corrupt caches, incompatible resumes, and resumed data augmentation."""
from itertools import islice
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kimodo_vla.data.episodes import KimodoWindowDataset, save_episode, validate_episode
from kimodo_vla.benchmarks.toolbench.policy import routes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train import batches, check_resume_data, prepared_requirements
from tune_toolbench import Search


def episode(root):
    n = 12
    record = dict(task='toy', episode_id='ep', instruction='x', fps=30, cameras=['head'],
                  target_motion=torch.zeros(n, 417), observed_motion=torch.ones(n, 417),
                  observed_mask=torch.ones(417, dtype=torch.bool), hand=torch.zeros(n, 2))
    frames = np.arange(n * 1 * 8 * 8 * 3, dtype=np.uint8).reshape(n, 1, 8, 8, 3)
    save_episode(root / 'ep', record, frames)
    return record, frames


@pytest.mark.parametrize('damage', ['truncated', 'wrong_dtype', 'wrong_length', 'nonfinite', 'empty'])
def test_invalid_episode_is_rejected_before_training(tmp_path, damage):
    record, frames = episode(tmp_path)
    path = tmp_path / 'ep.frames.npy'
    if damage == 'truncated':
        with path.open('r+b') as f:
            f.truncate(path.stat().st_size - 9)
    elif damage == 'wrong_dtype':
        np.save(path, frames.astype(np.float32))
    elif damage == 'wrong_length':
        np.save(path, frames[:-1])
    elif damage == 'nonfinite':
        record['observed_motion'][3, 7] = float('nan')
        torch.save(record, tmp_path / 'ep.pt')
    else:
        record['target_motion'] = record['target_motion'][:0]
        torch.save(record, tmp_path / 'ep.pt')
    with pytest.raises((ValueError, OSError)):
        KimodoWindowDataset([tmp_path])


def test_interrupted_image_write_preserves_existing_pair(tmp_path, monkeypatch):
    record, frames = episode(tmp_path)
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    def fail(*args, **kwargs):
        raise OSError('interrupted write')
    monkeypatch.setattr(np, 'save', fail)
    with pytest.raises(OSError, match='interrupted'):
        save_episode(tmp_path / 'ep', record, frames)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == original


def test_resume_checks_data_contents_and_window_configuration(tmp_path):
    record, _ = episode(tmp_path)
    identity = KimodoWindowDataset([tmp_path]).identity
    check_resume_data({'data_identity': identity}, identity)
    with pytest.raises(ValueError, match='legacy sampler offset'):
        check_resume_data({}, identity)
    assert KimodoWindowDataset([tmp_path], action_chunk=15).identity != identity
    record['target_motion'][0, 8] = 1
    torch.save(record, tmp_path / 'ep.pt')
    changed = KimodoWindowDataset([tmp_path]).identity
    with pytest.raises(ValueError, match='Resume data identity'):
        check_resume_data({'data_identity': identity}, changed)


def test_old_toolbench_configs_cannot_bypass_measured_input_requirements():
    required = prepared_requirements(['G1BallMove-L0-S'], {})
    assert required['history_source'] == 'measured_fullbody29_control_steps_v3'
    assert required['image_preprocessing'] == 'stretch_bilinear_antialias_224_v1'
    assert prepared_requirements(['move_plate'], {}) == {}


@pytest.mark.parametrize('workers', [0, 2])
def test_resume_reproduces_augmentation_across_epoch_boundary(tmp_path, workers):
    episode(tmp_path)
    dataset = KimodoWindowDataset([tmp_path], image_shift=2, proprio_mask_prob=.5)
    class Main(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)
    config = SimpleNamespace(main=Main(seed=13, batch_size=4, micro_batch_size=2,
                                      num_workers=workers, sampling='task_uniform'))
    full = list(islice(batches(dataset, config, 0), 11))
    resumed = list(islice(batches(dataset, config, 2), 7))
    for a, b in zip(full[4:], resumed):
        for key in ('egoview', 'condition_motion_mask', 'cut_index'):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_new_session_requires_reset_and_rejected_reset_does_not_consume_seed():
    class Policy:
        def __init__(self):
            self.seeds = []
        def reset(self, seed):
            self.seeds.append(seed)
        def __call__(self, request):
            if request.get('invalid'):
                raise ValueError('invalid request')
            return np.zeros((1, 36))
    policy = Policy()
    handlers = routes(policy, base_seed=100)
    act, release = handlers['POST', '/act'], handlers['POST', '/release']
    assert act({'session': 'a'})[0] == 400
    with pytest.raises(ValueError):
        act({'session': 'a', 'history': {'reset': True}, 'invalid': True})
    assert act({'session': 'a', 'history': {'reset': True}})[0] == 200
    assert policy.seeds == [100, 100]
    release({'session': 'a'})
    assert act({'session': 'b'})[0] == 400
    assert act({'session': 'b', 'history': {'reset': True}})[0] == 200


def test_tuning_uses_one_verified_dataset_for_all_execution_lengths(tmp_path):
    from omegaconf import OmegaConf
    import hashlib
    from kimodo_vla.benchmarks.toolbench.adapter import HISTORY_SOURCE, IMAGE_PREPROCESSING, CURRENT_OBSERVATION_SOURCE
    settings = dict(episodes=1, history_source=HISTORY_SOURCE, image_preprocessing=IMAGE_PREPROCESSING,
                    current_observation_source=CURRENT_OBSERVATION_SOURCE)
    path = tmp_path / 'preparation.json'
    path.write_text(json.dumps(settings))
    (tmp_path / 'preparation_complete.json').write_text(json.dumps(dict(
        passed=True, episodes=1, settings_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    search = object.__new__(Search)
    search.base = OmegaConf.create({'main': {'data_root': str(tmp_path)}})
    search.process = lambda *args, **kwargs: pytest.fail('verified cache must not be rebuilt')
    assert search.prepared(15) == search.prepared(50) == tmp_path
    path.write_text(json.dumps(dict(settings, episodes=2)))
    with pytest.raises(AssertionError):
        search.prepared(15)
