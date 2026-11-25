"""Our port of official Kimodo generation against outputs of NVIDIA's own pipeline (fixtures/official_kimodo)."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import native_kimodo as nk
from kimodo_vla.motion.representation.kimodo_motionrep import KimodoMotionRep
from kimodo_vla.skeleton.definitions import G1Skeleton34

FIXTURES = sorted(nk.FIXTURES.glob('*.npz'))


@pytest.fixture(scope='module')
def rep():
    return KimodoMotionRep(G1Skeleton34(), fps=30, stats_path=None)


def test_fixtures_present_and_small():
    assert {p.stem for p in FIXTURES} == {'text-forward', 'path-forward', 'two-prompts'}
    assert all(p.stat().st_size < 1_000_000 for p in FIXTURES)


@pytest.mark.parametrize('path', FIXTURES, ids=lambda p: p.stem)
def test_constraints_match_official(rep, path):
    """Our builders write the same constrained cells and values as the official constraint sets."""
    fx = nk.load_fixture(path)
    for i, (values, mask, heading) in enumerate(nk.our_conditions(rep, fx)):
        official = nk.official_condition(fx, i, rep.motion_rep_dim)
        if official is None:
            assert not mask.any()
            continue
        assert torch.equal(mask, official[1])
        assert float((values - official[0]).abs().max()) < 1e-4
        assert abs(heading - float(fx[f'heading{i}'])) < 1e-5


@pytest.mark.parametrize('path', FIXTURES, ids=lambda p: p.stem)
def test_stitching_matches_official(rep, path):
    """Joining the official segments our way gives the official final motion."""
    fx = nk.load_fixture(path)
    joints = nk.decoded_joints(rep, nk.stitched(rep, fx))
    assert joints.shape == fx['posed_joints'].shape
    assert float((joints - torch.from_numpy(fx['posed_joints'])).abs().max()) < 1e-4


CHECKPOINT = Path(os.environ.get('KVLA_CHECKPOINTS', '/nonexistent'))/'Kimodo-G1-RP-v1'


@pytest.mark.skipif(not CHECKPOINT.is_dir(), reason='needs the Kimodo-G1-RP-v1 weights (KVLA_CHECKPOINTS)')
@pytest.mark.parametrize('path', FIXTURES, ids=lambda p: p.stem)
def test_port_reproduces_official(path):
    """From the same noise our sampler gives the official motion (tolerance covers CPU vs GPU arithmetic)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = nk.load_model(CHECKPOINT, device)
    row = nk.compare(model, nk.load_fixture(path))
    print(row)
    assert row['raw_max_abs_normalized'] < 5e-3
    assert row['joints_max_abs_m'] < 2e-3
