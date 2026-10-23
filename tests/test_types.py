import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from kimodo_vla.core.types import ActionChunk, G1_JOINT_NAMES, rot6d_to_matrix
from kimodo_vla.motion.g1_reference import CANONICAL_G1_JOINT_NAMES_29, rot6d_row_to_matrix


def random_chunk(frames=6, seed=0):
    rng = np.random.default_rng(seed)
    return ActionChunk(
        root_delta_xy=rng.normal(size=(frames, 2)).astype(np.float32),
        root_height=rng.uniform(0.6, 0.8, frames).astype(np.float32),
        root_rot=Rotation.random(frames, random_state=seed).as_matrix().astype(np.float32),
        joint_pos=rng.normal(size=(frames, 29)).astype(np.float32),
        hand=rng.integers(0, 2, (frames, 2)).astype(np.float32),
        fps=50.0,
    )


def test_joint_order_matches_released_checkpoints():
    assert G1_JOINT_NAMES == CANONICAL_G1_JOINT_NAMES_29


def test_array40_round_trip():
    chunk = random_chunk()
    back = ActionChunk.from_array40(chunk.to_array40(), fps=50.0)
    for name in ("root_delta_xy", "root_height", "root_rot", "joint_pos", "hand"):
        np.testing.assert_allclose(getattr(back, name), getattr(chunk, name), atol=1e-5)


def test_rot6d_matches_original_decoder():
    rot6d = random_chunk().to_array40()[:, 3:9]
    expected = rot6d_row_to_matrix(torch.from_numpy(rot6d)).numpy()
    np.testing.assert_allclose(rot6d_to_matrix(rot6d), expected, atol=1e-5)


def test_slicing_keeps_fields_aligned():
    chunk = random_chunk()
    tail = chunk[2:]
    assert len(tail) == 4
    np.testing.assert_array_equal(tail.joint_pos, chunk.joint_pos[2:])
    with pytest.raises(TypeError):
        chunk[0]


def test_shape_mismatch_is_rejected():
    chunk = random_chunk()
    with pytest.raises(ValueError):
        ActionChunk(chunk.root_delta_xy, chunk.root_height, chunk.root_rot, chunk.joint_pos[:, :28], chunk.hand, 50.0)
