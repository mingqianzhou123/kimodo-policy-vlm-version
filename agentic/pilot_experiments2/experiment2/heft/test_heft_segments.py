"""The HITCH-segment encoder decodes back through Arena's own integration (demo_walks.integrate_root)."""
import sys
from pathlib import Path

import numpy as np
import pytest

Rotation = pytest.importorskip('scipy.spatial.transform').Rotation
sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_encode_round_trip_through_arena_integration():
    from heft_segments import encode
    from kimodo_vla.oracle.demo_walks import integrate_root, rot6d_rows
    t = np.linspace(0, 1, 30)
    yaw = .3+np.pi/2*t                        # turning while walking, from a start heading of 0.3 rad
    pos = np.stack([.4*t+1., .2*t**2-2., .75-.1*t], 1)
    quat = Rotation.from_euler('zy', np.stack([yaw, .05*t], 1)).as_quat(scalar_first=True)
    names = [f'j{i}' for i in range(29)]
    world = np.hstack([pos, quat, np.tile(np.arange(29.), (30, 1))])
    a = encode(world, names, names[::-1])
    head = Rotation.from_euler('z', -.3).as_matrix()
    assert np.allclose(integrate_root(a), np.c_[(head @ (pos-pos[0]).T).T[:, :2], pos[:, 2]], atol=1e-9)
    assert np.allclose(rot6d_rows(a), head @ Rotation.from_quat(quat, scalar_first=True).as_matrix(), atol=1e-9)
    assert a[0, 9:38].tolist() == list(np.arange(29.)[::-1])
