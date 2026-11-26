"""kimodo_vla.controllers.heft: the parts that do not need the checkpoint.

The full observation was checked against the release's own code (``TrackingPolicyRaw``) at every control step of
seven motions in MuJoCo (agentic/pilot_experiments2/experiment2/heft/heft_mujoco.py): largest difference 1.3e-5.
"""
import numpy as np
import pytest

from kimodo_vla.controllers.heft import HEFT_JOINT_NAMES, HeftSpec, HeftTracker, Reference, quat_rotate_inverse, quat_to_matrix
from kimodo_vla.core.types import G1_JOINT_NAMES

Rotation = pytest.importorskip('scipy.spatial.transform').Rotation


def test_checkpoint_order_is_the_arena_action_order():
    # the 40-D action's 29 joints and the checkpoint's action joints are the same list: no reordering in the port
    assert HEFT_JOINT_NAMES == G1_JOINT_NAMES


def test_quaternion_helpers_match_scipy_wxyz():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(5, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    v = rng.normal(size=(5, 3))
    r = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    assert np.allclose(quat_to_matrix(q), r.as_matrix())
    assert np.allclose(quat_rotate_inverse(q, v), r.inv().apply(v))


def test_reference_clamps_and_holds_the_tail():
    ref = Reference(np.arange(3)[:, None]*[1., 0, 0], np.tile([1., 0, 0, 0], (3, 1)), np.zeros((3, 2)))
    pos, _, _ = ref.window([0, 1, 4, -2])
    assert pos[:, 0].tolist() == [0, 1, 2, 0]
    for _ in range(5):
        ref.advance()
    assert ref.cursor == 2   # stops at the last frame, as update_obs does


def spec(n=2, future=(0, 1, -1)):
    z = np.zeros(n)
    return HeftSpec(tuple(f'j{i}' for i in range(n)), z+10, z+1, z+.1, z+.5, 10., 50, future, (0, 2), (0, 1), (0, 1),
                    (0,), 2, 5, False, None)


def test_observation_layout_and_size():
    s = spec()
    t = HeftTracker.__new__(HeftTracker)
    t.spec = s
    t.reset()
    ref = Reference([[0, 0, .7], [.1, 0, .7]], [[1, 0, 0, 0]]*2, [[.1, .2], [.3, .4]])
    obs = t.observe(ref, q=[.5, .6], dq=[0, 0], quat_wxyz=[1, 0, 0, 0], gyro_body=[0, 0, .3])
    assert obs.shape == (s.obs_size(),) == (1+2*3+3*6+3*2+3+3*3+2*3+2*3+2*2+1*2+2*2,)
    assert obs[0] == pytest.approx(4/5)                  # boot indicator counts down before the first observation
    assert obs[1:4].tolist() == pytest.approx([.1, 0, 0])   # future frame 1 relative to the current one
    assert obs[4:7].tolist() == pytest.approx([0, 0, 0])    # history frame -1 clamps to the current one
    rot6d = obs[7:25].reshape(3, 6)
    assert np.allclose(rot6d, [1, 0, 0, 0, 1, 0])          # first column, then second column (not Arena's row layout)


def test_action_scaling_and_clip():
    s = spec()
    t = HeftTracker.__new__(HeftTracker)
    t.spec = s
    t.reset()

    class Session:
        def get_inputs(self):
            return [type('I', (), {'name': 'policy'})]

        def run(self, _, feed):
            return [np.array([[30., -1.]], dtype=np.float32)]
    t._session = Session()
    target = t.act(np.zeros(s.obs_size(), np.float32))
    assert target.tolist() == pytest.approx([.1+.5*10, .1-.5])   # default + scale * clip(raw, +-10)
    assert t.last_action.tolist() == pytest.approx([10, -1])     # the clipped raw action is the next "previous"
