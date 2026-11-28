"""The squat goal added for experiment 2's command test: pose geometry and goal validation."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.model.features import G1_XML_PATH
from kimodo_vla.oracle.geometry import SIDES, XmlFK
from kimodo_vla.oracle.locomotion import LEAN_STEPS, episode_seconds, load_goal, squatted_pose

FEET = [f'{s}_ankle_roll_link' for s in SIDES]


def standing():
    rotation = Rotation.from_euler('z', .4).as_matrix() @ Rotation.from_euler('y', .03).as_matrix()
    root, joints = np.array([-4., -4., .78]), np.zeros(29)
    joints[[G1_JOINT_NAMES.index(f'{s}_knee_joint') for s in SIDES]] = .3
    joints[[G1_JOINT_NAMES.index(f'{s}_hip_pitch_joint') for s in SIDES]] = -.15
    joints[[G1_JOINT_NAMES.index(f'{s}_ankle_pitch_joint') for s in SIDES]] = -.15
    return joints, root, rotation


@pytest.mark.parametrize('drop', [.05, .15])
def test_squat_lowers_the_pelvis_upright_over_unmoved_feet(drop):
    fk = XmlFK(G1_XML_PATH, G1_JOINT_NAMES)
    joints, root, rotation = standing()
    squatted, new_root, new_rotation = squatted_pose(fk, joints, root, rotation, drop, FEET)
    before = fk(joints, root, rotation, body_names=FEET+['torso_link'])
    after = fk(squatted, new_root, new_rotation, body_names=FEET+['torso_link'])
    np.testing.assert_allclose(after[0][:2], before[0][:2], atol=1e-6)            # both ankles stay put
    np.testing.assert_allclose(after[1][:2], before[1][:2], atol=1e-6)            # and keep their orientation
    np.testing.assert_allclose(new_rotation, rotation, atol=1e-4)                 # pelvis keeps its orientation
    assert root[2]-new_root[2] == pytest.approx(drop, abs=1e-3)
    heading = Rotation.from_euler('z', .4).as_matrix()
    assert abs((new_root-root) @ heading[:, 0]) < 1e-3                            # over the same ground point
    upper = [i for i, n in enumerate(G1_JOINT_NAMES) if not any(k in n for k in ('hip_pitch', 'knee', 'ankle_pitch'))]
    np.testing.assert_array_equal(squatted[upper], joints[upper])


def test_squat_goal_validation_and_length():
    goal = load_goal({'name': 'squat-15', 'squat': {'drop_m': .15}})
    assert goal == {'name': 'squat-15', 'squat': {'drop_m': .15}}
    lean = load_goal({'name': 'lean', 'lean': {'pitch_deg': 30, 'by': 'hips'}})
    assert episode_seconds(goal) == pytest.approx(episode_seconds(lean))
    assert episode_seconds(goal) > LEAN_STEPS/50
    for bad in ({'drop_m': 0.}, {'drop_m': .5}, {'drop_m': .1, 'x': 1}):
        with pytest.raises(ValueError):
            load_goal({'name': 'x', 'squat': bad})
    with pytest.raises(ValueError):
        load_goal({'name': 'x', 'squat': {'drop_m': .1}, 'root': {'xy': [1, 0]}})
