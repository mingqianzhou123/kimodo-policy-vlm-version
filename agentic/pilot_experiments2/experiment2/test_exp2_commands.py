"""Experiment 2 command test: start perturbation, stop criterion and deviation arithmetic."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from exp2_commands import MOTIONS, SEEDS, START_RADIUS_M, goal_for, measure, start_offset, stop_index
from kimodo_vla.oracle.locomotion import load_goal


def quat(yaw_deg):
    x, y, z, w = Rotation.from_euler('z', yaw_deg, degrees=True).as_quat()
    return [w, x, y, z]


def walk(distance, yaw_deg=0., heading_deg=20., seconds=6., move_s=3.):
    """A root that moves ``distance`` along its heading in ``move_s`` seconds, then stands."""
    heading = Rotation.from_euler('z', heading_deg, degrees=True).as_matrix()
    out = []
    for t in np.arange(0, seconds, .02):
        u = min(t/move_s, 1.)
        xy = heading[:2, 0]*distance*u
        out.append({'time': 2.+t, 'root_pos': [1.+xy[0], 2.+xy[1], .78],
                    'root_quat': quat(heading_deg+yaw_deg*u), 'wrist_pos': [[0, 0, 0], [0, 0, 0]],
                    'wrist_rot': [np.eye(3).tolist()]*2})
    return out, heading


def test_start_offsets_are_fixed_small_and_distinct():
    offsets = [start_offset(s) for s in SEEDS]
    assert all(np.linalg.norm(o) <= START_RADIUS_M for o in offsets)
    assert len({tuple(np.round(o, 6)) for o in offsets}) == len(SEEDS)
    np.testing.assert_array_equal(start_offset(SEEDS[0]), start_offset(SEEDS[0]))


def test_every_motion_is_a_valid_goal():
    for motion in MOTIONS:
        load_goal(goal_for(motion))


def test_stop_is_found_after_the_last_phase_and_deviation_is_actual_minus_commanded():
    samples, heading = walk(.45)
    out = measure(samples, heading, MOTIONS['forward'][1], last_phase_time=4.)
    assert out['stopped'] and not out['fell']
    assert out['measured_at_s'] >= 3.+.5-.1           # still from 3 s on, measured after 0.5 s of it
    assert out['root_along_m'] == pytest.approx(-.05, abs=1e-6)
    assert out['root_across_m'] == pytest.approx(0., abs=1e-6)
    assert out['yaw_deg'] == pytest.approx(0., abs=1e-6)


def test_turn_deviation_and_not_stopped():
    samples, heading = walk(0., yaw_deg=80.)
    out = measure(samples, heading, MOTIONS['turn'][1], last_phase_time=2.)
    assert out['yaw_deg'] == pytest.approx(-10., abs=1e-6)
    moving, heading = walk(.5, seconds=3., move_s=3.)   # never still
    assert not measure(moving, heading, MOTIONS['forward'][1], 2.)['stopped']


def test_stop_index_needs_the_full_window():
    time = np.arange(0, 1., .02)
    root = np.zeros((len(time), 3))
    yaw = np.zeros(len(time))
    assert stop_index(time, root, yaw, 0.) == 24
    assert stop_index(time, root, yaw, .6) == 30+24
    assert stop_index(time[:20], root[:20], yaw[:20], 0.) is None
