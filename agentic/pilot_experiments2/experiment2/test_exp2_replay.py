"""Experiment 2 replays: resampling, the served timeline and the per-episode verdicts; no simulator."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from exp2_replay import BLEND_S, EpisodeReference, RATE, placement_error, resample, verdicts
from kimodo_vla.oracle.demo_walks import integrate_root


def walk_actions(stamps, speed=.3, yaw_rate=.2):
    """40-d actions of a root walking forward while turning, sampled at ``stamps``."""
    n = len(stamps)
    yaw = yaw_rate*stamps
    r = Rotation.from_euler('z', yaw[:, None]).as_matrix()
    world = np.zeros((n, 3))
    for t in range(1, n):
        world[t] = world[t-1]+r[t-1] @ [speed*(stamps[t]-stamps[t-1]), 0., 0.]
    a = np.zeros((n, 40))
    step = np.zeros_like(world)
    step[1:] = world[1:]-world[:-1]
    a[:, :2] = np.einsum('tji,tj->ti', r, step)[:, :2]
    a[:, 2] = .78
    a[:, 3:9] = r[:, :, :2].reshape(n, 6)
    a[:, 9:38] = np.linspace(0, 1, n)[:, None]
    a[:, 38] = stamps > stamps[-1]/2
    return a, world


def test_resampling_keeps_the_path_and_the_timing():
    stamps = 1000.+np.cumsum(np.r_[0., np.full(59, 1/12)])          # 12 Hz, 5 s
    a, world = walk_actions(stamps)
    out = resample(a, stamps)
    assert len(out) == int(round((stamps[-1]-stamps[0])*RATE))+1
    path = integrate_root(out)
    np.testing.assert_allclose(path[-1, :2], world[-1, :2], atol=2e-3)
    np.testing.assert_allclose(out[:, 9], np.linspace(0, 1, len(out)), atol=1e-6)
    assert out[0, 38] == 0 and out[-1, 38] == 1


def test_served_timeline_keeps_the_recorded_hands(tmp_path):
    a = np.zeros((100, 40))
    a[:, 2], a[:, 3], a[:, 6] = .78, 1., 1.
    a[40:, 38:40] = 1.
    episode = tmp_path/'e.json'
    episode.write_text(__import__('json').dumps({'episode': 'x', 'actions': a.tolist()}))
    policy = EpisodeReference(tmp_path, episode)
    first = {'time': 2., 'root_pos': [0., 0., .78], 'root_quat': [1., 0., 0., 0.], 'joints': [0.]*29,
             'wrist_pos': [[0., .2, 1.], [0., -.2, 1.]], 'wrist_rot': [np.eye(3).tolist()]*2}
    chunk = np.asarray(policy({'observation': {'oracle_history': [first]}})['action_chunk'])
    assert (chunk[:, 38:40] == 0).all()
    later = dict(first, time=2.+(BLEND_S+.9))                        # timeline index 50 + 45: recording frame 45
    chunk = np.asarray(policy({'observation': {'oracle_history': [later]}})['action_chunk'])
    assert (chunk[:, 38:40] == 1).all()


def test_open_loop_serving_walks_the_cursor(tmp_path):
    a = np.zeros((60, 40))
    a[:, 9] = np.arange(60)
    episode = tmp_path/'e.json'
    episode.write_text(__import__('json').dumps({'episode': 'x', 'actions': a.tolist()}))
    policy = EpisodeReference(tmp_path, episode)
    first = np.asarray(policy({'observation': {}})['action_chunk'])
    second = np.asarray(policy({'observation': {}})['action_chunk'])
    assert first[0, 9] == 0 and second[0, 9] == 25 and len(second) == 25


def test_verdicts_need_every_repeat_to_succeed():
    rows = [{'episode': 'a', 'success': True}]*3+[{'episode': 'b', 'success': True}, {'episode': 'b', 'success': False},
                                                  {'episode': 'b', 'success': True}]+[{'episode': 'c', 'success': False}]*3
    table = verdicts(rows, 3)
    assert table['a']['in_V'] and not table['b']['in_V'] and not table['b']['agree'] and table['c']['agree']


def test_placement_error():
    episode = {'objects': {'box': {'position': [1., 2., .75], 'orientation': [1., 0., 0., 0.]}}}
    x, y, z, w = Rotation.from_euler('z', 10, degrees=True).as_quat()
    error = placement_error({'box_pos': [1., 2.01, .75], 'box_quat': [w, x, y, z]}, episode, 'box')
    assert error['position_m'] == pytest.approx(.01) and error['angle_deg'] == pytest.approx(10.)
