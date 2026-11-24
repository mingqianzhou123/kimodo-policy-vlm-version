"""Walking on looped recorded strides (kimodo_vla/oracle/clip_legs.py), with synthetic cycles."""
import json

import numpy as np
import pytest

from kimodo_vla.oracle import clip_legs as cl
from kimodo_vla.oracle import demo_walks as dw
from rollout_kimodo import follow
from test_locomotion import make_policy


def stride(velocity, yaw_rate=0., n=50):
    """One synthetic cycle row set: constant body-frame velocity (m/s) and turn rate (deg/s), joints at 0.05."""
    a = np.zeros((n, 40))
    yaw = np.radians(yaw_rate)*np.arange(n)/dw.RATE
    r = np.stack([cl.rz(np.degrees(y)) for y in yaw])
    a[:, 3:9] = r[:, :, :2].reshape(n, 6)
    a[:, :2] = np.asarray(velocity)/dw.RATE
    a[:, 2], a[:, 9:38] = .78, .05
    return a


def library():
    out = {}
    for name, v in {'forward': [.5, 0], 'backward': [-.4, 0], 'left': [0, .4], 'right': [0, -.4]}.items():
        a = np.concatenate([stride(v), stride(v)[:1]])
        out[name] = [dict(cl.make_cycle(a, 0, 50, False), source=name)]
    a = np.concatenate([stride([0, 0], 60.), stride([0, 0], 60.)[:1]])
    out['turn-left'] = [dict(cl.make_cycle(a, 0, 50, True), source='turn')]
    return out


def test_cycles_loop_and_report_their_advance():
    lib = library()
    f = lib['forward'][0]
    assert len(f['actions']) == 50 and abs(f['advance_m'][0]-.5*49/50) < 1e-6 and f['yaw_deg'] == 0.
    assert abs(lib['turn-left'][0]['yaw_deg']-60*49/50) < 1e-3


def walk(goal_xy, goal_yaw=0., carry=1.):
    """A body that covers ``carry`` of each commanded step; returns the walker and where the body ended."""
    stance = np.zeros(40)
    stance[2], stance[3:9], stance[9:38] = .78, np.eye(3)[:, :2].reshape(6), 0.
    w = cl.ClipWalker(library(), goal_xy, goal_yaw, stance, tolerance_m=.1)
    xy, yaw, t = np.zeros(2), 0., 0.
    while not w.done and t < 60:
        rows = w.chunk(xy, yaw, t)
        rot = dw.rot6d_rows(rows)
        for a, r in zip(rows, rot):
            xy = xy+carry*(r[:2, :2] @ a[:2])
            yaw = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
            t += 1/dw.RATE
    return w, xy, yaw


@pytest.mark.parametrize('goal,within', [([.5, 0.], .2), ([0., .5], .2), ([0., -.5], .2), ([-.5, 0.], .2), ([.8, .3], .3)])
def test_walker_stops_near_a_ground_goal_even_when_the_tracker_covers_less(goal, within):
    # whole strides only: it stops within about half a stride per axis (strides here are 0.4-0.5 m)
    for carry in (1., .5):
        w, xy, _ = walk(goal, carry=carry)
        assert w.done and np.linalg.norm(xy-goal) < within, (carry, xy)


def test_walker_turns_left_to_a_heading():
    w, xy, yaw = walk([0., 0.], 90.)
    assert w.done and abs(yaw-90) < 35 and np.linalg.norm(xy) < .1


def test_policy_walks_on_cycles_then_holds(tmp_path):
    path = tmp_path/'cycles.json'
    path.write_text(json.dumps(library()))
    policy, start, rotation, root = make_policy(tmp_path, {'name': 'fwd', 'root': {'xy': [.5, 0.], 'tolerance_m': .1}})
    policy.clips = cl.load_library(path)
    rows = follow(policy, start, 8.)
    assert [p['phase'] for p in policy.phases] == ['walk', 'hold']
    assert policy.model.calls == []                            # Kimodo is never asked while walking on cycles
    moved = rotation[:2, :2].T @ (np.array(rows[-1]['body_xy'])-root[:2])
    assert abs(moved[0]-.5) < .2 and abs(moved[1]) < .05
    plans = [json.loads(x) for x in (tmp_path/'plans.jsonl').read_text().splitlines()]
    assert all(not p['model_called'] for p in plans)
