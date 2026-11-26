"""Experiment 1 follow-up, step 2: logged references as replay segments, the selection and the verdict."""
import numpy as np
import pytest

import replay_references as rr


def quat_z(yaw):
    return [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def test_reference_rows_become_body_frame_actions():
    from kimodo_vla.oracle.demo_walks import rot6d_rows
    rows = [{'reference_root_world': [1 + .01 * t, 2, .78 - .001 * t], 'reference_root_quat': quat_z(.5 + .01 * t),
             'reference_joints': list(np.full(29, .1 * t))} for t in range(5)]
    a = rr.to_actions(rows)
    assert a.shape == (5, 40) and np.allclose(a[:, 38:], 0)
    assert np.allclose(a[:, 2], [.78 - .001 * t for t in range(5)]) and np.allclose(a[3, 9:38], .3)
    yaws = np.arctan2(rot6d_rows(a)[:, 1, 0], rot6d_rows(a)[:, 0, 0])
    assert np.allclose(yaws, [.01 * t for t in range(5)])          # relative to the first heading
    # a world step of 1 cm along x, seen from a body turned by 0.5 + 0.01 t rad
    assert np.allclose(a[1, :2], [.01 * np.cos(.51), -.01 * np.sin(.51)])


def test_selection_follows_the_fixed_rule():
    reach = [{'run': f'exp1-map-{i:02d}', 'group': 'box · static-stand', 'cell': 'exp1-map', 'first_success': False,
              'mode': 'never_closed', 'class': 'holdable'} for i in range(10)]
    reach += [{'run': f'exp1-stance-{i}', 'group': 'box · static-stand', 'cell': 'exp1-stance', 'first_success': i > 2,
               'mode': 'never_closed', 'class': 'holdable'} for i in range(6)]
    reach += [{'run': f'exp1-map-ok{i}', 'group': 'box · static-stand', 'cell': 'exp1-map', 'first_success': True,
               'mode': 'closed_first', 'class': 'holdable'} for i in range(5)]
    reach.append({'run': 'x', 'skip': 'broken'})
    chosen = rr.select(reach)
    assert [r['run'] for o, r in chosen if o == 'fail'] == ['exp1-map-00', 'exp1-map-03', 'exp1-map-06', 'exp1-map-09',
                                                           'exp1-stance-0', 'exp1-stance-1']
    assert [r['run'] for o, r in chosen if o == 'ok'] == ['exp1-map-ok0', 'exp1-map-ok4', 'exp1-stance-3']


def test_verdict():
    yes, no = {'follows': True}, {'follows': False}
    assert rr.verdict(no, yes) == 'SONIC'
    assert rr.verdict(no, no).startswith('reference')
    assert rr.verdict(yes, no) == 'does not reproduce open loop'
    assert rr.verdict(None, yes) == 'incomplete'
