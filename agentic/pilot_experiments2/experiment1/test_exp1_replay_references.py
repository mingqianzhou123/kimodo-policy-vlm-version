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


def test_scene_placement_from_the_original_run(tmp_path):
    import json
    a, b = tmp_path / 'map', tmp_path / 'stance'
    a.mkdir(), b.mkdir()
    (a / 'run.json').write_text(json.dumps({'seed': 7003, 'layout_request': {'cell': {'depth': .08, 'lateral': .1, 'yaw': 0., 'extra': 0.}}}))
    (b / 'run.json').write_text(json.dumps({'seed': 3000, 'layout_request': None, 'start_at_stance': True}))
    assert rr.placement(a) == ['--seed', '7003', '--box-at', '0.08', '0.1', '0.0', '0.0']
    assert rr.placement(b) == ['--seed', '3000', '--start-at-stance']


def test_scene_contacts_and_reading():
    rows = [{'after': {'time': t, 'root_pos': [-b, 0, .78]}, 'body_contacts_n': c, 'box_contact_n': {'left_n': 0.}}
            for t, b, c in ((1., 0., {}), (1.2, .01, {'right_hand_index_1_link': 150.}), (1.4, .06, {'right_hand_index_1_link': 90.}))]
    out = rr.contacts(rows, 1., 2., np.array([1., 0.]), np.array([0., 0.]))
    assert out['peak_contact_n'] == {'right_hand_index_1_link': 150.}
    assert out['first_contact']['body'] == 'right_hand_index_1_link' and out['contact_before_backoff'] is True
    assert rr.scene_reading({'max_backoff_m': .2, 'contact_before_backoff': True}) == 'back-off with contact first'
    assert rr.scene_reading({'max_backoff_m': .2, 'contact_before_backoff': False}) == 'back-off without prior contact'
    assert rr.scene_reading({'max_backoff_m': .01}) == 'no back-off'
