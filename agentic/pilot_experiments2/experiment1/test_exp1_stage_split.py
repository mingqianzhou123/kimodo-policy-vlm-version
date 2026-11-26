"""Experiment 1 stage split: statistics, timing rules and one synthetic episode end to end."""
import json
import math

import numpy as np
import pytest

import stage_split as ss


# ---- statistics ----------------------------------------------------------------------------------
def test_auc_counts_ties_half_and_orders_failures_first():
    assert ss.auc([3, 4], [1, 2]) == 1.
    assert ss.auc([1, 2], [3, 4]) == 0.
    assert ss.auc([1, 1], [1, 1]) == .5
    assert math.isnan(ss.auc([], [1]))


def test_mannwhitney_matches_scipy():
    scipy_stats = pytest.importorskip('scipy.stats')
    rng = np.random.default_rng(1)
    a, b = rng.normal(1, 1, 12), rng.normal(0, 1, 25)
    a[:3] = b[:3]   # ties
    ours = ss.mannwhitney_p(a, b)
    ref = scipy_stats.mannwhitneyu(a, b, alternative='two-sided', method='asymptotic').pvalue
    assert ours == pytest.approx(ref, rel=1e-6)


def test_wilson_interval():
    lo, hi = ss.wilson(27, 30)
    assert lo == pytest.approx(.7438, abs=1e-3) and hi == pytest.approx(.9654, abs=1e-3)


def test_separation_rule_needs_both_outcomes_and_a_clear_auc():
    base = {'n_fail': 10, 'n_succ': 20, 'auc': .85, 'auc_ci': [.7, .95], 'p': .001}
    assert ss.separates(base) == 'yes'
    assert ss.separates({**base, 'auc': .2, 'auc_ci': [.05, .4]}) == 'yes'
    assert ss.separates({**base, 'p': .02}) == 'no'
    assert ss.separates({**base, 'auc': .65}) == 'no'
    assert ss.separates({**base, 'n_fail': 4}) == 'too_few'


def test_verdict_reads_the_two_primary_stages():
    group = lambda s1, s2: {'metrics': {'P.wrist_pos.1': {'separates': s1}, 'P.wrist_pos.2': {'separates': s2}}}
    assert ss.verdict(group('no', 'yes')).startswith('stage 2')
    assert ss.verdict(group('yes', 'no')).startswith('stage 1')
    assert ss.verdict(group('no', 'no')) == 'neither stage separates'
    assert ss.verdict(group('too_few', 'yes')).startswith('cannot conclude')


# ---- geometry ----------------------------------------------------------------------------------------
def test_rotations_match_the_policy_definitions():
    Rotation = pytest.importorskip('scipy.spatial.transform').Rotation
    assert np.allclose(ss.pitch_down(20), Rotation.from_euler('y', np.radians(20)).as_matrix())
    q = Rotation.from_euler('z', .3).as_quat()   # x, y, z, w
    assert np.allclose(ss.quat_matrix([q[3], *q[:3]]), Rotation.from_quat(q).as_matrix())
    assert ss.angle_deg(np.eye(3), Rotation.from_euler('x', 10, degrees=True).as_matrix()) == pytest.approx(10)


def test_hammer_rotation_is_the_policy_formula():
    pytest.importorskip('torch')
    from kimodo_vla.oracle.hammer_policy import grasp_rotation
    axis = np.array([np.cos(.4), np.sin(.4), 0.])
    assert np.allclose(ss.hammer_rotation(axis, 40), grasp_rotation(axis, 40))


# ---- timing rules --------------------------------------------------------------------------------------
def steps_of(rows):
    return [{'time': i * .02, 'round': r, 'phase': p, 'left_n': l, 'right_n': rn, 'hands_closed': c}
            for i, (r, p, l, rn, c) in enumerate(rows)]


def test_contact_close_and_pre_contact():
    steps = steps_of([(0, 'raise', 0, 0, False), (0, 'advance', 0, .01, False), (0, 'advance', 0, .2, False),
                      (0, 'advance', 0, .3, True), (1, 'advance', 5, 5, True)])
    assert ss.first_contact(steps, (0, 1), .05) == 2
    assert ss.first_contact(steps, (0,), .05) is None          # the left hand never touches in round 0
    assert ss.first_contact(steps, (0, 1), 1.) is None
    assert ss.first_close(steps) == 3
    assert ss.pre_contact(steps, 2, 3) == 1                     # before the earlier event
    assert ss.pre_contact(steps, None, 3) == 2
    assert ss.pre_contact(steps, None, None) == 3               # last reach step of round 0
    assert ss.failure_mode(2, 3) == 'touched_before_close'
    assert ss.failure_mode(4, 3) == 'closed_first' and ss.failure_mode(None, None) == 'never_closed'


def test_newest_plan_wins_and_targets_follow_updates():
    rows = [{'plan': 0, 'time': .02, 'v': 'old'}, {'plan': 0, 'time': .04, 'v': 'old'}, {'plan': 1, 'time': .04, 'v': 'new'}]
    latest = ss.latest_commands(rows)
    assert latest[1]['v'] == 'old' and latest[2]['v'] == 'new'
    rounds = [{'round': 0, 'time': 1., 'wrist_targets': {'pre': 'a'}},
              {'round': 0, 'time': 1., 'now': 2., 'wrist_targets': {'pre': 'b'}},
              {'round': 1, 'time': 3., 'wrist_targets': {'pre': 'c'}}]
    targets = ss.round_targets(rounds)
    assert [t[1]['pre'] for t in targets] == ['a', 'b']
    assert ss.target_at(targets, 1.5)[0]['pre'] == 'a' and ss.target_at(targets, 2.)[0]['pre'] == 'b'


# ---- one synthetic episode ------------------------------------------------------------------------------
class FlatFK:
    """Wrists at root + (0, +-0.2, 0) + joints[0] along x; torso turned by joints[1] about z."""
    def __call__(self, joints, root, rotation, body_names=None):
        q = np.asarray(joints)
        positions = [root + [q[0], .2, 0], root + [q[0], -.2, 0], root]
        c, s = np.cos(q[1]), np.sin(q[1])
        torso = rotation @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return np.array(positions), np.array([rotation, rotation, torso])


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def make_run(tmp_path, success=False):
    run = tmp_path / 'exp1-static-20261124-r1-s1'
    run.mkdir()
    pitch = ss.pitch_down(20)
    pre = [[.5, .2, 0.], [.5, -.2, 0.]]
    write_jsonl(run / 'rounds.jsonl', [
        {'kind': 'latch', 'round': 0, 'phase': 'wait', 'time': 0., 'box_pos': [.6, 0, 0]},
        {'kind': 'phase', 'round': 0, 'phase': 'raise', 'time': .02, 'box_pos': [.6, 0, 0], 'anchor_root': [0, 0, 0],
         'raise_end': .03, 'advance_end': .04, 'wrist_targets': {'via': pre, 'pre': pre}}])
    (run / 'grasp-result.json').write_text(json.dumps({'success': success, 'first_attempt_success': success,
                                                       'seed': 1, 'layout': None, 'attempts': [{'label': 'touched_not_held'}]}))
    # Box at x=.6; it is touched at step 4, pushed 3 cm by step 6 when the hands close.
    box = [[.6, 0, 0]] * 5 + [[.615, 0, 0], [.63, 0, 0]]
    forces = [0, 0, 0, 0, .1, .2, .3]
    write_jsonl(run / 'grasp-steps.jsonl', [
        {'time': .02 * (i + 1), 'round': 0, 'phase': 'advance', 'left_n': f, 'right_n': 0., 'robot_other_n': 0.,
         'hands_closed': i == 6, 'box_com': b} for i, (f, b) in enumerate(zip(forces, box))])
    rows, commands = [], []
    for i in range(7):
        t = .02 * (i + 1)
        # sent / consumed reference reaches x = .45 (5 cm short of the target); the robot lags 2 cm behind it
        actual_x = .43
        rows.append({'after': {'time': t, 'joints': [actual_x] + [0.] * 28, 'root_pos': [0, 0, 0],
                               'root_quat': [1, 0, 0, 0], 'wrist_pos': [[actual_x, .2, 0], [actual_x, -.2, 0]],
                               'wrist_rot': [pitch.tolist(), pitch.tolist()]},
                     'reference_joints': [.45] + [0.] * 28, 'reference_root_world': [0, 0, 0],
                     'reference_root_quat': [1, 0, 0, 0]})
        commands.append({'plan': 0, 'time': t, 'raw_joints': [.39] + [0.] * 28, 'raw_root_world': [.01, 0, 0],
                         'raw_root_rotation_world': np.eye(3).tolist(), 'sent_joints': [.45] + [0.] * 28,
                         'sent_root_world': [0, 0, 0], 'sent_root_rotation_world': np.eye(3).tolist()})
    write_jsonl(run / 'execution.jsonl', rows)
    write_jsonl(run / 'commands.jsonl', commands)
    write_jsonl(run / 'plans.jsonl', [{'plan': 0, 'time': .02, 'kimodo_fk_target_m': [[.1, .1], [.1, .1]],
                                       'projected_wrist_angle_deg': [5., 7.]}])
    return run


def test_episode_splits_the_stages(tmp_path):
    r = ss.episode(make_run(tmp_path), FlatFK())
    assert 'skip' not in r
    assert (r['family'], r['cell'], r['condition']) == ('box', 'exp1-static', 'static-stand')
    assert r['mode'] == 'touched_before_close' and r['first_label'] == 'touched_not_held'
    assert r['time_P'] == pytest.approx(.08)                    # the step before first contact (index 4)
    assert r['P.wrist_pos.1a'] == pytest.approx(.10)           # target .50 -> Kimodo .39 + root .01
    assert r['P.wrist_pos.1b'] == pytest.approx(.05)           # Kimodo .40 -> sent .45
    assert r['P.wrist_pos.handoff'] == pytest.approx(0.)
    assert r['P.wrist_pos.1'] == pytest.approx(.05)            # target -> consumed reference
    assert r['P.wrist_pos.2'] == pytest.approx(.02)            # consumed reference -> robot
    assert r['P.wrist_pos.total'] == pytest.approx(.07)
    assert r['P.wrist_rot.2'] == pytest.approx(20)             # reference wrists level, robot pitched 20 deg
    assert r['P.wrist_rot.total'] == pytest.approx(0, abs=1e-6)
    assert r['P.root_pos.1a'] == pytest.approx(.01) and r['P.root_pos.1b'] == pytest.approx(.01)
    assert r['P.torso_rot.2'] == pytest.approx(0, abs=1e-6)
    assert r['push_contact_to_close_m'] == pytest.approx(.03)
    assert r['staleness_P_m'] == pytest.approx(0.) and r['staleness_C_m'] == pytest.approx(.03)
    assert r['P.asked.kimodo_fk_target_m'] == pytest.approx(.1) and r['P.asked.kimodo_angle_deg'] == pytest.approx(6)
    assert r['W.wrist_pos.2'] == pytest.approx(.02) and r['C.wrist_pos.2'] == pytest.approx(.02)
    assert r['P.wrist_pos.lag'] == pytest.approx(0.) and r['P.wrist_pos.correction'] == pytest.approx(0.)
    assert r['correction_plans'] == 0


# ---- the planned path, the replayed correction and the walk ----------------------------------------------
def test_planned_path_follows_the_smoothstep_and_slerps():
    Rz = lambda deg: np.array([[np.cos(np.radians(deg)), -np.sin(np.radians(deg)), 0],
                               [np.sin(np.radians(deg)), np.cos(np.radians(deg)), 0], [0, 0, 1]])
    pieces = [(1., np.array([[1., 0, 0]] * 2), np.stack([Rz(90)] * 2))]
    rest_pos, rest_rot = np.zeros((2, 3)), np.stack([np.eye(3)] * 2)
    pos, rot = ss.planned_wrists(pieces, 0., rest_pos, rest_rot, .5)
    assert np.allclose(pos[0], [.5, 0, 0]) and np.allclose(rot[1], Rz(45))
    pos, rot = ss.planned_wrists(pieces, 0., rest_pos, rest_rot, 3.)
    assert np.allclose(pos[0], [1, 0, 0]) and np.allclose(rot[0], Rz(90))
    assert ss.planned_wrists(pieces, 0., rest_pos, rest_rot, .2)[0][0, 0] == pytest.approx(ss.smooth(.2))


def test_correction_replay_is_bounded_and_applies_from_the_next_step():
    history = ss.replay_bias([1., 1.1], wanted=lambda t: np.array([[1., 0, 0], [0, 0, 0]]),
                             measured=lambda t: np.zeros((2, 3)), hands=(0, 1))
    assert np.allclose(history[0][1][0], [.12, 0, 0])            # 0.7 m wanted, clipped to 0.12 m
    assert np.allclose(history[0][1][1], 0.)
    assert np.allclose(ss.bias_at(history, 1.0), 0.)              # the plan made at 1.0 acts from 1.02
    assert np.allclose(ss.bias_at(history, 1.02)[0], [.12, 0, 0])
    one = ss.replay_bias([1.], lambda t: np.ones((2, 3)), lambda t: np.zeros((2, 3)), hands=(1,))
    assert np.allclose(one[0][1][0], 0.) and np.linalg.norm(one[0][1][1]) == pytest.approx(.12)


def test_walk_is_compared_plan_by_plan_in_displacements(tmp_path):
    run = tmp_path / 'walk'
    run.mkdir()
    write_jsonl(run / 'plans.jsonl', [{'kind': 'walk', 'round': 0, 'time': 0., 'end_error_m': .1},
                                      {'kind': 'walk', 'round': 0, 'time': 1., 'end_error_m': .3}])
    rounds = [{'round': 0, 'phase': 'settle', 'time': 2.}, {'round': 0, 'phase': 'raise', 'time': 2.5, 'walk_goal': [1., 0.]}]
    # The reference root starts 3 m away from the robot: only displacements count.
    row = lambda r, a: {'reference_root_world': [3 + r, 0, 0], 'after': {'root_pos': [a, 0, 0]}}
    execution = {0: row(0, 0), 50: row(.5, .4), 100: row(1., .8)}
    out = ss.walk_split(run, rounds, execution)
    assert out['walk.root_pos.1'] == pytest.approx(.2) and out['walk.root_pos.2'] == pytest.approx(.1)
    assert out['walk.travel_ratio'] == pytest.approx(.8) and out['walk.root_pos.total'] == pytest.approx(.2)
    assert out['walk.native'] is False and out['walk.plans'] == 2


def test_summary_groups_and_compares(tmp_path):
    records = []
    for i in range(12):
        ok = i % 2 == 0
        records.append({'run': f'r{i}', 'family': 'box', 'cell': 'exp1-static', 'condition': 'static-stand',
                        'first_success': ok, 'contact_before_close': False, 'closed': True, 'commands_at_P': True,
                        'mode': 'closed_first', 'P.wrist_pos.1': .01, 'P.wrist_pos.2': (.01 if ok else .05) + i * 1e-4})
    records.append({'run': 'bad', 'skip': 'unreadable: x'})
    (group,) = ss.summarize(records)
    assert group['n'] == 12 and group['first_success'] == 6
    assert group['metrics']['P.wrist_pos.2']['auc'] == 1. and group['metrics']['P.wrist_pos.2']['separates'] == 'yes'
    assert group['metrics']['P.wrist_pos.1']['separates'] == 'no'
    assert ss.verdict(group).startswith('stage 2')
