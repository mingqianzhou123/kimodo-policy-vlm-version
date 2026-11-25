"""Target timing of the three update modes, the launcher flags, the runner names and the failure rules."""
import json

import numpy as np
import pytest

from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.model.features import G1_XML_PATH, KimodoFeatures
from kimodo_vla.oracle.async_grasp import AsyncBoxGraspPolicy, BoxLog, latency_draws, load_latencies
from kimodo_vla.oracle.geometry import SIDES, XmlFK
from test_grasp_loop import BOX, CFG, standing_sample
from run_oracle import parse_args
import analyze_exp4 as an
import run_exp4_suite as suite


def scripted(tmp_path, update, delay=75, latencies=None):
    policy = AsyncBoxGraspPolicy.__new__(AsyncBoxGraspPolicy)
    policy.cfg, policy.latch_delay, policy.run, policy.mode = CFG, delay, tmp_path, 'hold'
    policy.stand_still, policy.fixed_delay = True, delay
    policy.target_update, policy.latency_samples = update, latencies
    policy.query_interval, policy.retarget_tolerance, policy.perception = .5, .001, None
    policy.features, policy.fk = KimodoFeatures(50.), XmlFK(G1_XML_PATH, G1_JOINT_NAMES)
    names = policy.features.representation.skeleton.bone_order_names
    policy.ids = [[names.index(f'{s}_{j}_skel') for j in ('wrist_yaw', 'hand_roll')] for s in SIDES]
    policy.reset(6000)
    return policy


def sliding(t):
    """A box that moves 5 cm/s to the robot's right from time zero."""
    return [BOX[0], BOX[1] - .05 * t, BOX[2]]


def step(policy, t, closed=False, replan=False):
    sample = standing_sample(policy, t, sliding(t), closed)
    policy.advance([sample])
    if replan and policy.round.phase in ('raise', 'advance', 'grip', 'squeeze', 'lift', 'hold'):
        policy.correct(sample)
    return sample


def play(policy, until, closed_after=None, replan_every=.5):
    """50 Hz steps from 0.02 s; a replan every ``replan_every`` seconds after the reach began."""
    policy.begin_episode(standing_sample(policy, 0., sliding(0.)))
    begun = None
    for k in range(1, int(round(until * 50)) + 1):
        t = k * .02
        if begun is None and policy.round.phase == 'raise':
            begun = t - .02
        replan = begun is not None and abs(((t - begun) / replan_every) - round((t - begun) / replan_every)) < 1e-6
        step(policy, t, closed=closed_after is not None and t >= closed_after, replan=replan)
    return begun


def test_anchored_waits_one_delay_then_aims_at_where_the_box_is_now(tmp_path):
    policy = scripted(tmp_path, 'anchored', delay=75)
    begun = play(policy, 1.5 + 1.6)
    assert begun == pytest.approx(1.5)
    np.testing.assert_allclose(policy.round.box_pos, sliding(3.0), atol=1e-9)   # last replan at 3.0 s
    np.testing.assert_allclose(policy.way['pre'].mean(0)[1], sliding(3.0)[1])
    info = policy.grasp_info()
    assert info['event']['id'].startswith('round-0-u') and info['event']['positions'] == policy.way['pre'].tolist()
    assert policy.round.notes['target_info_time'] == pytest.approx(3.0)
    rows = [json.loads(l) for l in (tmp_path/'rounds.jsonl').read_text().splitlines() if '"update"' in l]
    assert [round(r['now'], 2) for r in rows] == [1.5, 2.0, 2.5, 3.0]
    assert all(r['info_time'] == pytest.approx(r['now']) for r in rows)


def test_fixed_aims_at_the_answer_one_delay_old(tmp_path):
    policy = scripted(tmp_path, 'fixed', delay=75)
    play(policy, 1.5 + 1.6)
    assert policy.round.notes['target_info_time'] == pytest.approx(1.5)        # asked at 1.5, answered at 3.0
    np.testing.assert_allclose(policy.round.box_pos, sliding(1.5), atol=1e-9)
    assert policy.belief(2.99)[2] == pytest.approx(1.0) and policy.belief(1.49) is None
    policy.episode_start = 1.0000004                                 # simulator clocks drift in the last digits
    assert policy.belief(2.5)[2] == pytest.approx(1.0)


def test_blocking_mode_keeps_the_latched_pose(tmp_path):
    policy = scripted(tmp_path, 'blocking', delay=75)
    begun = play(policy, 1.5 + 1.6)
    assert begun == pytest.approx(1.5)
    np.testing.assert_allclose(policy.round.box_pos, sliding(0.))
    np.testing.assert_allclose(policy.way['pre'].mean(0)[1], sliding(0.)[1])


def test_no_update_after_the_hands_close(tmp_path):
    policy = scripted(tmp_path, 'anchored', delay=25)
    play(policy, .5 + 2.5, closed_after=.5 + 2.2)
    assert policy.round.phase in ('grip', 'squeeze')
    aim = policy.round.box_pos
    step(policy, 3.02, closed=True, replan=True)
    assert policy.round.box_pos == aim


def test_later_asynchronous_rounds_do_not_wait(tmp_path):
    policy = scripted(tmp_path, 'anchored', delay=383)
    policy.begin_episode(standing_sample(policy, 0., BOX))
    policy.round.index = 0
    policy.begin_round(standing_sample(policy, 20., sliding(20.)))
    assert policy.round.index == 1 and policy.latch_delay == 0
    np.testing.assert_allclose(policy.round.box_pos, sliding(0.))   # nothing newer logged than the first sample


def test_latency_draws_are_shared_by_the_modes_and_fixed_may_hear_back_earlier(tmp_path):
    values = load_latencies(an.HERE/'gemini-latencies.json')
    assert len(values) == 80 and np.median(values) == pytest.approx(7.66, abs=.01)
    a, b = latency_draws(values, 6001), latency_draws(values, 6001)
    np.testing.assert_array_equal(a, b)
    first = {u: scripted(tmp_path, u, latencies=values) for u in ('blocking', 'fixed', 'anchored')}
    for policy in first.values():
        policy.reset(6001)
        policy.episode_start = 0.
    assert first['anchored'].first_answer() == pytest.approx(first['anchored'].latency(0))
    assert first['fixed'].first_answer() <= first['anchored'].first_answer()
    assert first['blocking'].latency(0) == first['anchored'].latency(0)


def test_box_log_returns_the_pose_at_or_before_a_time():
    log = BoxLog()
    for t in (0., .02, .04):
        log.add({'time': t, 'box_com': [t, 0, 0], 'box_quat': [1, 0, 0, 0]})
    assert log.at(.03)[0] == pytest.approx(.02) and log.at(.04)[1][0] == pytest.approx(.04)
    with pytest.raises(ValueError):
        log.at(-.1)


def test_launcher_flags():
    a = parse_args(['--run-id', 'x', '--grasp-loop', '--stand-still', '--box-in-region', '--target-update', 'anchored'])
    assert a.target_update == 'anchored' and a.latency_samples is None
    assert parse_args(['--run-id', 'x', '--grasp-loop']).target_update == 'blocking'
    with pytest.raises(ValueError):
        parse_args(['--run-id', 'x', '--grasp-loop', '--target-update', 'fixed'])


def test_runner_cells_and_names():
    assert suite.cell('anchored:194:0.1') == ('anchored', 194, .1)
    assert suite.name('fixed', 620, .03, 't', 6000) == 'exp4-fixed-d620-v003-t-s6000'
    assert suite.name('blocking', 'jitter', .1, 't', 6001) == 'exp4-blocking-jgem-v010-t-s6001'
    assert '--latency-samples' in suite.flags('fixed', 'jitter', .1)
    for bad in ('anchored:100:0.1', 'fixed:jitter:0.03', 'other:25:0.1', 'anchored:25:0.2'):
        with pytest.raises(SystemExit):
            suite.cell(bad)


def first_round(**kw):
    base = dict(label='knocked_over', reached=True, at=5., closed=True, backoff_m=.03, age_s=4.,
                target_lateral_m=0., hand_lateral_m=0., exec_lateral_m=0., box_speed_mps=0.)
    return {**base, **kw}


def test_failure_rules_follow_the_record():
    assert an.label(first_round(label='success'), 1, True) is None
    assert an.label(first_round(reached=False, at=None), 1, False) == 'delay_beyond_limit'
    assert an.label(first_round(closed=False, backoff_m=.2), 1, False) == 'backed_off'
    assert an.label(first_round(closed=False, box_speed_mps=.05), 1, False) == 'hands_lag'
    assert an.label(first_round(hand_lateral_m=.05, target_lateral_m=-.05, exec_lateral_m=0.), 1, False) == 'stale_target'
    assert an.label(first_round(hand_lateral_m=.05, target_lateral_m=-.04, exec_lateral_m=.01, age_s=.4), 1, False) == 'stale_by_replan'
    assert an.label(first_round(hand_lateral_m=.05, target_lateral_m=-.01, exec_lateral_m=.04), 1, False) == 'hands_lag'
    assert an.label(first_round(box_speed_mps=.08), 1, False) == 'moving_at_close'
    assert an.label(first_round(), 1, False) == 'other'


def test_the_target_in_force_is_the_last_aim_before_the_close():
    aims = [(0., [0, 0, 0], 0.), (1.5, [0, .1, 0], 1.5), (2., [0, .2, 0], 2.)]
    assert an.in_force(aims, 1.99)[1] == [0, .1, 0] and an.in_force(aims, 2.)[1] == [0, .2, 0]


# ---- part two: camera pose, frame points and the tracker's geometry ---------------------------

from scipy.spatial.transform import Rotation   # noqa: E402
from arena.camera_pose import backproject, compose, mount, project, transform   # noqa: E402
from arena.frame_stream import box_points   # noqa: E402
from kimodo_vla.vlm.box_tracker import box_pose, depth_at, extrapolate   # noqa: E402
from test_grasp_loop import quat   # noqa: E402

K = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1]])


def test_the_mount_carries_the_camera_with_its_link():
    link0 = transform([1., 2., 1.2], Rotation.from_euler('z', 30, degrees=True).as_matrix())
    camera0 = link0 @ transform([.1, 0., .05], Rotation.from_euler('xyz', [-100, 0, -90], degrees=True).as_matrix())
    offset = mount(link0, camera0)
    moved = Rotation.from_euler('zy', [45, 10], degrees=True).as_matrix()
    position, rotation = compose([0., 1., 1.], moved, offset)
    expected = transform([0., 1., 1.], moved) @ offset
    np.testing.assert_allclose(position, expected[:3, 3]) and np.testing.assert_allclose(rotation, expected[:3, :3])


def test_projection_and_backprojection_invert_each_other():
    rotation = Rotation.from_euler('xyz', [-120, 0, -90], degrees=True).as_matrix()   # looking forward and down
    position = np.array([-.9, -4.8, 1.3])
    points = np.array([[-.45, -4.85, .96], [-.5, -4.75, .9]])
    uv, depth = project(points, position, rotation, K)
    assert (depth > 0).all()
    np.testing.assert_allclose(backproject(uv, depth, position, rotation, K), points, atol=1e-9)


@pytest.mark.parametrize('yaw', [-40, 0, 25])
def test_the_two_box_points_give_back_the_box_pose(yaw):
    com = [-.45, -4.83, .85]
    top, edge = box_points(com, quat(yaw), .105)
    assert top[2] == pytest.approx(.955) and np.linalg.norm(top - edge) == pytest.approx(.105)
    estimate, q = box_pose(top, edge, .105)
    np.testing.assert_allclose(estimate, com, atol=1e-12)
    from kimodo_vla.oracle.grasp_policy import face_yaw
    assert face_yaw(q) == pytest.approx(face_yaw(quat(yaw)))


def test_depth_patch_ignores_holes_and_the_outside():
    depth = np.full((10, 10), 2.)
    depth[5, 5] = np.nan
    np.testing.assert_allclose(depth_at(depth, [[5, 5], [0, 0]]), [2., 2.])
    assert np.isnan(depth_at(depth, [[20, 3]])[0])


def test_extrapolation_moves_points_with_their_fitted_velocity():
    times = np.arange(6) * .04
    points = np.stack([[[0., .1 * t, 0.], [1., .1 * t, 0.]] for t in times])
    moved, velocity = extrapolate(times, points, times[-1] + .5, 6)
    np.testing.assert_allclose(velocity[:, 1], .1) and np.testing.assert_allclose(moved[:, 1], .1 * (times[-1] + .5))
    still, v = extrapolate(times[:1], points[:1], 3., 6)
    np.testing.assert_allclose(still, points[0]) and not v.any()


def test_tracker_flag_needs_the_anchored_target():
    a = parse_args(['--run-id', 'x', '--grasp-loop', '--stand-still', '--box-in-region', '--target-update', 'anchored',
                    '--perception', 'tracker'])
    assert a.perception == 'tracker'
    with pytest.raises(ValueError):
        parse_args(['--run-id', 'x', '--grasp-loop', '--stand-still', '--box-in-region', '--target-update', 'fixed',
                    '--perception', 'tracker'])


def test_tracker_labels_come_before_part_one_labels():
    close = lambda **k: {'status': 'tracked', 'background': False, 'lateral_error_m': .01, **k}
    assert an.tracker_label(None) is None and an.tracker_label(close()) is None
    assert an.tracker_label(close(status='lost')) == 'tracker_lost'
    assert an.tracker_label(close(background=True)) == 'tracker_background'
    assert an.tracker_label(close(lateral_error_m=-.04)) == 'tracker_drift'


def test_coverage_finds_missing_and_doubled_seeds():
    eps = [{'update': 'fixed', 'delay': 194, 'speed': .1, 'seed': s} for s in [*range(6000, 6029), 6003]]
    got = an.coverage(eps)['fixed 194 0.1']
    assert got == {'n': 30, 'missing': [6029], 'twice': [6003], 'outside': []}


def test_part_two_pairs_the_tracker_with_truth_on_the_same_seeds():
    ep = lambda u, s, ok: {'update': u, 'delay': 194, 'speed': .1, 'seed': s, 'success': ok, 'label': None if ok else 'other',
                           'tracker_close': {'status': 'tracked', 'background': False, 'lateral_error_m': .01}}
    eps = [ep('tracker', s, s < 6010) for s in range(6000, 6030)] + [ep('anchored', s, s < 6025) for s in range(6000, 6030)] \
        + [ep('fixed', s, s < 6005) for s in range(6000, 6030)]
    got = an.part_two(eps)['by_speed']['0.1 m/s']
    assert got['truth_minus_tracker_pp'] == 50.0 and got['verdict'] == 'perception is the new bottleneck'
    assert got['tracker_minus_fixed_pp'] == pytest.approx(16.7, abs=.1) and not got['core_finding_with_tracker']
