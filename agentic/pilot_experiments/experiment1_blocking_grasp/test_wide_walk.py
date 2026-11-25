"""The wide setting (2026-11-26): the official-way walk inside a grasp round, the wide box placements
and the run names. No weights, no simulator: the walker is a stand-in where it would call Kimodo."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from arena.box_layout import supported, wide
from kimodo_vla.core.types import G1_JOINT_NAMES, yaw_matrix
from kimodo_vla.model.features import G1_XML_PATH, KimodoFeatures
from kimodo_vla.oracle import grasp_walk
from kimodo_vla.oracle.geometry import SIDES, XmlFK
from kimodo_vla.oracle.grasp_policy import stance
from kimodo_vla.oracle.locomotion import load_goal
from run_grasp_suite import wide_prefix
from run_oracle import parse_args
from test_grasp_loop import BOX, CFG, CONFIG, run_until, scripted_policy, standing_sample

TABLE = [[CONFIG['scene']['table_edge_x_m'], -6.237], [.121, -3.763]]
START = [-1.58, -5.0]   # the task's start, where the robot is when a wide episode begins
WIDE = CONFIG['wide']


def test_wide_placements_follow_the_seed_only_and_stay_on_the_table():
    assert wide(8001, CONFIG) == wide(8001, CONFIG) and wide(8001, CONFIG) != wide(8002, CONFIG)
    seen = []
    for seed in WIDE['seeds']['debug'] + WIDE['seeds']['accept']:
        p = wide(seed, CONFIG)
        assert WIDE['depth_m'][0] <= p['depth_m'] <= WIDE['depth_m'][1] and abs(p['yaw_deg']) <= WIDE['max_yaw_deg']
        assert WIDE['box_y_m'][0] <= p['box_xy'][1] <= WIDE['box_y_m'][1] and p['robot_xy'] is None
        assert p['box_xy'][0] == pytest.approx(CONFIG['scene']['table_edge_x_m'] + p['depth_m'])
        assert p['direction'] in (-1, 1) and p['segment_y'] == WIDE['segment_y_m']
        assert all(supported([p['box_xy'][0], y], p['yaw_deg'], CONFIG['scene']['box_half_m'], TABLE)
                   for y in [p['box_xy'][1], *p['segment_y']])
        seen.append(p['box_xy'][1])
    assert max(seen) - min(seen) > 1.2   # the acceptance seeds really spread over the range


def test_any_yaw_keeps_a_box_at_either_end_of_the_wide_range_on_the_table():
    for y in WIDE['box_y_m'] + WIDE['segment_y_m']:
        for yaw in (-45, -20, 0, 30, 45):
            assert supported([CONFIG['scene']['table_edge_x_m'] + .16, y], yaw, CONFIG['scene']['box_half_m'], TABLE)


def test_wide_seeds_are_new():
    used = set(range(1000, 1012)) | set(range(3000, 3030)) | set(range(6000, 6030)) | set(range(7000, 7100))
    debug, accept = set(WIDE['seeds']['debug']), set(WIDE['seeds']['accept'])
    assert len(accept) == 30 and not debug & accept and not (debug | accept) & used


def test_wide_conditions_have_their_own_names():
    pairs = [(0., 75), (.03, 75), (.1, 75), (0., 194), (.03, 194), (.1, 194), (0., 383), (.03, 383), (.1, 383)]
    assert [wide_prefix(v, d, CONFIG) for v, d in pairs] == [
        'exp1-wide-static', 'exp1-wide-move003', 'exp1-wide-move010', 'exp1-wide-long-static', 'exp1-wide-long-move003',
        'exp1-wide-long-move010', 'exp1-wide-xlong-static', 'exp1-wide-xlong-move003', 'exp1-wide-xlong-move010']
    for v, d in [(.3, 75), (.1, 100)]:
        with pytest.raises(SystemExit):
            wide_prefix(v, d, CONFIG)


def test_wide_run_flags():
    a = parse_args(['--run-id', 'x', '--grasp-loop', '--box-wide', '--native-walk', '--box-speed', '.03'])
    assert a.box_wide and a.native_walk and a.box_speed == .03 and not a.stand_still
    plain = parse_args(['--run-id', 'x', '--grasp-loop'])
    assert not plain.box_wide and not plain.native_walk and plain.steps == 1450   # the default is unchanged
    for bad in (['--box-wide', '--stand-still', '--box-in-region'], ['--native-walk', '--start-at-stance'],
                ['--box-speed', '.1']):
        with pytest.raises(ValueError):
            parse_args(['--run-id', 'x', '--grasp-loop', *bad])
    with pytest.raises(ValueError):
        parse_args(['--run-id', 'x', '--box-wide'])


def test_walk_goal_is_seen_from_the_start_in_the_episode_heading():
    np.testing.assert_allclose(grasp_walk.walk_goal_xy([-1.58, -5., .78], [-.9, -4.2], np.eye(3)), [.68, .8])
    turned = yaw_matrix(np.radians(90.))     # facing +y: world +x is to the right
    np.testing.assert_allclose(grasp_walk.walk_goal_xy([0., 0., .78], [1., 0.], turned), [0., -1.], atol=1e-12)


def test_walk_settings_default_to_experiment_two_and_reject_unknown_keys():
    s = grasp_walk.settings(WIDE['walk'])
    assert (s['walk_steps'], s['segment_seconds'], s['transition_frames'], s['approach_gain'], s['tolerance_m']) == (
        50, 5., 5, 2., .05)
    assert grasp_walk.settings() == {k: v for k, v in s.items()}
    with pytest.raises(ValueError):
        grasp_walk.settings({'gain': 3})


def test_walker_frame_keeps_the_episode_heading_although_the_body_has_turned(tmp_path):
    walker = grasp_walk.NativeApproach.__new__(grasp_walk.NativeApproach)
    walker.features, walker.fk = KimodoFeatures(50.), XmlFK(G1_XML_PATH, G1_JOINT_NAMES)
    names = walker.features.representation.skeleton.bone_order_names
    walker.ids = [[names.index(f'{s}_{j}_skel') for j in ('wrist_yaw', 'hand_roll')] for s in SIDES]
    walker.model = SimpleNamespace(representation=walker.features.representation)
    walker.run, walker.mode, walker.steps, walker.noise_seed, walker.walk_steps, walker.arm_ik = tmp_path, 'hold', 2, 5, 50, False
    walker.native_settings = grasp_walk.settings()
    walker.goal = load_goal({'name': 'x', 'root': {'xy': [.1, 0.]}})
    walker.reset(5)
    rotation = Rotation.from_euler('z', .3).as_matrix()      # the body drifted 17 degrees left
    root, joints = np.array([-1.5, -5., .78]), np.zeros(29)
    wrist, wrist_rot = walker.fk(joints, root, rotation)
    quat = np.roll(Rotation.from_matrix(rotation).as_quat(), 1).tolist()
    sample = {'time': 3., 'root_pos': root.tolist(), 'root_quat': quat, 'joints': joints.tolist(), 'wrist_pos': wrist.tolist(),
              'wrist_rot': wrist_rot.tolist(), 'reference_root_pos': root.tolist(), 'reference_root_quat': quat,
              'reference_joints': joints.tolist()}
    goal = [-.9, -4.6]
    xy = grasp_walk.start(walker, run=tmp_path/'walk-r0', goal_world=goal, heading=np.eye(3),
                          history=[dict(sample, time=3. - .02 * k) for k in range(40, 0, -1)] + [sample], seed=77)
    assert len(walker.primed) == 30 and walker.seed == 77 and (tmp_path/'walk-r0').is_dir()
    walker.initialize(walker.primed[0])
    np.testing.assert_allclose(walker.frame.heading, np.eye(3), atol=1e-12)   # not the body's own heading
    world = walker.frame.world([walker.root_end[0], 0., walker.root_end[1]])[:2]
    np.testing.assert_allclose(world, goal, atol=1e-9)
    np.testing.assert_allclose(xy, np.subtract(goal, root[:2]))


class FakeWalker:
    """The walker as the grasp sees it: plans until it has sent the stop plan."""
    def __init__(self, plans_before_stop=3):
        self.native_settings = grasp_walk.settings()
        self.plans_before_stop, self.calls = plans_before_stop, []

    def reset(self, seed=None):
        self.seed, self.stop_sent, self.plan_index, self.phases = seed, False, 0, []

    def __call__(self, request):
        samples = request['observation']['oracle_history']
        self.calls.append(len(samples))
        self.plan_index += 1
        if self.plan_index > self.plans_before_stop:
            self.stop_sent = True
            self.phases = [{'phase': 'walk'}, {'phase': 'stop', 'time': samples[-1]['time'], 'walk_timed_out': False,
                                                'root_remaining_m': .03}]
        return {'action_chunk': np.zeros((80 if self.stop_sent else 50, 40)).tolist()}


def native_policy(tmp_path):
    policy = scripted_policy(tmp_path, delay=5)
    policy.native_walk, policy.walker = dict(WIDE), FakeWalker()
    policy.history = []
    return policy


def test_native_goal_is_the_stance_moved_back_and_aside():
    policy = scripted_policy(Path('.'), delay=5)
    policy.native_walk = dict(WIDE)
    policy.episode_heading = np.eye(3)
    goal = policy.native_goal(BOX)
    np.testing.assert_allclose(goal, stance(BOX, CFG) + [-WIDE['stance_extra_m'], -WIDE['box_lateral_m']])
    assert goal[1] - BOX[1] == pytest.approx(.05)          # the box ends 5 cm to the robot's right


def test_a_native_round_walks_through_the_walker_then_settles_and_reaches(tmp_path):
    policy = native_policy(tmp_path)
    far = standing_sample(policy, 0., BOX, root_xy=START)
    policy.begin_episode(far)
    for step in range(1, 7):
        policy.advance([dict(far, time=step * .02)])
        policy.history.append(dict(far, time=step * .02))
    assert policy.round.phase == 'walk' and policy.walk['native'] and not policy.walk['primed']
    assert policy.round.notes['walk_generator'] == 'native'
    np.testing.assert_allclose(policy.round.notes['walk_goal'], policy.native_goal(BOX))
    request = {'task': 'pick', 'observation': {'oracle_history': [dict(far, time=.14)]}}
    policy.remember(request['observation']['oracle_history'])
    actions = policy.native_step(request)
    assert actions.shape == (50, 40) and policy.walk['primed'] and policy.walker.calls == [7]   # primed with history
    assert policy.walker.seed == policy.seed * 1000 and (tmp_path/'walk-r0').is_dir()
    time = .14
    while not policy.walker.stop_sent:
        time += 1.
        policy.advance([dict(far, time=time)])
        assert policy.round.phase == 'walk'
        policy.native_step({'task': 'pick', 'observation': {'oracle_history': [dict(far, time=time)]}})
    assert policy.walker.calls[1:] == [1, 1, 1]                 # later plans get only the new samples
    there = policy.round.notes['walk_goal']
    policy.advance([standing_sample(policy, time + 1.6, BOX, root_xy=there)])
    n = policy.round.notes
    assert policy.round.phase == 'settle' and n['walk_plans'] == 4 and not n['walk_timed_out']
    assert n['walk_stop_remaining_m'] == .03
    run_until(policy, time + 1.6, 'raise', BOX)                 # settles (here by its timeout), then reaches
    assert policy.round.notes['stance_error_m'] == pytest.approx(np.hypot(.05, .05))


def test_a_native_round_already_at_its_stance_does_not_walk(tmp_path):
    policy = native_policy(tmp_path)
    policy.episode_heading = np.eye(3)
    here = policy.native_goal(BOX) + [.02, .02]
    policy.episode_heading = None
    policy.begin_episode(standing_sample(policy, 0., BOX, root_xy=here))
    for step in range(1, 7):
        policy.advance([standing_sample(policy, step * .02, BOX, root_xy=here)])
    assert policy.round.phase == 'raise' and policy.round.notes['walk_skipped'] and policy.walker.calls == []


# ---- the analysis rules written into the record before the data ------------------------------

def test_first_round_causes_follow_the_written_order():
    import analyze_wide as aw
    base = dict(label='no_contact', gate_fired=False, clearance_m=.21, lateral_m=-.05, backoff_m=.03)
    assert aw.first_round_cause(dict(base, label='success')) == 'success'
    assert aw.first_round_cause(dict(base, gate_fired=True, clearance_m=.40)) == 'after_close'
    assert aw.first_round_cause(dict(base, clearance_m=.15, backoff_m=.3)) == 'landing'     # landing comes first
    assert aw.first_round_cause(dict(base, lateral_m=.02)) == 'landing'
    assert aw.first_round_cause(dict(base, backoff_m=.2)) == 'backoff'
    assert aw.first_round_cause(base) == 'other'
    assert aw.landing_ok(dict(base, clearance_m=.27, lateral_m=-.105)) and not aw.landing_ok(dict(base, clearance_m=None))


def test_out_of_reach_needs_a_walk_a_failure_and_the_box_outside_the_span():
    import analyze_wide as aw
    r = dict(label='no_contact', walked=True, box_rel_lateral_m=-.05, box_depth_m=.16)
    assert not aw.out_of_reach(r)
    assert aw.out_of_reach(dict(r, box_rel_lateral_m=-.30)) and aw.out_of_reach(dict(r, box_depth_m=.30))
    assert not aw.out_of_reach(dict(r, box_rel_lateral_m=-.30, label='success'))
    assert not aw.out_of_reach(dict(r, box_rel_lateral_m=-.30, walked=False))


def test_displacement_bins_and_logistic_fit():
    import analyze_wide as aw
    assert aw.bins([0., .029, .03, .07, .2, .9]).tolist() == [0, 0, 1, 2, 4, 6]
    x = np.r_[np.zeros(20), np.full(20, 10.)]
    y = np.r_[np.ones(18), np.zeros(2), np.ones(4), np.zeros(16)]
    w, ll = aw.logistic(x, y)
    assert w[1] < 0 and 1 / (1 + np.exp(-w[0])) == pytest.approx(.9, abs=1e-6) and ll < 0
