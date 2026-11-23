import json

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from analyze_motion import gait
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.model.features import KimodoFeatures, G1_XML_PATH
from kimodo_vla.oracle.geometry import SIDES, XmlFK
from kimodo_vla.oracle.locomotion import (CANVAS, STAND_FROM, LocomotionPolicy, approach_rotation, ease,
                                          episode_seconds, ground_path, load_goal, walk_allowance)
from rollout_kimodo import follow
from run_oracle import parse_args

WALK = {'name': 'forward', 'root': {'xy': [.5, 0.]}}
REACH = {'name': 'reach', 'root': {'xy': [.4, -.1]},
         'hand': {'side': 'right', 'position': [.75, -.2, .95], 'approach': [1, 0, -.3], 'duration_s': 1.5}}
NEAR = {'name': 'near', 'hand': {'side': 'right', 'position': [.3, -.25, .9], 'approach': [1, 0, 0], 'duration_s': 1.5}}


def test_goal_file_is_validated_and_bounded_in_time():
    assert load_goal(WALK)['root'] == {'xy': [.5, 0.], 'speed_m_s': .35, 'tolerance_m': .05}
    assert walk_allowance(load_goal(WALK)) == pytest.approx(.5/.05+5)
    assert episode_seconds(load_goal(NEAR)) == pytest.approx(.5+1.5+2)
    assert episode_seconds(load_goal(REACH)) > episode_seconds(load_goal(WALK)) > 10
    for bad in ({'name': 'x'}, {'root': {'xy': [1, 0]}}, {'name': 'x', 'root': {'xy': [1, 0], 'speed_m_s': 3}},
                {'name': 'x', 'root': {'xy': [1, 0], 'duration_s': 1}},
                {'name': 'x', 'hand': {'side': 'both', 'position': [0, 0, 1], 'approach': [1, 0, 0], 'duration_s': 1}},
                {'name': 'x', 'hand': {'side': 'right', 'position': [0, 0, 1], 'approach': [0, 1, 0], 'duration_s': 1}}):
        with pytest.raises((ValueError, KeyError)):
            load_goal(bad)


def test_ease_starts_and_ends_at_rest():
    u = np.linspace(0, 1, 1001)
    w = ease(u)
    assert w[0] == 0 and w[-1] == 1 and np.all(np.diff(w) >= 0)
    assert abs(np.gradient(w, u)[[0, -1]]).max() < 1e-3


def test_ground_path_speeds_up_brakes_and_never_passes_the_goal():
    path = ground_path([0., 0.], [0., .3], 0., .35)
    along = path[:, 1]
    speed = np.diff(along)*30
    assert path.shape == (CANVAS, 2) and np.all(path[:, 0] == 0) and along[0] == 0
    assert np.all(np.diff(along) >= 0) and along[-1] == pytest.approx(.3) and speed.max() <= .35+1e-9
    assert speed[0] == pytest.approx(1/30) and speed[-1] == 0
    far = ground_path([1., 1.], [1., 4.], .35, .35)
    np.testing.assert_allclose(np.diff(far[:, 1])*30, .35)        # already at cruise speed: keeps it
    np.testing.assert_allclose(ground_path([2., 2.], [2., 2.], .2, .35), np.tile([2., 2.], (CANVAS, 1)))


@pytest.mark.parametrize('approach', [[1, 0, 0], [1, 0, -1], [.5, .2, .6], [0, 0, -1]])
def test_approach_rotation_points_the_forearm_and_keeps_the_palm_inward(approach):
    r = approach_rotation(approach)
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) > 0
    np.testing.assert_allclose(r[:, 0], np.asarray(approach)/np.linalg.norm(approach))
    assert r[1, 1] > 0 and abs(r[:, 1] @ r[:, 0]) < 1e-12
    if approach == [1, 0, 0]:
        np.testing.assert_allclose(r, np.eye(3), atol=1e-12)


def test_moving_goal_switches_off_standing_support_and_ik():
    a = parse_args(['--run-id', 'x', '--motion-goal', 'goal.json', '--start-xy', '-4', '-4'])
    assert (a.support_constraints, a.hand_constraints, a.trajectory_projection) == ('none', 'keyframes', 'none')
    assert (a.variant, a.dynamics_profile, a.walk_steps, a.start_xy) == ('bounded', 'actuator', 75, [-4., -4.])
    plain = parse_args(['--run-id', 'x'])
    assert plain.motion_goal is None and plain.start_xy is None and plain.noise_seed is None


class FollowPath:
    """Stands in for Kimodo: the starting pose slid along the conditioned root path."""
    def __init__(self, features, pose):
        self.representation, self.device, self.pose, self.calls = features.representation, 'cpu', pose, []

    def predict_future(self, task, image, history, *, history_feature_mask, text_feat, future_condition,
                       future_condition_mask, diffusion_steps, generator):
        self.calls.append(dict(history=history, history_mask=history_feature_mask, values=future_condition,
                               mask=future_condition_mask))
        start = self.representation.slice_dict['smooth_root_pos'].start
        root = self.pose['root_positions'].repeat(CANVAS, 1)
        root[1:, 0], root[1:, 2] = future_condition[1:, start], future_condition[1:, start+2]
        root[0, [0, 2]] = future_condition[0, [start, start+2]]
        local = self.pose['local_rot_mats'].repeat(CANVAS, 1, 1, 1)
        return {'local_rot_mats': local, 'root_positions': root, 'history_last_root_position': root[0].clone(),
                'motion_features': self.representation(local, root, to_normalize=False)[0]}


def make_policy(tmp_path, goal, yaw=.4):
    policy = LocomotionPolicy.__new__(LocomotionPolicy)
    policy.goal, policy.noise_seed, policy.walk_steps = load_goal(goal), 1003, 75
    policy.features = KimodoFeatures(50.)
    policy.fk = XmlFK(G1_XML_PATH, G1_JOINT_NAMES)
    names = policy.features.representation.skeleton.bone_order_names
    policy.ids = [[names.index(f'{s}_{j}_skel') for j in ('wrist_yaw', 'hand_roll')] for s in SIDES]
    policy.run, policy.mode, policy.steps = tmp_path, 'hold', 2
    policy.variant, policy.handoff_seconds, policy.joint_speed = 'bounded', .2, 4.
    policy.constraint_cfg = 2.
    policy.reset(1000)
    joints = np.zeros(29)
    joints[[G1_JOINT_NAMES.index(f'{s}_shoulder_pitch_joint') for s in SIDES]] = .2
    rotation = Rotation.from_euler('z', yaw).as_matrix()
    quat = np.roll(Rotation.from_matrix(rotation).as_quat(), 1).tolist()
    root = np.array([-4., -4., .78])
    wrist, wrist_rot = policy.fk(joints, root, rotation)
    start = {'time': 10., 'root_pos': root.tolist(), 'root_quat': quat, 'joints': joints.tolist(),
             'wrist_pos': wrist.tolist(), 'wrist_rot': wrist_rot.tolist(),
             'reference_root_pos': root.tolist(), 'reference_root_quat': quat, 'reference_joints': joints.tolist()}
    pose = policy.features.codec.decode_joint_configuration_pose(joints[None], np.array([[0., 0., .78]]),
        root_rotation_matrices=np.eye(3)[None], joint_names=G1_JOINT_NAMES)
    policy.model = FollowPath(policy.features, pose)
    return policy, start, rotation, root


def test_perfect_tracker_walks_to_the_goal_then_stands_and_holds(tmp_path):
    policy, start, rotation, root = make_policy(tmp_path, WALK)
    rows = follow(policy, start, 8.)
    goal = (root+rotation @ [.5, 0, 0])[:2]
    np.testing.assert_allclose(rows[-1]['body_xy'], goal, atol=policy.goal['root']['tolerance_m'])
    assert [p['phase'] for p in policy.phases] == ['walk', 'stop', 'hold']
    assert policy.phases[0]['elapsed_s'] == pytest.approx(.6) and not policy.phases[1]['walk_timed_out']
    target = json.loads((tmp_path/'target.json').read_text())
    assert target['arrival_time'] == pytest.approx(policy.phases[2]['time']) and target['active_sides'] == []
    assert target['noise_seed'] == 1003
    np.testing.assert_allclose(target['root_goal_world'], goal, atol=1e-6)
    plans = [json.loads(x) for x in (tmp_path/'plans.jsonl').read_text().splitlines()]
    assert [p['phase'] for p in plans][-1] == 'stop' and all(p['sent_steps'] == 75 for p in plans)
    assert all(p['anchor_shift_m'] < 1e-5 and p['root_path_error_m'] < 1e-4 for p in plans)
    rep = policy.model.representation
    walk, stop = policy.model.calls[0], policy.model.calls[-1]
    path = torch.zeros(rep.motion_rep_dim, dtype=torch.bool)
    path[[policy.root_slot, policy.root_slot+2]] = True
    path[policy.heading_slice] = True
    assert walk['mask'][1:, path].all() and not walk['mask'][1:, ~path].any()   # feet and hands are free
    assert walk['mask'][0, :policy.pose_end].all() and walk['history_mask'].all()
    assert stop['mask'][STAND_FROM:].all() and not stop['mask'][1:STAND_FROM, ~path].any()
    saved = np.load(tmp_path/'plan-000.npz')
    assert saved['projected_wrist_world'].shape == (CANVAS, 2, 3) and saved['raw_future_root_world'].shape == (CANVAS, 3)
    # The stand-in slides without stepping, and the gait measure says so.
    walking = [r for r in rows if r['phase'] == 'walk']
    assert gait([r['sent_feet'] for r in walking], [r['sent_root_world'] for r in walking])['both_moving_share'] > .6


def test_a_tracker_that_covers_less_ground_still_arrives_because_plans_start_from_the_body(tmp_path):
    policy, start, rotation, root = make_policy(tmp_path, WALK)
    rows = follow(policy, start, 14., carry=.6)
    goal = (root+rotation @ [.5, 0, 0])[:2]
    assert np.linalg.norm(np.array(rows[-1]['body_xy'])-goal) <= policy.goal['root']['tolerance_m']+1e-6
    assert [p['phase'] for p in policy.phases] == ['walk', 'stop', 'hold'] and not policy.phases[1]['walk_timed_out']
    plans = [json.loads(x) for x in (tmp_path/'plans.jsonl').read_text().splitlines()]
    assert len(plans) > 3 and max(p['anchor_shift_m'] for p in plans) > .05      # the reference was pulled back
    # Each walking plan's history is continuous although the logged reference jumped at every plan.
    for call in policy.model.calls[1:]:
        ground = call['history'][:, [policy.root_slot, policy.root_slot+2]].numpy()
        assert np.abs(np.diff(ground, axis=0)).max() < .03


def test_a_walk_that_cannot_get_there_is_stopped_at_its_allowance(tmp_path):
    policy, start, _, _ = make_policy(tmp_path, WALK)
    follow(policy, start, 19., carry=0.)
    stop = next(p for p in policy.phases if p['phase'] == 'stop')
    assert stop['walk_timed_out'] and stop['root_remaining_m'] == pytest.approx(.5)
    assert stop['elapsed_s'] >= .6+walk_allowance(policy.goal)


def test_reach_holds_the_stance_and_conditions_one_wrist_on_a_world_path(tmp_path):
    policy, start, rotation, root = make_policy(tmp_path, REACH)
    rows = follow(policy, start, 9.)
    assert [p['phase'] for p in policy.phases] == ['walk', 'stop', 'reach']
    reach = policy.phases[-1]
    target = json.loads((tmp_path/'target.json').read_text())
    assert target['arrival_time'] == pytest.approx(reach['time']+1.5) and target['active_sides'] == ['right']
    goal = np.array([root[0], root[1], 0.])+rotation @ np.array(REACH['hand']['position'])
    np.testing.assert_allclose(np.array(target['targets_world'])[1, 0], goal, atol=1e-9)
    np.testing.assert_allclose(np.array(target['wrist_rotation_world'])[1][:, 0],
                               rotation @ np.array([1, 0, -.3])/np.linalg.norm([1, 0, -.3]), atol=1e-9)
    plans = [json.loads(x) for x in (tmp_path/'plans.jsonl').read_text().splitlines()]
    first = next(i for i, p in enumerate(plans) if p['phase'] == 'reach')
    call, record = policy.model.calls[first], plans[first]
    rep = policy.model.representation
    names = rep.skeleton.bone_order_names
    positions = rep.slice_dict['local_joints_positions'].start
    rotations = rep.slice_dict['global_rot_data'].start
    wrist, left = names.index('right_wrist_yaw_skel'), names.index('left_wrist_yaw_skel')
    assert record['sent_steps'] == 25 and record['dense_hand_frames'] == list(range(1, 16))
    arrival = record['keyframes'][0]
    assert 43 <= arrival <= 45
    for k in record['dense_hand_frames']+record['keyframes']:
        assert call['mask'][k, positions+3*wrist:positions+3*wrist+3].all()
        assert call['mask'][k, rotations+6*wrist:rotations+6*wrist+6].all()
        assert not call['mask'][k, positions+3*left:positions+3*left+3].any()
    assert not call['mask'][20, positions+3*wrist:positions+3*wrist+3].any()
    assert call['mask'][1:, policy.feet_channels].all() and policy.feet_channels.sum() > 40
    # The root's ground point is held where the body stood when the reach began.
    stance = call['values'][1:, [policy.root_slot, policy.root_slot+2]].numpy()
    np.testing.assert_allclose(stance, np.tile(policy.stance_xz, (CANVAS-1, 1)), atol=1e-6)
    # At the arrival keyframe the conditioned wrist, put back on that ground point, is the world target.
    local = call['values'][arrival, positions+3*wrist:positions+3*wrist+3].numpy().astype(float)
    local[[0, 2]] += stance[arrival-1]
    np.testing.assert_allclose(policy.frame.world(local), goal, atol=1e-5)
    np.testing.assert_allclose(target['hand_start']['wrist_world'],
                               next(r for r in rows if r['time'] > reach['time'])['sent_wrist'][1], atol=.02)
    assert plans[-1]['keyframes'] == [15, 30, 49] and plans[-1]['anchor_shift_m'] == 0


def test_a_goal_without_a_walk_reaches_from_the_start_stance(tmp_path):
    policy, start, _, _ = make_policy(tmp_path, NEAR)
    follow(policy, start, 3.)
    assert [p['phase'] for p in policy.phases] == ['reach'] and policy.phases[0]['elapsed_s'] == pytest.approx(.6)


def test_gait_tells_stepping_from_sliding():
    t = np.arange(200)/50
    slide = np.zeros((200, 2, 3))
    slide[:, :, 0] = (.25*t)[:, None]
    slide[:, :, 2] = .035
    root = np.zeros((200, 3))
    root[:, 0] = .25*t
    sliding = gait(slide, root)
    assert sliding['both_moving_share'] == pytest.approx(1.) and sliding['swings'] == 0
    assert sliding['root_travel_m'] == pytest.approx(.995) and sliding['foot_travel_m'] == pytest.approx(1.99)
    step = np.zeros((200, 2, 3))
    step[:, :, 2] = .035
    for foot, begin in ((0, 0), (1, 50), (0, 100), (1, 150)):  # each foot moves only while lifted
        phase = np.linspace(0, 1, 50)
        step[begin:begin+50, foot, 2] += .08*np.sin(np.pi*phase)
        step[begin:, foot, 0] += .25
        step[begin:begin+50, foot, 0] += .25*(phase-1)
    stepping = gait(step, root)
    assert stepping['swings'] == 4 and stepping['both_moving_share'] == 0
    assert stepping['max_lift_cm'] == pytest.approx(8., abs=.1) and stepping['foot_travel_m'] == pytest.approx(1., abs=.01)


def test_leaned_pose_pitches_the_trunk_and_keeps_the_feet(tmp_path):
    from kimodo_vla.oracle.locomotion import LEAN_FROM, leaned_pose
    from analyze_motion import pitch_deg
    fk = XmlFK(G1_XML_PATH, G1_JOINT_NAMES)
    feet = [f'{s}_ankle_roll_link' for s in SIDES]
    rotation = Rotation.from_euler('z', .4).as_matrix()
    root, joints = np.array([-4., -4., .78]), np.zeros(29)
    leaned, moved_root, pitched = leaned_pose(fk, joints, root, rotation, 30., feet)
    before = fk(joints, root, rotation, body_names=feet+['torso_link'])
    after = fk(leaned, moved_root, pitched, body_names=feet+['torso_link'])
    np.testing.assert_allclose(after[0][:2], before[0][:2], atol=1e-9)                 # ankles did not move
    np.testing.assert_allclose(after[1][:2], before[1][:2], atol=1e-9)                 # nor turn
    assert pitch_deg(after[1][2], rotation)-pitch_deg(before[1][2], rotation) == pytest.approx(30., abs=1e-6)
    assert leaned[G1_JOINT_NAMES.index('waist_pitch_joint')] <= fk.joint_limits[G1_JOINT_NAMES.index('waist_pitch_joint'), 1]
    policy, start, _, _ = make_policy(tmp_path, {'name': 'lean-30', 'lean': {'pitch_deg': 30}})
    rows = follow(policy, start, 4.)
    assert [p['phase'] for p in policy.phases] == ['lean', 'hold']
    call = policy.model.calls[0]
    assert len(policy.model.calls) == 1 and call['mask'][LEAN_FROM:].all() and call['mask'][1:, policy.feet_channels].all()
    assert not call['mask'][1:LEAN_FROM, ~(policy.feet_channels | call['mask'][1])].any()
    plans = [json.loads(x) for x in (tmp_path/'plans.jsonl').read_text().splitlines()]
    assert plans[0]['sent_steps'] == 80 and plans[0]['pose_frames'][0] == LEAN_FROM and len(rows) == 80
    target = json.loads((tmp_path/'target.json').read_text())
    assert target['arrival_time'] == pytest.approx(policy.phases[1]['time']) and 'lean_pose' in target
    with pytest.raises(ValueError):
        load_goal({'name': 'x', 'lean': {'pitch_deg': 30}, 'root': {'xy': [1, 0]}})
