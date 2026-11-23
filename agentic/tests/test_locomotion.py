import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.model.features import KimodoFeatures, G1_XML_PATH
from kimodo_vla.oracle.geometry import BASIS, SIDES, XmlFK
from kimodo_vla.oracle.locomotion import (CANVAS, LocomotionPolicy, approach_rotation, ease, end_time, load_goal,
                                          root_progress)
from run_oracle import parse_args

WALK = {'name': 'forward', 'root': {'xy': [.5, 0.], 'start_s': .5, 'duration_s': 2.5}}
REACH = {'name': 'reach', 'root': {'xy': [.4, -.1], 'start_s': .5, 'duration_s': 2.},
         'hand': {'side': 'right', 'position': [.75, -.2, .95], 'approach': [1, 0, -.3], 'start_s': 3., 'duration_s': 1.5}}


def test_goal_file_is_validated_and_timed():
    assert load_goal(WALK)['root'] == {'xy': [.5, 0.], 'start_s': .5, 'duration_s': 2.5}
    assert end_time(load_goal(REACH)) == 4.5
    np.testing.assert_allclose(root_progress(load_goal(WALK), [0., .5, 1.75, 3., 9.]), [0, 0, .5, 1, 1])
    for bad in ({'name': 'x'}, {'root': {'xy': [1, 0], 'duration_s': 1}}, {'name': 'x', 'root': {'xy': [1, 0]}},
                {'name': 'x', 'root': {'xy': [1, 0], 'duration_s': 1, 'yaw': 3}},
                {'name': 'x', 'hand': {'side': 'both', 'position': [0, 0, 1], 'approach': [1, 0, 0], 'start_s': 0, 'duration_s': 1}},
                {'name': 'x', 'hand': {'side': 'right', 'position': [0, 0, 1], 'approach': [0, 1, 0], 'start_s': 0, 'duration_s': 1}}):
        with pytest.raises((ValueError, KeyError)):
            load_goal(bad)


def test_ease_starts_and_ends_at_rest():
    u = np.linspace(0, 1, 1001)
    w = ease(u)
    assert w[0] == 0 and w[-1] == 1 and np.all(np.diff(w) >= 0)
    assert abs(np.gradient(w, u)[[0, -1]]).max() < 1e-3


@pytest.mark.parametrize('approach', [[1, 0, 0], [1, 0, -1], [.5, .2, .6], [0, 0, -1]])
def test_approach_rotation_points_the_forearm_and_keeps_the_palm_inward(approach):
    r = approach_rotation(approach, 'right')
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) > 0
    np.testing.assert_allclose(r[:, 0], np.asarray(approach)/np.linalg.norm(approach))
    assert r[1, 1] > 0 and abs(r[:, 1] @ r[:, 0]) < 1e-12
    if approach == [1, 0, 0]:
        np.testing.assert_allclose(r, np.eye(3), atol=1e-12)


def test_moving_goal_switches_off_standing_support_and_ik():
    a = parse_args(['--run-id', 'x', '--motion-goal', 'goal.json', '--start-xy', '-4', '-4'])
    assert (a.support_constraints, a.hand_constraints, a.trajectory_projection) == ('none', 'keyframes', 'none')
    assert (a.variant, a.dynamics_profile, a.history_source, a.start_xy) == ('bounded', 'actuator', 'reference', [-4., -4.])
    plain = parse_args(['--run-id', 'x'])
    assert plain.motion_goal is None and plain.start_xy is None


class FollowPath:
    """Stands in for Kimodo: the starting pose slid along the constrained root path."""
    def __init__(self, features, pose):
        self.representation, self.device, self.pose, self.calls = features.representation, 'cpu', pose, []

    def predict_future(self, task, image, history, *, history_feature_mask, text_feat, future_condition,
                       future_condition_mask, diffusion_steps, generator):
        self.calls.append(dict(history=history, history_mask=history_feature_mask, values=future_condition,
                               mask=future_condition_mask))
        start = self.representation.slice_dict['smooth_root_pos'].start
        root = self.pose['root_positions'].repeat(CANVAS, 1)
        root[1:, 0], root[1:, 2] = future_condition[1:, start], future_condition[1:, start+2]
        root[0, [0, 2]] = history[-1, [start, start+2]]
        local = self.pose['local_rot_mats'].repeat(CANVAS, 1, 1, 1)
        return {'local_rot_mats': local, 'root_positions': root, 'history_last_root_position': root[0].clone(),
                'motion_features': self.representation(local, root, to_normalize=False)[0]}


def make_policy(tmp_path, goal, source='reference', yaw=.4):
    policy = LocomotionPolicy.__new__(LocomotionPolicy)
    policy.goal, policy.history_source, policy.noise_seed = load_goal(goal), source, 1003
    policy.sent, policy.root_condition = 25, 'path'
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
    root = np.array([-4., -4., .78])
    wrist, wrist_rot = policy.fk(joints, root, rotation)

    def sample(k):
        return {'time': 10.+k/50, 'root_pos': root.tolist(), 'root_quat': np.roll(Rotation.from_matrix(rotation).as_quat(), 1).tolist(),
                'joints': joints.tolist(), 'wrist_pos': wrist.tolist(), 'wrist_rot': wrist_rot.tolist(),
                'reference_root_pos': root.tolist(), 'reference_root_quat': np.roll(Rotation.from_matrix(rotation).as_quat(), 1).tolist(),
                'reference_joints': joints.tolist()}
    pose = policy.features.codec.decode_joint_configuration_pose(joints[None], np.array([[0., 0., .78]]),
        root_rotation_matrices=Rotation.from_euler('z', 0.).as_matrix()[None], joint_names=G1_JOINT_NAMES)
    policy.model = FollowPath(policy.features, pose)
    return policy, sample, rotation, root


def test_root_path_is_dense_in_the_start_frame_and_feet_are_free(tmp_path):
    policy, sample, rotation, root = make_policy(tmp_path, WALK)
    first = policy({'task': 'x', 'observation': {'oracle_history': [sample(k) for k in range(6)]}})
    call = policy.model.calls[0]
    rep = policy.model.representation
    start = rep.slice_dict['smooth_root_pos'].start
    heading = rep.slice_dict['global_root_heading']
    allowed = torch.zeros(rep.motion_rep_dim, dtype=torch.bool)
    allowed[[start, start+2]] = True
    allowed[heading] = True
    assert call['mask'][1:, allowed].all() and not call['mask'][1:, ~allowed].any()
    assert call['mask'][0, :rep.slice_dict['global_rot_data'].stop].all()
    assert call['history_mask'].all()
    # Forward in the start frame is model +z whatever the world heading; the path ends 0.5 m ahead.
    np.testing.assert_allclose(policy.root_xz(9.), [0., .5], atol=1e-6)
    np.testing.assert_allclose(policy.target_info['root_goal_world'], (root+rotation @ [.5, 0, 0])[:2], atol=1e-6)
    assert len(first['action_chunk']) == 25 and first['oracle_target']['mode'] == 'motion-goal'
    target = json.loads((tmp_path/'target.json').read_text())
    assert target['active_sides'] == [] and target['arrival_time'] == pytest.approx(13.)
    assert target['noise_seed'] == policy.seed == 1003


def test_sent_reference_walks_the_path_and_plans_chain_on_the_reference(tmp_path):
    policy, sample, rotation, root = make_policy(tmp_path, WALK)
    history = [sample(k) for k in range(6)]
    reference = np.array(root)
    out = None
    for plan in range(8):
        out = policy({'task': 'x', 'observation': {'oracle_history': history}})
        history = []
        for action in np.asarray(out['action_chunk']):
            reference = reference + rotation @ np.r_[action[:2], 0.]
            k = round((policy.history[-1]['time']-10.)*50)+len(history)+1
            history.append({**sample(k), 'reference_root_pos': [reference[0], reference[1], action[2]],
                            'reference_joints': action[9:38].tolist()})
    # The body is never told to move here (measured root stays put), yet the reference reaches the goal.
    np.testing.assert_allclose(reference[:2], (root+rotation @ [.5, 0, 0])[:2], atol=2e-3)
    record = out['oracle_plan']
    assert record['root_path_progress'] == pytest.approx(1.) and record['root_path_error_m'] < 1e-3
    assert record['measured_root_path_m'] == pytest.approx(.5, abs=1e-3)
    plans = [json.loads(x) for x in (tmp_path/'plans.jsonl').read_text().splitlines()]
    assert [p['plan'] for p in plans] == list(range(8)) and all(p['history_source'] == 'reference' for p in plans)
    saved = np.load(tmp_path/'plan-007.npz')
    assert saved['projected_wrist_world'].shape == (CANVAS, 2, 3) and saved['raw_future_root_world'].shape == (CANVAS, 3)
    assert len((tmp_path/'commands.jsonl').read_text().splitlines()) == 8*25


def test_measured_last_history_keeps_one_frame(tmp_path):
    policy, sample, _, _ = make_policy(tmp_path, WALK, source='measured-last')
    policy({'task': 'x', 'observation': {'oracle_history': [sample(k) for k in range(30)]}})
    mask = policy.model.calls[0]['history_mask']
    assert mask[-1].all() and not mask[:-1].any()


def test_one_hand_is_constrained_on_a_world_path_from_where_it_was(tmp_path):
    policy, sample, rotation, root = make_policy(tmp_path, REACH)
    policy({'task': 'x', 'observation': {'oracle_history': [sample(k) for k in range(6)]}})
    rep = policy.model.representation
    assert not policy.model.calls[0]['mask'][1:, rep.slice_dict['local_joints_positions']].any()
    policy({'task': 'x', 'observation': {'oracle_history': [sample(k) for k in range(6, 6+150)]}})  # now at 3.1 s
    call = policy.model.calls[1]
    names = rep.skeleton.bone_order_names
    positions = rep.slice_dict['local_joints_positions'].start
    rotations = rep.slice_dict['global_rot_data'].start
    wrist, left = names.index('right_wrist_yaw_skel'), names.index('left_wrist_yaw_skel')
    record = json.loads((tmp_path/'plans.jsonl').read_text().splitlines()[1])
    assert record['dense_hand_frames'] == list(range(1, 16)) and record['keyframes'] == [42]
    for k in record['dense_hand_frames']+record['keyframes']:
        assert call['mask'][k, positions+3*wrist:positions+3*wrist+3].all()
        assert call['mask'][k, rotations+6*wrist:rotations+6*wrist+6].all()
        assert not call['mask'][k, positions+3*left:positions+3*left+3].any()
    assert not call['mask'][20, positions+3*wrist:positions+3*wrist+3].any()
    # At the arrival keyframe the constrained wrist, put back on the root path, is the world target.
    k = 42
    local = call['values'][k, positions+3*wrist:positions+3*wrist+3].numpy().astype(float)
    start = rep.slice_dict['smooth_root_pos'].start
    local[[0, 2]] += call['values'][k, [start, start+2]].numpy()
    goal = np.array([root[0], root[1], 0.])+rotation @ np.array(REACH['hand']['position'])
    np.testing.assert_allclose(policy.frame.world(local), goal, atol=1e-5)
    target = json.loads((tmp_path/'target.json').read_text())
    np.testing.assert_allclose(np.array(target['targets_world'])[1, 0], goal, atol=1e-9)
    np.testing.assert_allclose(np.array(target['wrist_rotation_world'])[1][:, 0],
                               rotation @ np.array([1, 0, -.3])/np.linalg.norm([1, 0, -.3]), atol=1e-9)
    np.testing.assert_allclose(target['hand_start']['wrist_world'], sample(0)['wrist_pos'][1], atol=1e-9)
    assert target['active_sides'] == ['right'] and BASIS.shape == (3, 3)


def test_gait_tells_stepping_from_sliding():
    from analyze_motion import gait
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


def test_kimodo_only_rollout_feeds_each_reference_back(tmp_path):
    from rollout_kimodo import follow
    from analyze_motion import gait
    policy, sample, rotation, root = make_policy(tmp_path, WALK)
    policy.foot_names = [f'{s}_ankle_roll_link' for s in SIDES]
    rows = follow(policy, sample(0), 5.)
    assert len(rows) >= 5*50-5 and policy.history_source == 'reference'
    np.testing.assert_allclose(rows[-1]['sent_root_world'][:2], (root+rotation @ [.5, 0, 0])[:2], atol=2e-3)
    # The stand-in slides the body along the path without stepping, and the gait measure says so.
    walked = [r for r in rows if 10.6 < r['time'] < 12.9]
    assert gait([r['sent_feet'] for r in walked], [r['sent_root_world'] for r in walked])['both_moving_share'] > .6
