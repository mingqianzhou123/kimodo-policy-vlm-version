"""Ways of using Kimodo while walking (LocomotionPolicy options), with the path-following stand-in."""
import json

import numpy as np
import pytest
import torch

from kimodo_vla.model.native_policy import ConstraintGuidance, SeparatedGuidance
from kimodo_vla.oracle import demo_walks as dw
from kimodo_vla.oracle.locomotion import CANVAS, SPARSE_FRAMES, load_goal, walk_allowance
from rollout_kimodo import follow
from test_locomotion import WALK, make_policy


def path_mask(policy):
    rep = policy.model.representation
    m = torch.zeros(rep.motion_rep_dim, dtype=torch.bool)
    m[[policy.root_slot, policy.root_slot+2]] = True
    m[policy.heading_slice] = True
    return m


@pytest.mark.parametrize('kind,frames', [('dense', list(range(1, CANVAS))), ('sparse', list(SPARSE_FRAMES)), ('none', [])])
def test_root_constraints_choose_which_walking_frames_are_conditioned(tmp_path, kind, frames):
    policy, start, _, _ = make_policy(tmp_path, WALK)
    policy.root_constraints = kind
    follow(policy, start, 1.2)
    walk = policy.model.calls[0]
    conditioned = walk['mask'][:, path_mask(policy)].any(-1).nonzero().flatten().tolist()
    assert conditioned == [0]+frames                         # frame 0 is always the present pose
    plan = json.loads((tmp_path/'plans.jsonl').read_text().splitlines()[0])
    assert plan['root_constraints'] == kind and plan['text'] is None


def test_history_frames_limits_the_conditioning_history(tmp_path):
    policy, start, _, _ = make_policy(tmp_path, WALK)
    policy.history_frames = 1
    follow(policy, start, 3.)
    assert all(len(c['history']) == 1 for c in policy.model.calls)


def test_turn_goal_turns_the_heading_condition_and_keeps_the_feet_on_the_spot(tmp_path):
    goal = {'name': 'turn', 'root': {'xy': [0., 0.], 'yaw_deg': 90, 'yaw_rate_deg_s': 45}}
    assert walk_allowance(load_goal(goal)) == pytest.approx(9+5)
    with pytest.raises(ValueError):
        load_goal({'name': 'x', 'root': {'xy': [.3, 0.], 'yaw_deg': 90}})
    policy, start, _, _ = make_policy(tmp_path, goal)
    follow(policy, start, 1.2)
    walk = policy.model.calls[0]
    heading = walk['values'][:, policy.heading_slice].numpy()
    angle = np.degrees(np.unwrap(np.arctan2(heading[1:, 1], heading[1:, 0])))
    turned = abs(angle[-1]-angle[0])
    assert 45*48/30-1 < turned < 45*48/30+1                # 45 deg/s over the 48 frames after frame 1
    assert policy.yaw_remaining(start) == pytest.approx(90.)  # not turned yet: all of it still to go
    root = walk['values'][1:, [policy.root_slot, policy.root_slot+2]].numpy()
    assert np.abs(root-root[0]).max() < 1e-6                  # on the spot
    assert policy.phases[0]['phase'] == 'walk'


def test_a_half_turn_goes_left_and_is_tracked_past_180_degrees(tmp_path):
    from scipy.spatial.transform import Rotation
    goal = {'name': 'turn', 'root': {'xy': [0., 0.], 'yaw_deg': 180, 'yaw_rate_deg_s': 45}}
    policy, start, rotation, _ = make_policy(tmp_path, goal)
    follow(policy, start, .7)                                 # enters the walk
    assert policy.yaw_remaining(start) == pytest.approx(180.)
    walk = policy.model.calls[0]['values'][:, policy.heading_slice].numpy()
    left = policy.heading_features([0., 10.]).numpy()
    # the conditioned heading moves the way +10 degrees (left) moves it
    cross2 = lambda a, b: a[0]*b[1]-a[1]*b[0]   # numpy 2 no longer takes 2-d vectors in np.cross
    assert np.sign(cross2(walk[1], walk[-1])) == np.sign(cross2(left[0], left[1]))
    for yaw in (90., 170., 200., 260.):                       # the body keeps turning left
        sample = dict(start, root_quat=np.roll(Rotation.from_matrix(
            Rotation.from_euler('z', np.radians(yaw)).as_matrix() @ rotation).as_quat(), 1).tolist())
        left_to_go = policy.yaw_remaining(sample)
    assert left_to_go == pytest.approx(180.-260.)


def test_heading_features_turn_left_for_positive_yaw(tmp_path):
    policy, start, _, _ = make_policy(tmp_path, WALK)
    follow(policy, start, .2)                                 # initializes the frame
    h = policy.heading_features([0., 30.]).numpy()
    a = np.degrees(np.arctan2(h[:, 1], h[:, 0]))
    assert abs(abs(a[1]-a[0])-30) < 1e-3
    np.testing.assert_allclose(h[0], policy.standing_features[policy.heading_slice].numpy(), atol=1e-5)


def segment_file(tmp_path, n=260, window=200):
    a = np.zeros((n, 40))
    a[:, 3:9] = np.eye(3)[:, :2].reshape(6)
    a[1:, 0] = .4/dw.RATE                                     # 0.4 m/s forward
    a[:, 2] = .78
    a[:, 9:38] = .05
    path = tmp_path/'lead.json'
    path.write_text(json.dumps({'name': 'lead', 'direction': 'forward', 'window': [window, n-1],
                                'actions': a.tolist(), 'source': {}}))
    return path


def test_lead_in_plays_the_recording_then_walks_from_where_it_ended(tmp_path):
    policy, start, rotation, root = make_policy(tmp_path, {'name': 'side', 'root': {'xy': [0., .3]}})
    policy.lead = dw.load_segment(segment_file(tmp_path))
    rows = follow(policy, start, 7.)
    phases = [p['phase'] for p in policy.phases]
    assert phases[:2] == ['lead', 'walk']
    lead = [r for r in rows if r['phase'] == 'lead']
    assert len(lead) == 50+200                                # 1 s blend, then the 4 s before the window
    plans = [json.loads(x) for x in (tmp_path/'plans.jsonl').read_text().splitlines()]
    assert all(not p.get('model_called', True) for p in plans if p['phase'] == 'lead')
    # the recording moved the body forward; the walk's goal is 0.3 m to the left of where it ended
    after = np.array(lead[-1]['body_xy'])
    moved = rotation[:2, :2].T @ (after-root[:2])
    assert moved[0] > .4*3.5
    goal = np.array(json.loads((tmp_path/'target.json').read_text())['root_goal_world'])
    np.testing.assert_allclose(rotation[:2, :2].T @ (goal-after), [0., .3], atol=.02)
    # and the first walking plan was conditioned on the recorded walk as its history
    first = policy.model.calls[0]['history'][:, [policy.root_slot, policy.root_slot+2]].numpy()
    assert np.ptp(first[:, 1]) > .4*3.                        # 100 frames of forward walking


def test_separated_guidance_reduces_to_constraint_guidance_without_text():
    calls = []

    def model(**kw):
        calls.append(kw)
        return kw['text_feat'].sum()*10+kw['motion_mask'].float().sum()
    kw = dict(text_feat=torch.ones(1, 1, 3), text_feat_pad_mask=torch.ones(1, 1, dtype=torch.bool),
              motion_mask=torch.ones(1, 4, 2, dtype=torch.bool))
    assert SeparatedGuidance(model, 0., 2.)(**kw) == ConstraintGuidance(model, 2.)(**kw)
    u, c, t = 0., 8., 30.                                     # empty, constraints only, text only
    assert SeparatedGuidance(model, 3., 2.)(**kw) == pytest.approx(u+2*(c-u)+3*(t-u))


def test_segment_generator_plans_whole_segments_without_history(tmp_path, monkeypatch):
    from kimodo_vla.oracle import locomotion as lo
    policy, start, rotation, root = make_policy(tmp_path, {'name': 'fwd', 'root': {'xy': [.6, 0.]}})
    policy.generator_mode, policy.segment_frames, policy.segment_steps, policy.walk_steps = 'segment', 120, 100, 150
    calls = []
    stand_in = policy.model

    def fake_sample(model, values, mask, *, first_heading, text_feat, text_weight, constraint_weight, steps, generator):
        calls.append(dict(frames=len(values), steps=steps, first=mask[0].any().item(), heading=first_heading))
        start_ = stand_in.representation.slice_dict['smooth_root_pos'].start
        r = stand_in.pose['root_positions'].repeat(len(values), 1)
        r[:, 0], r[:, 2] = values[:, start_], values[:, start_+2]
        local = stand_in.pose['local_rot_mats'].repeat(len(values), 1, 1, 1)
        return stand_in.representation(local, r, to_normalize=False)[0], {'local_rot_mats': local, 'root_positions': r}
    monkeypatch.setattr(lo, 'sample_segment', fake_sample)
    follow(policy, start, 8.)
    assert calls and all(c['frames'] == 120 and c['steps'] == 100 and c['first'] for c in calls)
    assert stand_in.calls == []                               # predict_future (the history canvas) never called
    plans = [json.loads(x) for x in (tmp_path/'plans.jsonl').read_text().splitlines()]
    walk = [p for p in plans if p['phase'] == 'walk']
    assert walk and all(p['generator'] == 'segment' and p['canvas_frames'] == 120 and p['sent_steps'] == 150 for p in walk)
    with pytest.raises(ValueError):
        lo.LocomotionPolicy.__init__(lo.LocomotionPolicy.__new__(lo.LocomotionPolicy), motion_goal=WALK,
                                     generator='segment', segment_seconds=4., walk_steps=250)


def test_native_generator_continues_the_sent_reference_over_a_few_frames(tmp_path, monkeypatch):
    from kimodo_vla.oracle import locomotion as lo
    policy, start, rotation, root = make_policy(tmp_path, {'name': 'fwd', 'root': {'xy': [.6, 0.]}})
    policy.generator_mode, policy.segment_frames, policy.segment_steps = 'native', 90, 100
    policy.transition_frames, policy.walk_steps = 5, 50
    rep = policy.model.representation
    slot, calls, stand_in = rep.slice_dict['smooth_root_pos'].start, [], policy.model
    joints = rep.slice_dict['local_joints_positions']

    def fake_sample(model, values, mask, *, first_heading, text_feat, text_weight, constraint_weight, steps, generator):
        calls.append(dict(frames=len(values), mask=mask.clone(), values=values.clone(), heading=first_heading,
                          text=text_feat))
        r = stand_in.pose['root_positions'].repeat(len(values), 1)
        known = mask[:, slot]
        xz = values[:, [slot, slot+2]].clone()
        for k in range(len(xz)):                              # hold the last conditioned ground point
            if not known[k] and k:
                xz[k] = xz[k-1]
        r[:, 0], r[:, 2] = xz[:, 0], xz[:, 1]
        local = stand_in.pose['local_rot_mats'].repeat(len(values), 1, 1, 1)
        feats = stand_in.representation(local, r, to_normalize=False)
        return (feats[0] if feats.ndim == 3 else feats), {}
    monkeypatch.setattr(lo, 'sample_segment', fake_sample)
    follow(policy, start, 6.)
    assert calls and stand_in.calls == []                     # the history canvas is never used
    plans = [json.loads(x) for x in (tmp_path/'plans.jsonl').read_text().splitlines()]
    assert len(plans) == len(calls) and all(p['generator'] == 'native' and p['transition_frames'] == 5 for p in plans)
    phases = [p['phase'] for p in plans]
    assert 'walk' in phases and 'stop' in phases
    for c, phase in zip(calls, phases):
        body = c['mask'][:, joints].all(-1)
        assert c['frames'] == 5+90 and body[:5].all()        # the five transition frames, in full
        assert c['values'][0, slot] == 0. and c['values'][0, slot+2] == 0.   # moved to the ground origin
        assert c['text'] is None                              # no text given
        if phase == 'walk':
            assert not body[5:].any() and c['mask'][5:, slot].all()   # then only the path, on every frame
        else:                                                 # the stop plan ends in the standing pose
            assert body[5+lo.STAND_FROM-1:].all()
    reach = lo.LocomotionPolicy.__new__(lo.LocomotionPolicy)
    with pytest.raises(ValueError):
        lo.LocomotionPolicy.__init__(reach, motion_goal=WALK, text='auto')            # 'auto' is native only
    with pytest.raises(ValueError):
        lo.LocomotionPolicy.__init__(reach, motion_goal=WALK, generator='native', transition_frames=0)


@pytest.mark.parametrize('gain', [1., 2.])
def test_approach_gain_aims_past_the_goal_by_what_is_left(tmp_path, gain):
    policy, start, rotation, root = make_policy(tmp_path, {'name': 'near', 'root': {'xy': [.1, 0.]}})
    policy.approach_gain = gain
    follow(policy, start, 1.2)
    walk = policy.model.calls[0]
    path = walk['values'][:, [policy.root_slot, policy.root_slot+2]].numpy()
    assert np.linalg.norm(path[-1]-path[0]) == pytest.approx(.1*gain, abs=2e-3)


@pytest.mark.parametrize('open_loop', [False, True])
def test_open_loop_plans_from_the_reference_not_the_body(tmp_path, monkeypatch, open_loop):
    """With a body that covers half of each step, open loop never moves the plans onto it.

    (The stand-in tracker assumes each plan starts on the body, so the body's own end point is not checked.)"""
    from kimodo_vla.oracle import locomotion as lo
    policy, start, rotation, root = make_policy(tmp_path, {'name': 'fwd', 'root': {'xy': [.6, 0.]}})
    policy.generator_mode, policy.segment_frames, policy.segment_steps = 'native', 90, 100
    policy.transition_frames, policy.walk_steps, policy.open_loop = 5, 50, open_loop
    rep, stand_in = policy.model.representation, policy.model
    slot = rep.slice_dict['smooth_root_pos'].start

    def fake_sample(model, values, mask, *, first_heading, text_feat, text_weight, constraint_weight, steps, generator):
        r = stand_in.pose['root_positions'].repeat(len(values), 1)
        xz = values[:, [slot, slot+2]].clone()
        for k in range(1, len(xz)):
            if not mask[k, slot]:
                xz[k] = xz[k-1]
        r[:, 0], r[:, 2] = xz[:, 0], xz[:, 1]
        feats = stand_in.representation(stand_in.pose['local_rot_mats'].repeat(len(values), 1, 1, 1), r, to_normalize=False)
        return (feats[0] if feats.ndim == 3 else feats), {}
    monkeypatch.setattr(lo, 'sample_segment', fake_sample)
    rows = follow(policy, start, 8., carry=.5)
    plans = [json.loads(x) for x in (tmp_path/'plans.jsonl').read_text().splitlines()]
    shifts = [p['anchor_shift_m'] for p in plans]
    reference = rotation[:2, :2].T @ (np.array(rows[-1]['sent_root_world'][:2])-root[:2])
    if open_loop:
        assert max(shifts) == 0.                              # never moved onto the body
        assert reference[0] == pytest.approx(.6, abs=.06)     # the reference carried out the command
    else:
        assert max(shifts) > .05                              # replanned from the body
    with pytest.raises(ValueError):
        lo.LocomotionPolicy.__init__(lo.LocomotionPolicy.__new__(lo.LocomotionPolicy), motion_goal=WALK, open_loop=True)


def test_native_stop_after_a_turn_stands_facing_the_turned_heading(tmp_path):
    from kimodo_vla.motion.feature_utils import compute_heading_angle
    policy, start, rotation, root = make_policy(tmp_path, {'name': 'turn', 'root': {'xy': [0., 0.], 'yaw_deg': 90}})
    policy.initialize(start)
    rep = policy.model.representation
    standing = policy.standing_features.clone()
    target = policy.heading_features(90.)[0]
    turned = policy.turned_standing(standing, target)
    assert torch.allclose(turned[policy.heading_slice], target, atol=1e-4)
    # the joints turned with it: the hip heading of the decoded pose is the target's
    def hips(f):
        return float(compute_heading_angle(rep.inverse(f[None, None], is_normalized=False)['posed_joints'], rep.skeleton)[0, 0])
    want = float(torch.atan2(target[1], target[0]))
    assert abs((hips(turned)-want+np.pi) % (2*np.pi)-np.pi) < np.radians(2)
    assert abs((hips(standing)-want+np.pi) % (2*np.pi)-np.pi) > np.radians(80)
