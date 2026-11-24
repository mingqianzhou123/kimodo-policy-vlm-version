"""Gate, judge, attribution and geometry of the blocking box grasp; no weights or simulator."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from arena.grasp_judge import Attempt, Criteria, GraspJudge, summarize
from arena.grasp_loop import GraspLoop, up_z
from arena.box_layout import BoxMotion, cell_name, in_region, map_cells, place, supported
from arena.grasp_scene import hand_side, split_forces
from kimodo_vla.controllers.gripper_gate import SynchronizedGripperGate
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.model.features import G1_XML_PATH, KimodoFeatures
from kimodo_vla.oracle.geometry import SIDES, XmlFK
from kimodo_vla.oracle.grasp_policy import (BoxGraspPolicy, WristPath, clip_profile, face_yaw, graspable, half_width,
                                             stance, waypoints)
import overview
from report_grasp import displacement, pick_region, summarize as summarize_runs
from run_grasp_suite import region_prefix
from run_oracle import parse_args

CONFIG = json.loads((Path(__file__).resolve().parents[2]/'configs/grasp-loop.json').read_text())
CFG = CONFIG['policy']
BOX = [-.46, -4.6, .85]   # centre of mass; the asset origin is on the bottom face
EYE = np.eye(3)
TARGETS = [[0., .2, 1.], [0., -.2, 1.]]
HELD = dict(left_n=1., right_n=1., robot_other_n=0., external_n=0.)
RESTING = dict(left_n=0., right_n=0., robot_other_n=0., external_n=.49)


def quat(yaw_deg):
    x, y, z, w = Rotation.from_euler('z', yaw_deg, degrees=True).as_quat()
    return [w, x, y, z]


def tick(gate, t, left=TARGETS[0], right=TARGETS[1], speeds=(0., 0.)):
    return gate.update(now_s=t, observation_time_s=t, positions=[left, right],
                       rotations=[EYE, EYE], speeds_mps=list(speeds))


def test_both_hands_close_on_the_same_tick_only_after_both_arrived():
    gate = SynchronizedGripperGate()
    assert gate.submit('round-0', TARGETS, [EYE, EYE], 5.)
    assert not gate.submit('round-0', TARGETS, [EYE, EYE], 5.)   # a replan does not rearm it
    far = [0., -.4, 1.]
    for t in np.arange(0, .3, .02):                              # left has arrived, right has not
        assert not tick(gate, t, right=far)
    assert gate.gates[0].status == 'fired' and gate.gates[1].status == 'pending' and not gate.closed
    states = [tick(gate, t) for t in np.arange(.3, .5, .02)]
    assert states.index(True) == 5 and gate.fired_at == pytest.approx(.4)   # right needs its own dwell


def test_a_hand_that_leaves_while_waiting_has_to_arrive_again():
    gate = SynchronizedGripperGate()
    gate.submit('round-0', TARGETS, [EYE, EYE], 5.)
    far = [0., -.4, 1.]
    for t in np.arange(0, .2, .02):
        tick(gate, t, right=far)
    tick(gate, .2, left=[0., .4, 1.], right=far)                 # left drifts out before right arrives
    assert gate.gates[0].status == 'pending'
    assert not any(tick(gate, t, left=[0., .4, 1.]) for t in np.arange(.22, .5, .02))
    assert [tick(gate, t) for t in np.arange(.5, .64, .02)][-1]


def test_deadline_keeps_the_hands_open_and_release_reopens_them():
    gate = SynchronizedGripperGate()
    gate.submit('round-0', TARGETS, [EYE, EYE], .1)
    assert not any(tick(gate, t, right=[0., -.4, 1.]) for t in np.arange(0, .3, .02))
    assert gate.status == 'timed_out'
    gate.release()
    gate.submit('round-1', TARGETS, [EYE, EYE], 5.)
    assert [tick(gate, t) for t in np.arange(.3, .5, .02)][-1]
    gate.release()
    assert not gate.closed and not tick(gate, .5)


def test_success_needs_height_grip_and_no_support_for_half_a_second():
    judge = GraspJudge(.85)
    times = np.arange(0, 1.2, .02)
    assert not any(judge.update(t, .85 + .099, HELD) for t in times)                    # 1 mm short
    assert not any(judge.update(t, 1., {**HELD, 'external_n': .2}) for t in times + 2)   # still supported
    assert not any(judge.update(t, 1., {**HELD, 'right_n': 0.}) for t in times + 4)      # one hand only
    run = [judge.update(6 + t, 1., HELD if round(t, 2) != .3 else RESTING) for t in times]
    assert run.index(True) == 16 + 25 and judge.success_time == pytest.approx(6.82)      # the gap restarts the clock


def attempt(rows, **kwargs):
    a = Attempt(0, 0., tuple(BOX), .85, **kwargs)
    for i, (height, forces, extra) in enumerate(rows):
        a.update(i*.02, [BOX[0] + extra.get('dx', 0.), BOX[1], height], extra.get('up', 1.), forces, extra.get('closed', False))
    a.close(len(rows)*.02)
    return a


def test_attempt_labels_follow_what_happened_to_the_box():
    rest = (.85, RESTING, {})
    touch = (.85, {**RESTING, 'left_n': .3}, {})
    held = (.90, HELD, {'closed': True})
    fallen = (.85, RESTING, {'up': .3})
    assert attempt([rest]*20).label == 'no_contact'
    assert attempt([rest]*5 + [touch]*5 + [rest]*5).label == 'touched_not_held'
    assert attempt([touch]*3 + [held]*4 + [rest]*10).label == 'touched_not_held'     # too brief to be a hold
    slipped = attempt([touch]*3 + [held]*10 + [rest]*10)
    assert slipped.label == 'slipped' and slipped.events['held'] == pytest.approx(.06)
    assert slipped.events['slipped'] == pytest.approx(.26) and slipped.events['closed'] == pytest.approx(.06)
    assert attempt([touch]*3 + [held]*10).label == 'held_at_cutoff'
    assert attempt([touch]*3 + [fallen]*5).label == 'knocked_over'
    assert attempt([touch]*3 + [(.85, RESTING, {'dx': .12})]*3).label == 'knocked_over'
    assert attempt([touch]*3 + [held]*10 + [fallen]*10).label == 'slipped'            # dropped, not knocked over


def test_summary_reports_contact_and_slip_counts():
    rows = [attempt([(.85, RESTING, {})]*5), attempt([(.90, HELD, {})]*10 + [(.85, RESTING, {})]*10)]
    s = summarize(rows, None, None, 0.)
    assert (s['attempt_count'], s['touched_attempts'], s['held_attempts'], s['slipped_attempts']) == (2, 1, 1, 1)
    assert not s['success'] and s['attempts_to_success'] is None and not s['first_attempt_success']


def test_contact_forces_are_split_by_hand_and_everything_that_is_not_the_robot():
    names = ['left_hand_palm_link', 'left_hand_index_1_link', 'right_hand_thumb_2_link', 'left_elbow_link',
             'right_wrist_yaw_link', 'right_hand_camera_base_link', 'right_wrist_pitch_link', 'torso_link']
    assert [hand_side(n) for n in names] == ['left', 'left', 'right', None, 'right', 'right', None, None]
    matrix = np.array([[0, -1., 0], [0, -.5, 0], [0, 2., 0], [.2, 0, 0], [0, 1., 0], [0, 0, 0], [0, 0, 0], [0, 0, .01]])
    forces = split_forces(matrix.sum(0) + [0, 0, .49], matrix, names)
    assert forces.pop('bodies') == {'left_hand_palm_link': 1., 'left_hand_index_1_link': .5, 'right_hand_thumb_2_link': 2.,
                                    'left_elbow_link': .2, 'right_wrist_yaw_link': 1.}
    assert forces == pytest.approx(dict(left_n=1.5, right_n=3., robot_other_n=.2 + 5e-4, external_n=.49), abs=1e-3)
    with pytest.raises(ValueError):
        split_forces(np.zeros(3), matrix[:2], names)


def test_grasp_loop_closes_judges_and_stops(tmp_path):
    stopped = []
    loop = GraspLoop(tmp_path, CONFIG, stop=lambda: stopped.append(True), post_success_steps=2)
    w = waypoints(BOX, quat(0), [-.84, -4.6], CFG)
    def sample(t, wrists, height=.85):
        return {'time': t, 'wrist_pos': wrists.tolist(), 'wrist_rot': w['rotation'].tolist(), 'root_pos': [-.84, -4.6, .78],
                'wrist_velocity': [[0.]*3]*2, 'box_com': [BOX[0], BOX[1], height], 'box_quat': quat(0)}
    info = {'round': 0, 'phase': 'advance', 'latch_time': 0., 'latched_box_pos': BOX, 'hands': 'open', 'event': None}
    loop.on_plan(info, sample(0., w['pre']))
    assert not loop.before_step(sample(0., w['pre']))                # nothing is armed yet
    loop.after_step(sample(.02, w['pre']), RESTING)
    event = {'id': 'round-0', 'positions': w['pre'].tolist(), 'rotations': w['rotation'].tolist(), 'deadline': 3.}
    loop.on_plan({**info, 'hands': 'gate', 'event': event}, sample(.02, w['via']))
    assert not loop.before_step(sample(.03, w['via']))               # still on the way in
    closed = [loop.before_step(sample(t, w['pre'])) for t in np.arange(.04, .3, .02)]
    assert closed.index(True) == 5
    for t in np.arange(.3, 1.2, .02):
        loop.after_step(sample(t, w['lift'], 1.), HELD)
    result = json.loads((tmp_path/'grasp-result.json').read_text())
    assert result['success'] and result['attempts'][0]['label'] == 'success' and stopped
    assert result['completion_s'] == pytest.approx(.3 + .5, abs=.021) and result['first_attempt_success']
    loop.on_plan({**info, 'round': 1, 'phase': 'wait', 'event': None}, sample(1.2, w['pre']))
    assert not loop.closed and len(loop.attempts) == 2               # a new round opens the hands


def test_up_axis_of_a_tipped_box():
    assert up_z(quat(40)) == pytest.approx(1.)
    x, y, z, w = Rotation.from_euler('x', 60, degrees=True).as_quat()
    assert up_z([w, x, y, z]) == pytest.approx(.5)


@pytest.mark.parametrize('yaw', [0, 20, -45, 44])
def test_waypoints_open_past_the_thumbs_and_press_fists_on_the_widest_line(yaw):
    w = waypoints(BOX, quat(yaw), [-.89, -4.6], CFG)
    angle = np.radians(yaw)
    extent = .105 * (abs(np.cos(angle)) + abs(np.sin(angle)))
    pitch = np.radians(CFG['wrist_pitch_down_deg'])
    lateral = lambda wrists: abs(wrists[:, 1] - BOX[1])
    assert np.all(lateral(w['pre']) - CFG['palm_offset_m'] >= extent + .03)       # the fingers pass the widest line
    thumb_x = w['pre'][0, 0] + CFG['thumb_forward_m'] - BOX[0]
    assert np.all(lateral(w['pre']) - CFG['thumb_reach_m'] >= half_width(angle, .105, thumb_x) + .03)   # the thumb too
    assert np.all(lateral(w['pre']) <= extent + .12)                              # and no wider than the old opening
    fist = lateral(w['squeeze']) - CFG['fist_offset_m']                            # surface of a closed hand
    np.testing.assert_allclose(fist, .105 - CFG['squeeze_m'])                      # presses a squared-up box
    contact = w['squeeze'][:, 0] + CFG['fist_centre_m'] * np.cos(pitch)
    beyond = {0: 0., 20: CFG['grip_forward_m'] / 3, -45: CFG['grip_forward_m'], 44: CFG['grip_forward_m']}[yaw]
    np.testing.assert_allclose(contact, BOX[0] + beyond)                           # past the widest line if turned
    np.testing.assert_allclose(w['squeeze'][:, 2] - CFG['fist_centre_m'] * np.sin(pitch), BOX[2] + CFG['grip_height_m'])
    assert w['squeeze'][0, 1] > BOX[1] > w['squeeze'][1, 1]
    np.testing.assert_allclose(w['lift'][:, 2] - w['squeeze'][:, 2], CFG['lift_m'])
    assert np.all(w['via'][:, 0] <= -.89 + CFG['via_forward_m'] + 1e-9) and np.all(w['via'][:, 2] > w['pre'][:, 2])
    assert w['rotation'][0] @ [1, 0, 0] == pytest.approx([np.cos(pitch), 0, -np.sin(pitch)])
    fingertip = w['via'][:, 0] + .2 * (w['via_rotation'][0] @ [1, 0, 0])[0]       # fingers stay down on the way up
    assert np.all(fingertip <= -.89 + CFG['via_forward_m'] + .2 * np.cos(np.radians(CFG['via_pitch_down_deg'])) + 1e-9)
    assert CFG['via_pitch_down_deg'] >= CFG['wrist_pitch_down_deg']


def test_half_width_of_a_turned_box():
    assert half_width(0., .105, .05) == pytest.approx(.105) and half_width(0., .105, .2) == 0.
    diamond = np.radians(45)
    assert half_width(diamond, .105, 0.) == pytest.approx(.105 * np.sqrt(2))
    assert half_width(diamond, .105, -.05) == pytest.approx(.105 * np.sqrt(2) - .05)
    assert half_width(np.radians(-20), .105, .03) == pytest.approx(half_width(np.radians(20), .105, -.03))


def test_face_yaw_stance_and_graspable_region():
    assert face_yaw(quat(100)) == pytest.approx(np.radians(10))
    assert face_yaw(quat(-50)) == pytest.approx(np.radians(40))
    reach = lambda box_x: waypoints([box_x, -4.6, .85], quat(40), [-2., -4.6], CFG)['pre'][0, 0] - stance([box_x, -4.6, .85], CFG)[0]
    assert reach(-.58) == pytest.approx(CFG['stance_reach_m']) and reach(-.55) == pytest.approx(CFG['stance_reach_m'])
    deep = stance([-.40, -4.6, .85], CFG)                            # cannot follow a deep box onto the table
    assert deep[0] == pytest.approx(CFG['table_edge_x_m'] - CFG['stance_clearance_m']) and reach(-.40) > CFG['stance_reach_m']
    assert stance(BOX, CFG)[1] == -4.6
    assert graspable(BOX, quat(30), CFG)
    assert not graspable([BOX[0], BOX[1], .105], quat(0), CFG)       # on the floor
    assert not graspable([.1, BOX[1], .85], quat(0), CFG)           # pushed out of reach
    x, y, z, w = Rotation.from_euler('y', 90, degrees=True).as_quat()
    assert not graspable(BOX, [w, x, y, z], CFG)                     # on its side


def test_clip_profile_stops_at_the_goal_or_keeps_a_steady_pace():
    arriving = clip_profile(.5, True, stop_frame=40)
    assert arriving[0] == 0 and arriving[40:] == pytest.approx(.5) and np.all(np.diff(arriving) >= 0)
    assert np.diff(arriving)[[0, 39]] == pytest.approx(0., abs=1e-3)       # leaves and arrives at rest
    passing = clip_profile(.8, False)
    speed = np.diff(passing)
    assert passing[0] == 0 and passing[-1] == pytest.approx(.8) and np.all(speed >= 0)
    assert speed[20:] == pytest.approx(speed[-1]) and speed[0] < .1 * speed[-1]   # eases in, then steady


def test_wrist_path_is_continuous_and_holds_between_and_after_pieces():
    start = np.zeros((2, 3))
    tilt = np.stack([Rotation.from_euler('y', 25, degrees=True).as_matrix()]*2)
    path = WristPath(1., start, np.stack([EYE, EYE]))
    assert path.to(start + [.2, 0, 0], tilt, 1.) == 2.
    assert path.to(start + [.2, 0, .1], tilt, .5, start=3.) == 3.5
    np.testing.assert_allclose(path(1.5)[0], start + [.1, 0, 0])
    assert np.degrees(np.linalg.norm(Rotation.from_matrix(path(1.5)[1][0]).as_rotvec())) == pytest.approx(12.5)
    np.testing.assert_allclose(path(2.7)[0], start + [.2, 0, 0])     # waiting for the delayed piece
    np.testing.assert_allclose(path(9.)[0], start + [.2, 0, .1])
    assert (path.piece_end(1.2), path.piece_end(2.), path.piece_end(9.)) == (2., 3., 3.5)


def test_grasp_loop_run_defaults():
    a = parse_args(['--run-id', 'x', '--grasp-loop', '--seed', '3004'])
    assert (a.steps, a.seed, a.latch_delay_steps, a.dynamics_profile) == (1450, 3004, 75, 'actuator')
    assert not a.actuator_audit and not a.visual_markers
    assert parse_args(['--run-id', 'x']).seed == 1000


def test_debug_and_heldout_seeds_do_not_overlap():
    seeds = CONFIG['seeds']
    assert len(seeds['heldout']) == 30 and not set(seeds['debug']) & set(seeds['heldout'])


def test_report_counts_episodes_and_attempts_with_the_plan_denominators():
    episode = lambda labels, termination: {'success': termination == 'success', 'termination': termination,
        'first_attempt_success': labels[0] == 'success', 'attempt_count': len(labels), 'completion_s': 9.5,
        'attempts_to_success': labels.index('success') + 1 if 'success' in labels else None,
        'attempts': [{'label': x, 'touched': x != 'no_contact', 'held': x in ('success', 'slipped', 'held_at_cutoff')}
                     for x in labels]}
    s = summarize_runs([episode(['success'], 'success'), episode(['slipped', 'success'], 'success'),
                        episode(['no_contact', 'touched_not_held'], 'timeout'), episode(['knocked_over'], 'fall')])
    assert (s['episodes'], s['successes'], s['first_attempt_successes'], s['attempts']) == (4, 2, 1, 6)
    assert s['terminations'] == {'success': 2, 'timeout': 1, 'fall': 1}
    assert s['contact_rate'] == pytest.approx(5/6) and s['slip_rate'] == pytest.approx(1/3)
    assert s['attempts_to_success'] == [1, 2]
    assert summarize_runs([])['success_rate'] is None


def test_progress_page_scores_a_grasp_run_by_the_physical_verdict(tmp_path):
    run = tmp_path/'exp1-static-x-s3000'
    run.mkdir()
    (run/'run.json').write_text(json.dumps({'status': 'completed', 'success_rate_evaluation': True}))
    (run/'episode.json').write_text(json.dumps({'success': False, 'seed': 3000}))
    (run/'grasp-result.json').write_text(json.dumps({'success': True}))
    assert overview.describe(run, tmp_path/'overview')['wins'] == 1


def scripted_policy(tmp_path, delay=75):
    """The round logic without weights: everything except the two calls into the diffusion model."""
    policy = BoxGraspPolicy.__new__(BoxGraspPolicy)
    policy.cfg, policy.latch_delay, policy.run, policy.mode = CFG, delay, tmp_path, 'hold'
    policy.stand_still = False
    policy.features, policy.fk = KimodoFeatures(50.), XmlFK(G1_XML_PATH, G1_JOINT_NAMES)
    names = policy.features.representation.skeleton.bone_order_names
    policy.ids = [[names.index(f'{s}_{j}_skel') for j in ('wrist_yaw', 'hand_roll')] for s in SIDES]
    policy.reset(1000)
    return policy


def standing_sample(policy, time, box, closed=False, root_xy=None):
    root = np.array([*(stance(BOX, CFG) if root_xy is None else root_xy), .78])
    joints = np.zeros(29)
    feet = ['left_ankle_roll_link', 'right_ankle_roll_link']
    wrists, wrist_rot = policy.fk(joints, root, EYE)
    foot, foot_rot = policy.fk(joints, root, EYE, body_names=feet)
    return {'time': time, 'root_pos': root.tolist(), 'root_quat': [1., 0, 0, 0], 'joints': joints.tolist(),
            'reference_root_pos': root.tolist(), 'reference_root_quat': [1., 0, 0, 0], 'reference_joints': joints.tolist(),
            'wrist_pos': wrists.tolist(), 'wrist_rot': wrist_rot.tolist(), 'foot_pos': foot.tolist(),
            'foot_rot': foot_rot.tolist(), 'box_com': list(box), 'box_quat': quat(10), 'box_pos': list(box),
            'gripper_command': [int(closed)] * 2}


def run_until(policy, start, phase, box, closed=False, limit=10.):
    """Step the round logic at 50 Hz until it enters ``phase``; returns that time."""
    for step in range(1, int(limit * 50)):
        time = start + step * .02
        policy.advance([standing_sample(policy, time, box, closed)])
        if policy.round.phase == phase:
            return time
    raise AssertionError(f'never reached {phase}, stuck in {policy.round.phase}')


def test_a_round_latches_the_box_once_waits_and_follows_the_measured_hands(tmp_path):
    policy = scripted_policy(tmp_path)
    policy.begin_episode(standing_sample(policy, 0., BOX))
    moved = [BOX[0] + .2, BOX[1] - .3, BOX[2]]                       # the box moves right after the latch
    start = run_until(policy, 0., 'raise', moved)
    assert start == pytest.approx(75 * .02) and policy.round.box_pos == BOX
    np.testing.assert_allclose(policy.way['squeeze'].mean(0)[1], BOX[1])      # still aiming at the latched pose
    advance = run_until(policy, start, 'advance', moved)
    assert advance == pytest.approx(start + CFG['raise_s'])
    info = policy.grasp_info()
    assert info['hands'] == 'gate' and info['event']['id'] == 'round-0' and info['latched_box_pos'] == BOX
    assert info['event']['positions'] == policy.way['pre'].tolist()           # the hands close beside the box
    assert info['event']['deadline'] == pytest.approx(advance + CFG['advance_s'] + CFG['gate_timeout_s'])
    policy.advance([standing_sample(policy, advance + CFG['advance_s'] + .1, moved)])
    assert policy.round.phase == 'advance'                           # open hands: nothing is squeezed yet
    np.testing.assert_allclose(policy.path(advance + 9.)[0], policy.way['pre'])
    grip = run_until(policy, advance + CFG['advance_s'] + .1, 'grip', moved, closed=True)
    squeeze = run_until(policy, grip, 'squeeze', moved, closed=True)
    lift = run_until(policy, squeeze, 'lift', moved, closed=True)
    hold = run_until(policy, lift, 'hold', moved, closed=True)
    assert squeeze == pytest.approx(grip + CFG['grip_s'])            # the fingers get time to close first
    assert lift == pytest.approx(squeeze + CFG['squeeze_s'] + CFG['press_s']) and hold == pytest.approx(lift + CFG['lift_s'])
    np.testing.assert_allclose(policy.path(lift)[0], policy.way['squeeze'])
    np.testing.assert_allclose(policy.path(hold + 1)[0], policy.way['lift'])
    release = run_until(policy, hold, 'release', moved, closed=True)
    assert release == pytest.approx(hold + CFG['hold_s'], abs=.021) and policy.grasp_info()['hands'] == 'open'
    again = run_until(policy, release, 'wait', moved)
    assert policy.round.index == 1 and policy.round.box_pos == moved and policy.round.latch_time == pytest.approx(again)
    rows = [json.loads(line) for line in (tmp_path/'rounds.jsonl').read_text().splitlines()]
    assert [r['phase'] for r in rows if r['round'] == 0] == ['wait', 'raise', 'advance', 'grip', 'squeeze', 'lift', 'hold', 'release']


def test_a_gate_that_never_fires_releases_and_starts_over(tmp_path):
    policy = scripted_policy(tmp_path, delay=0)
    policy.begin_episode(standing_sample(policy, 0., BOX))
    advance = run_until(policy, 0., 'advance', BOX)
    release = run_until(policy, advance, 'release', BOX)
    assert release == pytest.approx(policy.round.event['deadline'], abs=.021)
    assert policy.round.notes['release_reason'] == 'gate_timeout'
    run_until(policy, release, 'wait', BOX)
    assert policy.round.index == 1


def test_too_close_to_the_table_is_not_a_stance(tmp_path):
    policy = scripted_policy(tmp_path, delay=0)
    goal = stance(BOX, CFG) + [CFG['walk_lead_m'], 0]
    policy.begin_episode(standing_sample(policy, 0., BOX))
    policy.walk = {'goal': goal, 'clips': 1, 'stopped': True}
    assert policy.at_stance(standing_sample(policy, .02, BOX, root_xy=goal + [.01, .03]))
    edge = CFG['table_edge_x_m']
    close = [edge - CFG['stance_min_clearance_m'] + .01, goal[1]]
    assert np.linalg.norm(np.subtract(close, goal)) <= CFG['stance_tolerance_m']     # near enough, but
    assert not policy.at_stance(standing_sample(policy, .02, BOX, root_xy=close))    # the hands would catch the edge
    assert not policy.at_stance(standing_sample(policy, .02, BOX, root_xy=goal + [-.2, 0]))


def test_far_from_the_stance_walks_and_an_unreachable_box_waits(tmp_path):
    policy = scripted_policy(tmp_path, delay=0)
    far = standing_sample(policy, 0., BOX, root_xy=[-1.58, -5.0])
    policy.begin_episode(far)
    policy.advance([dict(far, time=.02)])
    assert policy.round.phase == 'walk' and policy.walk['clips'] == 0
    np.testing.assert_allclose(policy.walk['goal'], stance(BOX, CFG) + [CFG['walk_lead_m'], 0])
    floor = [BOX[0], BOX[1], .105]
    policy = scripted_policy(tmp_path/'floor', delay=0)
    (tmp_path/'floor').mkdir()
    policy.begin_episode(standing_sample(policy, 0., floor))
    assert not policy.round.graspable
    run_until(policy, 0., 'idle', floor)
    run_until(policy, .02, 'wait', floor)
    assert policy.round.index == 1


def test_wrists_that_stop_short_beside_the_box_get_a_bounded_correction(tmp_path):
    policy = scripted_policy(tmp_path, delay=0)
    policy.begin_episode(standing_sample(policy, 0., BOX))
    advance = run_until(policy, 0., 'advance', BOX)
    arrived = advance + CFG['advance_s']
    short = lambda time: dict(standing_sample(policy, time, BOX), wrist_pos=(policy.way['pre'] - [.06, 0, 0]).tolist())
    policy.correct(short(arrived - .2))
    assert not policy.bias.any()                                     # still travelling: lag is not an error
    policy.correct(short(arrived))
    np.testing.assert_allclose(policy.bias, CFG['reach_feedback_gain'] * np.array([[.06, 0, 0]] * 2))
    np.testing.assert_allclose(policy.goal(arrived)[0], policy.way['pre'] + policy.bias)
    for step in range(1, 9):
        policy.correct(short(arrived + .5 * step))
    np.testing.assert_allclose(np.linalg.norm(policy.bias, axis=1), CFG['reach_feedback_max_m'])
    policy.round.phase = 'squeeze'
    before = policy.bias.copy()
    policy.correct(short(arrived + 5.))
    np.testing.assert_allclose(policy.bias, before)                  # pressing on the box is not a shortfall


# ---- standing still: the workspace map, the region and the moving box ---------------------

def test_map_cells_place_the_robot_by_the_stance_rule_and_the_box_by_depth_and_lateral():
    cells = map_cells(CONFIG)
    assert len(cells) == 100 and len({cell_name(c) for c in cells}) == 100
    assert cell_name(dict(depth=.16, lateral=-.1, yaw=40, extra=.05)) == 'd16-l-10-y40-e05'
    edge, y0 = CONFIG['scene']['table_edge_x_m'], CONFIG['stand']['robot_y_m']
    near, deep = place(.08, .1, 0, 0., CONFIG), place(.24, -.2, 40, .05, CONFIG)
    np.testing.assert_allclose(near['box_xy'], [edge + .08, y0 + .1])
    assert near['robot_xy'][1] == y0 and deep['robot_xy'][1] == y0
    rule = lambda d: max(CFG['stance_clearance_m'], edge - stance([edge + d, 0., .85], CFG)[0])
    assert near['clearance_m'] == pytest.approx(rule(.08)) and near['clearance_m'] > CFG['stance_clearance_m']
    assert deep['clearance_m'] == pytest.approx(CFG['stance_clearance_m'] + .05)


def test_region_placements_follow_the_seed_only_and_stay_inside(tmp_path):
    config = json.loads(json.dumps(CONFIG))
    config['stand'].update(region={'depth_m': [.1, .18], 'lateral_m': [-.05, .1]}, stance_extra_m=.03)
    a, b = in_region(6001, config), in_region(6001, config)
    assert a == b and a != in_region(6002, config)
    for seed in config['seeds']['region']:
        p = in_region(seed, config)
        assert .1 <= p['depth_m'] <= .18 and -.05 <= p['lateral_m'] <= .1 and abs(p['yaw_deg']) <= 45
        assert p['direction'] in (-1, 1) and p['stance_extra_m'] == .03
        np.testing.assert_allclose(p['segment_y'], np.array([-.05, .1]) + config['stand']['robot_y_m'])
    config['stand']['region'] = None
    with pytest.raises(ValueError):
        in_region(6000, config)                                       # no region fixed yet


def test_the_fixed_region_keeps_the_box_on_the_table_and_inside_the_map():
    stand, m = CONFIG['stand'], CONFIG['stand']['map']
    assert stand['stance_extra_m'] in m['stance_extra_m']
    for key, grid in (('depth_m', m['depth_m']), ('lateral_m', m['lateral_m'])):
        low, high = stand['region'][key]
        assert low in grid and high in grid and low <= high      # spans measured map points only
    table = [[CONFIG['scene']['table_edge_x_m'], -6.237], [.121, -3.763]]
    for seed in CONFIG['seeds']['region']:
        p = in_region(seed, CONFIG)
        assert all(supported([p['box_xy'][0], y], p['yaw_deg'], CONFIG['scene']['box_half_m'], table)
                   for y in [p['box_xy'][1], *p['segment_y']])


def test_a_box_over_the_table_edge_is_not_supported():
    table = [[-.641, -6.237], [.121, -3.763]]
    assert supported([-.5, -4.8], 40, .105, table)
    assert not supported([-.5, -3.8], 0, .105, table) and not supported([-.7, -4.8], 0, .105, table)


def test_box_motion_turns_at_both_ends():
    m = BoxMotion([0., .3], .1, .1, 1)
    assert m.position(0) == pytest.approx(.1) and m.position(1.) == pytest.approx(.2) and m.velocity(1.) == .1
    assert m.position(3.) == pytest.approx(.2) and m.velocity(3.) == -.1          # turned at 0.3 after 2 s
    assert m.position(6.) == pytest.approx(.1) and m.velocity(6.) == .1           # turned at 0 after 5 s
    assert (m.turns(0, 1.9), m.turns(0, 2.1), m.turns(1.9, 5.1), m.turns(3, 3)) == (0, 1, 2, 0)
    down = BoxMotion([0., .3], .1, .1, -1)
    assert down.position(.5) == pytest.approx(.05) and down.velocity(.5) == -.1 and down.position(2.) == pytest.approx(.1)
    still = BoxMotion([0., .3], 0., .1, 1)
    assert still.position(9.) == pytest.approx(.1) and still.velocity(9.) == 0. and still.turns(0, 9) == 0


def test_a_driven_box_is_released_on_the_first_hand_contact_and_never_driven_again(tmp_path):
    layout = {'speed_mps': .1, 'segment_y': [-4.9, -4.6], 'direction': 1}
    loop = GraspLoop(tmp_path, CONFIG, stop=lambda: None, layout=layout)
    w = waypoints(BOX, quat(0), [-.84, -4.6], CFG)
    def sample(t, box_y, root_x=-.84):
        return {'time': t, 'wrist_pos': (w['pre'] - [.05, 0, 0]).tolist(), 'wrist_rot': w['rotation'].tolist(),
                'root_pos': [root_x, -4.75, .78], 'wrist_velocity': [[0.]*3]*2,
                'box_com': [BOX[0], box_y, .85], 'box_quat': quat(0)}
    assert loop.drive(0.) is None                                     # nothing before the first latch
    info = {'round': 0, 'phase': 'raise', 'latch_time': 1., 'latched_box_pos': [BOX[0], -4.7, .85],
            'hands': 'open', 'event': None}
    loop.on_plan(info, sample(1., -4.7))
    y, vy = loop.drive(2.)
    assert y == pytest.approx(-4.6) and vy == .1
    assert loop.drive(2.5)[1] == -.1                                  # turned at the far end
    loop.after_step(sample(1.02, -4.698), RESTING)
    loop.after_step(sample(1.04, -4.696, root_x=-.90), RESTING)       # the pelvis backs off 6 cm
    event = {'id': 'round-0', 'positions': w['pre'].tolist(), 'rotations': w['rotation'].tolist(), 'deadline': 9.}
    loop.on_plan({**info, 'phase': 'advance', 'hands': 'gate', 'event': event}, sample(1.06, -4.694))
    loop.after_step(sample(1.06, -4.694, root_x=-.88), RESTING)
    touch = dict(RESTING, left_n=.3)
    loop.after_step(sample(1.08, -4.692, root_x=-.88), touch)
    assert loop.drive_released_at == pytest.approx(1.08) and loop.drive(1.1) is None
    result = json.loads((tmp_path/'grasp-result.json').read_text())
    notes = result['attempts'][0]['notes']
    assert notes['backoff_m'] == pytest.approx(.06) and notes['anchor_clearance_m'] == pytest.approx(CFG['table_edge_x_m'] + .84)
    np.testing.assert_allclose(notes['wrist_gap_m'], [.05, .05])
    assert result['drive_released_round'] == 0 and result['layout'] == layout
    loop.on_plan({**info, 'round': 1, 'phase': 'wait', 'latch_time': 3.}, sample(3., -4.69))
    assert loop.drive(3.5) is None                                    # a new round does not restart the drive


def test_hands_closing_records_where_the_box_went_since_the_latch(tmp_path):
    layout = {'speed_mps': .3, 'segment_y': [-4.9, -4.6], 'direction': 1}
    loop = GraspLoop(tmp_path, CONFIG, stop=lambda: None, layout=layout)
    loop.gate.closed = True
    base = {'wrist_pos': [[0.]*3]*2, 'wrist_rot': [EYE.tolist()]*2, 'root_pos': [-.84, -4.75, .78],
            'wrist_velocity': [[0.]*3]*2, 'box_quat': quat(0)}
    loop.on_plan({'round': 0, 'phase': 'grip', 'latch_time': 0., 'latched_box_pos': [BOX[0], -4.8, .85],
                  'hands': 'gate', 'event': None}, {**base, 'time': 0., 'box_com': [BOX[0], -4.8, .85]})
    loop.after_step({**base, 'time': 2., 'box_com': [BOX[0], -4.7, .85]}, RESTING)
    n = loop.attempts[0].notes
    assert n['latch_to_close_m'] == pytest.approx(.1) and n['close_time'] == 2. and n['turned']


def test_stand_still_reaches_from_where_it_stands(tmp_path):
    policy = scripted_policy(tmp_path, delay=5)
    policy.stand_still = True
    aside = standing_sample(policy, 0., BOX, root_xy=stance(BOX, CFG) + [-.05, -.15])
    policy.begin_episode(aside)
    for step in range(1, 7):
        policy.advance([standing_sample(policy, step * .02, BOX, root_xy=stance(BOX, CFG) + [-.05, -.15])])
    assert policy.round.phase == 'raise' and policy.walk is None and policy.round.notes['stand_still']


def test_stand_still_run_flags():
    a = parse_args(['--run-id', 'x', '--grasp-loop', '--stand-still', '--box-at', '.16', '-.1', '40', '.05'])
    assert a.box_at == [.16, -.1, 40., .05] and a.box_speed == 0.
    b = parse_args(['--run-id', 'x', '--grasp-loop', '--stand-still', '--box-in-region', '--box-speed', '.1'])
    assert b.box_in_region and b.box_speed == .1
    for bad in (['--stand-still'], ['--box-in-region'], ['--box-in-region', '--box-speed', '.1'],
                ['--stand-still', '--box-at', '.1', '0', '0', '0', '--box-in-region'],
                ['--stand-still', '--box-in-region', '--start-at-stance'],
                ['--stand-still', '--box-in-region', '--box-speed', '-.1']):
        with pytest.raises(ValueError):
            parse_args(['--run-id', 'x', '--grasp-loop', *bad])


def test_region_conditions_have_the_progress_page_names():
    pairs = [(0., 75), (.03, 75), (.1, 75), (0., 194), (.03, 194), (.1, 194), (0., 383), (.03, 383), (.1, 383)]
    assert [region_prefix(v, d, CONFIG) for v, d in pairs] == [
        'exp1-region', 'exp1-move003', 'exp1-move010', 'exp1-long-static', 'exp1-long-move003', 'exp1-long-move010',
        'exp1-xlong-static', 'exp1-xlong-move003', 'exp1-xlong-move010']
    for v, d in [(.3, 75), (.1, 100)]:
        with pytest.raises(SystemExit):
            region_prefix(v, d, CONFIG)
    assert not set(CONFIG['seeds']['region']) & (set(CONFIG['seeds']['debug']) | set(CONFIG['seeds']['heldout']))


def test_a_driven_box_away_from_its_latch_is_not_knocked_over_until_it_is_pushed(tmp_path):
    loop = GraspLoop(tmp_path, CONFIG, stop=lambda: None, layout={'speed_mps': .3, 'segment_y': [-5., -4.5], 'direction': 1})
    base = {'wrist_pos': [[0.]*3]*2, 'wrist_rot': [EYE.tolist()]*2, 'root_pos': [-.84, -4.75, .78],
            'wrist_velocity': [[0.]*3]*2, 'box_quat': quat(0)}
    at = lambda t, y: {**base, 'time': t, 'box_com': [BOX[0], y, .85]}
    loop.on_plan({'round': 0, 'phase': 'wait', 'latch_time': 0., 'latched_box_pos': [BOX[0], -4.9, .85],
                  'hands': 'open', 'event': None}, at(0., -4.9))
    loop.after_step(at(1., -4.6), RESTING)                            # 30 cm from the latch, carried by the drive
    assert loop.attempts[0].label == 'no_contact'
    loop.after_step(at(1.02, -4.6), dict(RESTING, right_n=.2))        # a hand touches: the drive lets go here
    loop.after_step(at(1.5, -4.55), RESTING)
    assert loop.attempts[0].label == 'touched_not_held'
    loop.after_step(at(2., -4.45), RESTING)                           # now pushed 15 cm by the hand
    assert loop.attempts[0].label == 'knocked_over'


def test_region_rule_takes_the_largest_rectangle_at_ninety_percent():
    def episode(d, l, y, e, ok):
        return {'success': ok, 'layout': {'depth_m': d, 'lateral_m': l, 'yaw_deg': y, 'stance_extra_m': e}}
    m = CONFIG['stand']['map']
    # near stance: everything at depth <= 0.16 and |lateral| <= 0.10 works, one 40-degree miss inside
    near = [episode(d, l, y, 0., d <= .16 and abs(l) <= .1 and (d, l, y) != (.12, .1, 40))
            for d in m['depth_m'] for l in m['lateral_m'] for y in m['yaw_deg']]
    # far stance: only the centre column works
    far = [episode(d, l, y, .05, l == 0.) for d in m['depth_m'] for l in m['lateral_m'] for y in m['yaw_deg']]
    pick = pick_region(near + far, CONFIG)
    assert pick == {'stance_extra_m': 0., 'region': {'depth_m': [.08, .16], 'lateral_m': [-.1, .1]},
                    'episodes': 18, 'successes': 17}
    assert pick_region(far, CONFIG)['region'] == {'depth_m': [.08, .24], 'lateral_m': [0., 0.]}
    assert pick_region([], CONFIG) is None


def test_rounds_are_binned_by_how_far_the_box_moved_before_the_hands_got_there():
    def attempt(label, dy, closed=True):
        n = {'reach_box_pos': [0., dy, .85]}
        if closed:
            n['close_box_pos'] = [0., dy + .001, .85]
        return {'attempt': 0, 'label': label, 'latched_box_pos': [0., 0., .85], 'notes': n}
    results = [{'run': 'a', 'attempts': [attempt('success', .004), attempt('touched_not_held', .06, closed=False)]},
               {'run': 'b', 'attempts': [attempt('success', -.015), {'attempt': 1, 'label': 'no_contact',
                                                                     'latched_box_pos': [0, 0, .85], 'notes': {}}]}]
    rows, table = displacement(results)
    assert len(rows) == 3                                             # a round that never reached is left out
    lateral = {b['from_cm']: (b['rounds'], b['successes']) for b in table['lateral_cm']}
    assert lateral[0] == (1, 1) and lateral[1] == (1, 1) and lateral[5] == (1, 0)
