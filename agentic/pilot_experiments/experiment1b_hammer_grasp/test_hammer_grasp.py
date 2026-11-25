"""Geometry, gate, judge and launcher switches of the one-handed hammer grasp; no weights or simulator."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from arena.grasp_judge import Criteria
from arena.hammer_layout import (LinearMotion, axes, cell_name, displaced, footprint, hammer_pose, heading_matrix,
                                 map_cells, place, placeable, supported, yaw_quat)
from arena.hammer_loop import HammerAttempt, HammerGraspLoop, HammerJudge, yaw_of_quat
from kimodo_vla.oracle.hammer_policy import grasp_pose, grasp_rotation, handle_frame, handle_yaw_deg
from run_hammer_suite import jobs_for
from run_oracle import parse_args

CONFIG = json.loads((Path(__file__).resolve().parents[2]/'configs/hammer-grasp.json').read_text())
# The real scene numbers are filled in from the scene record; the geometry tests use their own.
SCENE = {**CONFIG['scene'], 'table_edge_point_m': [-3.0, -2.9], 'table_top_z_m': .78, 'hammer_rest_z_m': .794}
TEST = {**CONFIG, 'scene': SCENE, 'policy': {**CONFIG['policy'], 'table_edge_point_m': SCENE['table_edge_point_m']}}
HEADING = heading_matrix(SCENE['facing_deg'])
HELD = dict(left_n=0., right_n=1., robot_other_n=0., external_n=0.)
RESTING = dict(left_n=0., right_n=0., robot_other_n=0., external_n=3.4)


def test_robot_facing_minus_y_has_its_left_along_plus_x():
    forward, left = axes(-90)
    assert forward == pytest.approx([0, -1]) and left == pytest.approx([1, 0])
    assert HEADING[:, 0] == pytest.approx([0, -1, 0]) and HEADING[:, 1] == pytest.approx([1, 0, 0])


def test_grasp_rotation_is_a_rotation_with_the_curl_axis_along_the_handle():
    axis = np.array([np.cos(.3), np.sin(.3), 0.])
    r = grasp_rotation(axis, 40)
    assert r.T @ r == pytest.approx(np.eye(3), abs=1e-9) and np.linalg.det(r) == pytest.approx(1)
    assert r[:, 2] == pytest.approx(axis)
    assert np.degrees(np.arcsin(-r[2, 0])) == pytest.approx(40)       # fingers 40 degrees below horizontal
    assert r[2, 1] < 0                                                # palm faces down
    level = grasp_rotation([1., 0, 0], 0)                             # handle along the robot's left (+x)
    assert level[:, 0] == pytest.approx([0, -1, 0]) and level[:, 1] == pytest.approx([0, 0, -1])


def test_the_grip_point_sits_at_its_place_in_the_hand():
    root, quat = [-3.1, -3.05, .794], yaw_quat(np.radians(-80))
    w = grasp_pose(root, quat, HEADING, TEST['policy'])
    p = TEST['policy']
    in_hand = w['rotation'] @ [p['grip_in_hand_m'][0], p['grip_in_hand_m'][1], 0]
    assert w['grip'] + in_hand - [0, 0, p['grip_height_m']] == pytest.approx(w['grip_point'])
    assert w['pre'] - w['grip'] == pytest.approx([0, 0, p['approach_up_m']])
    assert w['lift'] - w['grip'] == pytest.approx([0, 0, p['lift_m']])
    # the wrist is behind (robot side of) and above the handle
    assert (w['grip'] - w['grip_point']) @ HEADING[:, 0] < 0 and w['grip'][2] > w['grip_point'][2] + .1


def test_a_hammer_whose_head_points_to_the_robots_right_is_refused():
    with pytest.raises(ValueError):
        grasp_pose([-3.1, -3.05, .794], yaw_quat(np.radians(90)), HEADING, TEST['policy'])


@pytest.mark.parametrize('yaw', [-20, 0, 20, 45])
def test_layout_puts_the_grip_point_where_asked_with_the_asked_handle_yaw(yaw):
    lay = place(.13, -.2, yaw, .05, TEST)
    point, axis = handle_frame(lay['hammer_pos'], yaw_quat(lay['hammer_yaw_rad']), TEST['policy'])
    assert point[:2] == pytest.approx(lay['grip_xy'])
    assert handle_yaw_deg(axis, HEADING) == pytest.approx(yaw)
    forward, left = axes(-90)
    edge = np.asarray(SCENE['table_edge_point_m'])
    assert (point[:2] - edge) @ forward == pytest.approx(.13)
    assert (point[:2] - np.asarray(lay['robot_xy'])) @ left == pytest.approx(-.2)
    # reach rule: wrist 0.25 ahead of the root unless that brings the root nearer than 0.19 to the edge; then 0.05 back
    assert lay['clearance_m'] >= .19 + .05 - 1e-9
    assert (lay['clearance_m'] == pytest.approx(.24) and lay['wrist_ahead_of_root_m'] >= .30 - 1e-9
            or lay['wrist_ahead_of_root_m'] == pytest.approx(.30))


def test_positive_yaw_turns_the_head_towards_the_robot():
    lay = place(.13, -.2, 20, 0, TEST)
    _, axis = handle_frame(lay['hammer_pos'], yaw_quat(lay['hammer_yaw_rad']), TEST['policy'])
    assert axis[:2] @ axes(-90)[0] < 0


def test_default_task_orientation_is_the_unturned_handle():
    # Arena's hammer rests with yaw -89 degrees: head towards +x, the robot's left when it faces -y.
    _, axis = handle_frame([0, 0, 0], [0.7139, 0, 0, -0.7002], TEST['policy'])
    assert handle_yaw_deg(axis, HEADING) == pytest.approx(0, abs=2)


def test_support_check_uses_the_whole_hammer():
    lay = place(.13, -.2, 0, 0, TEST)
    table = [[-4.0, -3.4], [-1.5, -2.9]]
    assert supported(lay['hammer_pos'], lay['hammer_yaw_rad'], TEST, table)
    near = place(.03, -.2, 0, 0, TEST)         # the hammer is 12 cm wide across: 3 cm leaves it over the edge
    assert not supported(near['hammer_pos'], near['hammer_yaw_rad'], TEST, table)
    assert len(footprint(lay['hammer_pos'], 0., TEST)) == 8


def test_map_cells_whose_hammer_overhangs_the_edge_are_not_placeable():
    cfg = {**TEST, 'scene': {**SCENE, 'table_rect_m': [[-4.0, -3.4], [-1.5, -2.9]]}}
    assert placeable(dict(depth=.13, lateral=-.2, yaw=0, extra=0), cfg)
    assert not placeable(dict(depth=.05, lateral=-.2, yaw=20, extra=0), cfg)   # the head swings over the edge


def test_injected_displacements_are_along_across_and_about_the_grip_point():
    lay = place(.13, -.2, 0, 0, TEST)
    root, yaw = lay['hammer_pos'], lay['hammer_yaw_rad']
    grip, axis = handle_frame(root, yaw_quat(yaw), TEST['policy'])
    moved, y = displaced(root, yaw, grip, axis, along=.05)
    assert np.subtract(moved, root)[:2] == pytest.approx(.05 * axis[:2]) and y == yaw
    moved, _ = displaced(root, yaw, grip, axis, across=.04)
    assert np.subtract(moved, root)[:2] == pytest.approx([0, -.04])   # away from the robot (it faces -y)
    moved, y = displaced(root, yaw, grip, axis, turn_deg=30)
    new_grip, _ = handle_frame(moved, yaw_quat(y), TEST['policy'])
    assert new_grip == pytest.approx(grip) and y == pytest.approx(yaw + np.radians(30))


def test_linear_motion_stops_at_its_travel_limit():
    m = LinearMotion([0, 0], [1, 0], .1, .3)
    assert m.position(2) == pytest.approx([.2, 0]) and m.velocity(2) == pytest.approx([.1, 0])
    assert m.position(10) == pytest.approx([.3, 0]) and m.velocity(10) == pytest.approx([0, 0])


def test_one_hand_is_enough_and_both_hands_are_not_required():
    judge, c = HammerJudge(.794, Criteria()), Criteria()
    for t in np.arange(0, .6, .02):
        judge.update(t, .794 + .12, HELD)
    assert judge.success_time == pytest.approx(.5)
    a = HammerAttempt(0, 0., (-3.1, -3.05, .794), .794, c)
    for t in np.arange(0, .3, .02):
        a.update(t, (-3.1, -3.05, .81), 1., HELD, True)
    assert a.label == 'held_at_cutoff' and 'left_contact' not in a.events
    b = HammerAttempt(0, 0., (-3.1, -3.05, .794), .794, c)
    b.update(0., (-3.1, -3.05, .794), 1., RESTING, True)
    b.update(.02, (-3.1, -3.25, .794), 1., RESTING, True)     # pushed 20 cm without being held
    assert b.label == 'knocked_over'


def test_the_loop_closes_only_the_right_hand_on_its_own_arrival(tmp_path):
    loop = HammerGraspLoop(tmp_path, TEST, stop=lambda: None)
    target, rot = [-3.1, -2.95, .92], np.eye(3)
    info = {'round': 0, 'phase': 'advance', 'latch_time': 0., 'latched_box_pos': [-3.1, -3.05, .794],
            'latched_box_quat': [1, 0, 0, 0], 'graspable': True, 'hands': 'gate',
            'event': {'id': 'round-0', 'positions': [[9, 9, 9], target], 'rotations': [rot.tolist(), rot.tolist()],
                      'deadline': 5.}}
    sample = lambda t, right: {'time': t, 'wrist_pos': [[9, 9, 9], right], 'wrist_rot': [rot, rot],
                               'wrist_velocity': [[0, 0, 0], [0, 0, 0]], 'box_com': [-3.1, -3.05, .794],
                               'box_pos': [-3.1, -3.05, .794], 'box_quat': [1, 0, 0, 0], 'root_pos': [-3.1, -2.6, .78]}
    loop.on_plan(info, sample(0., target))
    assert loop.hands() == (False, False)
    for t in np.arange(0, .2, .02):
        loop.before_step(sample(t, target))
    assert loop.hands() == (False, True) and loop.gate.fired_at == pytest.approx(.1)
    info.update(hands='open', event=None, round=1)
    loop.on_plan(info, sample(.3, target))
    assert loop.hands() == (False, False) and loop.gate.fired_at is None


def test_yaw_of_quat():
    x, y, z, w = Rotation.from_euler('z', 33, degrees=True).as_quat()
    assert np.degrees(yaw_of_quat([w, x, y, z])) == pytest.approx(33)


def test_map_is_within_the_episode_budget_and_named_uniquely():
    cells = map_cells(CONFIG)
    assert len(cells) <= 100 and len({cell_name(c) for c in cells}) == len(cells)


def test_suite_names_and_arguments():
    class A:
        group, tag, per_gpu, seeds, auto_gpus = 'map', 't', 2, None, None
    jobs = jobs_for(A, CONFIG)
    assert jobs[0][0].startswith('exp1b-map-t-') and '--hammer-at' in jobs[0][2] and '--gpu-min-free-mib' in jobs[0][2]
    A.group = 'region'
    assert [j[1] for j in jobs_for(A, CONFIG)] == CONFIG['seeds']['region']


def test_launcher_task_switch_keeps_the_box_default():
    a = parse_args(['--run-id', 'x', '--grasp-loop'])
    assert a.task == 'pp_box' and a.start_timeout_s is None
    a = parse_args(['--run-id', 'x', '--grasp-loop', '--task', 'doubledesk', '--stand-still', '--hammer-at', '.1', '-.2', '0', '0'])
    assert a.task == 'doubledesk' and a.steps == 1450
    for bad in (['--grasp-loop', '--task', 'doubledesk', '--stand-still'],                       # no placement
                ['--grasp-loop', '--task', 'doubledesk', '--hammer-in-region'],                  # walking
                ['--grasp-loop', '--stand-still', '--hammer-in-region'],                         # box task
                ['--grasp-loop', '--task', 'doubledesk', '--stand-still', '--box-in-region'],
                ['--grasp-loop', '--task', 'doubledesk', '--stand-still', '--hammer-at', '.1', '0', '0', '0',
                 '--hammer-speed', '.1'],                                                        # direction needed
                ['--grasp-loop', '--task', 'doubledesk', '--stand-still', '--hammer-in-region',
                 '--hammer-speed', '.1', '--inject', '.01', '0', '0']):
        with pytest.raises(ValueError):
            parse_args(['--run-id', 'x', *bad])


def test_a_partial_closing_amount_stops_between_open_and_fist(tmp_path):
    class Provider:
        _right_hand_target = np.zeros(7)
        def _get_hand_pose_from_binary(self, side, closed):
            return np.full(7, 2.) if closed else np.zeros(7)
    cfg = {**TEST, 'policy': {**TEST['policy'], 'close_amount': .6}}
    loop = HammerGraspLoop(tmp_path, cfg, stop=lambda: None, provider=Provider())
    calls = []
    def apply(left, right):
        calls.append((left, right))
        loop.provider._right_hand_target[:] = loop.provider._get_hand_pose_from_binary('right', right)
    loop.command_hands(apply)
    assert calls == [(False, False)] and loop.provider._right_hand_target == pytest.approx(np.zeros(7))
    loop.gate.closed = True
    loop.command_hands(apply)
    assert calls[-1] == (False, True) and loop.provider._right_hand_target == pytest.approx(np.full(7, 1.2))


def test_tolerance_is_where_the_rate_falls_to_half_of_the_zero_level():
    from analyze_hammer import half_point, predicted
    pts = [(-10., 20, 0), (-5., 20, 10), (0., 20, 18), (5., 20, 16), (10., 20, 12)]
    assert half_point(pts, -1) == pytest.approx(-5 - 5 * (.5 - .45) / .5)    # half of 90% is reached between -5 and -10
    assert half_point(pts, +1) is None                      # still above half at the largest tested level
    curves = {'along': pts, 'across': [(-2., 10, 5), (0., 10, 10), (2., 10, 10)], 'turn': [(0., 10, 10), (20., 10, 5)]}
    assert predicted(curves, 0., 0., 0.) == pytest.approx(.9)
    assert predicted(curves, -5., -1., 10.) == pytest.approx(.5 * .75 * .75)


def test_failure_sides_follow_the_box_definition():
    from analyze_hammer import failure_class
    assert failure_class({'label': 'no_contact', 'events': {}}) == 'body: no contact'
    assert failure_class({'label': 'knocked_over', 'events': {}}) == 'body: knocked before closing'
    assert failure_class({'label': 'knocked_over', 'events': {'closed': 1.}}) == 'object: knocked after closing'
    assert failure_class({'label': 'touched_not_held', 'events': {'closed': 1.}}) == 'object: closed, not held'


def test_step_two_groups_are_paired_on_the_first_twenty_region_seeds():
    class A:
        group, tag, per_gpu, seeds, auto_gpus = 'inject', 't', 1, None, 2
    jobs = jobs_for(A, CONFIG)
    seeds = CONFIG['seeds']['region'][:20]
    assert len(jobs) == 3 * 4 * 20 and sorted({j[1] for j in jobs}) == seeds
    assert any(j[0].startswith('exp1b-inj-along-10-t-') and j[2][2:5] == ['-0.1', '0', '0'] for j in jobs)
    assert any(j[0].startswith('exp1b-inj-turn+40-t-') and j[2][2:5] == ['0', '0', '40'] for j in jobs)
    A.group = 'motion'
    jobs = jobs_for(A, CONFIG)
    assert len(jobs) == 2 * 2 * 20 and {j[0].split('-t-')[0] for j in jobs} == {
        'exp1b-move003-d75', 'exp1b-move003-d194', 'exp1b-move010-d75', 'exp1b-move010-d194'}
