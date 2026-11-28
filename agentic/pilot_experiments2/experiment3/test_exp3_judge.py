"""The five-link judge on hand-made episodes: one per link for each task, plus success and unclassified.

Episodes are written to disk in the log format and read back with ``load_episode``, so the
format and the judge are tested together. Every threshold comes from exp3_thresholds.json.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from kimodo_vla.agent.episode_log import load_episode
from kimodo_vla.agent.judge import box_reach_region, judge, rectangle_margin

HERE = Path(__file__).resolve().parent
THRESHOLDS = json.loads((HERE/'exp3_thresholds.json').read_text())
GRASP = json.loads((HERE.parents[1]/'configs'/'grasp-loop.json').read_text())
HALF = .105
BOX = [.5, 0., .85]


def state(root, yaw=0., wrists=None):
    eye = np.eye(3).tolist()
    w = wrists or {'left': [root[0]+.2, root[1]+.2, root[2]], 'right': [root[0]+.2, root[1]-.2, root[2]]}
    return {'root': list(root), 'yaw_deg': yaw, 'pelvis_height_m': root[2], 'torso_pitch_deg': 0.,
            'wrists': {s: {'position': p, 'rotation': eye} for s, p in w.items()},
            'shoulders': {'left': [root[0], root[1]+.2, root[2]+.3], 'right': [root[0], root[1]-.2, root[2]+.3]}}


def command(index, kind, time, end, body, target=None, start=None, finish=None, error=None):
    start = start or state([0., 0., .78])
    return [{'kind': 'start', 'index': index, 'type': kind, 'time': time, 'command': {'type': kind, **body},
             'target': target or {}, 'measured': start, 'reference': start},
            {'kind': 'end', 'index': index, 'type': kind, 'time': end, 'status': 'completed', 'measured': finish or start,
             'reference': finish or start, 'measured_error': error or {}}]


def write(run, task, scene, commands, steps, success, end_reason='commander_done'):
    loop = run/'loop'
    loop.mkdir(parents=True)
    (loop/'meta.json').write_text(json.dumps({'task': task}))
    (loop/'scene.json').write_text(json.dumps({'task': task, **scene}))
    (loop/'commands.jsonl').write_text(''.join(json.dumps(r)+'\n' for c in commands for r in c))
    (loop/'steps.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in steps))
    (loop/'summary.json').write_text(json.dumps({'end_reason': end_reason}))
    (run/'episode.json').write_text(json.dumps({'success': success, 'failure_reason': 'success' if success else 'max_steps'}))
    return load_episode(run)


# ---- P&PBox -----------------------------------------------------------------------------
def box_episode(**change):
    """A complete pick-and-place that passes every check; ``change`` breaks one thing."""
    forward, lateral = box_reach_region(GRASP)
    pelvis = [BOX[0]-np.mean(forward), BOX[1]-np.mean(lateral), .78]
    left = [BOX[0]-.1, BOX[1]+HALF+.05, BOX[2]+.04]
    right = [BOX[0]-.1, BOX[1]-HALF-.05, BOX[2]+.04]
    eye = np.eye(3).tolist()
    target = {'wrists': {'left': {'position': change.get('left', left), 'rotation': eye},
                         'right': {'position': right, 'rotation': eye}}}
    at = state(pelvis)
    reach_error = change.get('reach_error', {'left_wrist_m': .02, 'left_wrist_deg': 5., 'right_wrist_m': .02, 'right_wrist_deg': 5.})
    layer = {'name': 'shelf_target_layer', 'kind': 'region', 'min': [1.5, -.3, .65], 'max': [2.4, .3, .65]}
    # Carrying: the wrists 10 cm behind and 4 cm above the box centre, as the steps below have them.
    held = state([1.4, 0., .78], wrists={'left': [1.85, HALF+.04, .99], 'right': [1.85, -HALF-.04, .99]})
    place = {'wrists': {'left': {'position': [1.85, HALF+.04, .65+HALF+.04], 'rotation': eye},
                        'right': {'position': [1.85, -HALF-.04, .65+HALF+.04], 'rotation': eye}}}
    commands = [command(0, 'walk', 0.6, 4., {'forward_m': .5}, {'root_xy': pelvis[:2]}, error={'root_xy_m': change.get('walk_error', .03)}),
                command(1, 'hand_pose', 4., 6., {'hands': {}}, target, at, at, reach_error),
                command(2, 'hand', 6., 6.6, {'left': 'close', 'right': 'close'}, {}, at, at),
                command(3, 'hand_pose', 6.6, 8., {'hands': {}}, {}, at, at),
                command(4, 'walk', 8., 12., {'forward_m': 1.}, {}, at, held, {'root_xy_m': change.get('carry_error', .04)}),
                command(5, 'hand_pose', 12., 14., {'hands': {}}, place, held, held,
                        {'left_wrist_m': .02, 'right_wrist_m': .02}),
                command(6, 'hand', 14., 14.6, {'left': 'open', 'right': 'open'}, {}, held, held)]
    lift = change.get('lift', lambda t: 0. if t < 7. else .1)
    steps = []
    hands_up = change.get('hands_up', lift)
    for t in np.arange(0., 20.5, .1):
        t = round(float(t), 2)
        carried = t >= 8. and hands_up(t) > 0.
        x = 1.95 if carried or (t >= 14. and change.get('placed', True)) else BOX[0]
        z, hz = BOX[2]+lift(t), BOX[2]+hands_up(t)
        if t >= 13. and change.get('placed', True):
            z = hz = .65+HALF
        s = {'time': t, 'fallen': bool(change.get('fall_at') is not None and t >= change['fall_at']), 'tilt_deg': 2.,
             'objects': [{'name': 'box', 'position': [x, 0., z], 'yaw_deg': 0.}],
             'wrist_pos': [[x-.1, HALF+.04, hz+.04], [x-.1, -HALF-.04, hz+.04]],
             'other_contacts_n': change.get('contacts', {}) if change.get('contact_window', (4., 6.))[0] <= t <= change.get('contact_window', (4., 6.))[1] else {},
             'arena_success': bool(change.get('stays', True) and change.get('placed', True) and t >= 14.6),
             # the hands press the box while closed (6.6-14 s), and after opening only if they stay on it
             'box_contact': {'left_n': 8. if 6.6 <= t < 14. or change.get('hands_on') else 0.,
                             'right_n': 8. if 6.6 <= t < 14. else 0.}}
        steps.append(s)
    scene = {'objects': [{'name': 'box', 'kind': 'object', 'position': BOX, 'yaw_deg': 0., 'size': [.21]*3}, layer]}
    return commands, steps, scene


def verdict_box(tmp_path, success=False, **change):
    commands, steps, scene = box_episode(**change)
    return judge(write(tmp_path, 'pp_box', scene, commands, steps, success), THRESHOLDS, grasp_config=GRASP)


def test_box_success(tmp_path):
    """P&PBox success is the 5 s reading (2026-11-28), not Arena's flag: kept on the target 5 s after the hands
    open, with both hands off it."""
    v = verdict_box(tmp_path, success=False)
    assert v['success'] and v['link'] is None and v['arena_success'] is False
    assert v['success_rule'] == 'released_and_kept' and v['openings'] == [{'time': 14., 'kept': True}]


def test_box_arena_success_without_release_is_not_success(tmp_path):
    """Arena saw the box over the shelf once (it fell past it), the hands never opened: not a success, link 5."""
    v = verdict_box(tmp_path, success=True, lift=lambda t: .1 if 7. <= t < 9. else 0., hands_up=lambda t: 0. if t < 7. else .1,
                    placed=False)
    assert v['arena_success'] is True and v['success'] is False
    assert (v['link'], v['reason']) == (5, 'kept the box until the hands opened')


def test_box_hands_still_on_the_box_is_not_kept(tmp_path):
    v = verdict_box(tmp_path, hands_on=True)
    assert v['success'] is False and (v['link'], v['reason']) == (4, 'place: box stayed on the target after release')


def test_box_log_ending_before_the_hold_is_not_kept(tmp_path):
    commands, steps, scene = box_episode()
    v = judge(write(tmp_path, 'pp_box', scene, commands, [s for s in steps if s['time'] < 18.], True), THRESHOLDS, grasp_config=GRASP)
    assert v['success'] is False and v['openings'] == [{'time': 14., 'kept': None}] and v['link'] == 4


def test_box_all_checks_pass_on_the_reference_episode(tmp_path):
    v = verdict_box(tmp_path)
    assert v['success'] and v['link'] is None
    assert all(c['passed'] for c in v['checks'])
    assert {c['phase'] for c in v['checks']} == {'pick', 'place'}
    assert v['margins']['box_region_m'] > 0
    assert v['arrived'] == {'pick:1': True, 'pick:2': True, 'pick:3': True, 'pick:4': True, 'pick:5': True,
                            'place:2': True, 'place:3': True, 'place:4': True}


NO_LIFT = {'lift': lambda t: 0., 'placed': False}   # a reach that went wrong does not lift the box afterwards


@pytest.mark.parametrize('change, link, check', [
    ({'contacts': {'torso_link': 80.}, **NO_LIFT}, 1, 'body collision during the reach'),
    ({'fall_at': 5., **NO_LIFT}, 1, 'stable during the reach'),
    ({'left': [BOX[0]-.1, BOX[1]+HALF+.35, BOX[2]+.04], **NO_LIFT}, 2, 'left wrist beside its face'),
    ({'reach_error': {'left_wrist_m': .15, 'left_wrist_deg': 5., 'right_wrist_m': .02, 'right_wrist_deg': 5.}, **NO_LIFT},
     3, 'left wrist arrived'),
    ({'lift': lambda t: 0., 'placed': False}, 4, 'box lifted after closing'),
    ({'walk_error': .2, **NO_LIFT}, 3, 'pick: root arrived (last walk)'),
    ({'lift': lambda t: .1 if 7. <= t < 9. else 0., 'hands_up': lambda t: 0. if t < 7. else .1, 'placed': False},
     5, 'kept the box until the hands opened'),
    ({'carry_error': .3, 'stays': False}, 4, 'place: box stayed on the target after release'),
    ({'stays': False}, 4, 'place: box stayed on the target after release')])
def test_box_one_episode_per_link(tmp_path, change, link, check):
    v = verdict_box(tmp_path, **change)
    assert (v['link'], v['reason']) == (link, check)
    assert v['failed_check']['passed'] is False


def test_box_out_of_reach_is_link_one(tmp_path):
    commands, steps, scene = box_episode(**NO_LIFT)
    far = copy.deepcopy(commands)
    far[1][0]['measured']['root'][0] -= .4          # the robot stood 40 cm further back when it reached
    v = judge(write(tmp_path, 'pp_box', scene, far, steps, False), THRESHOLDS, grasp_config=GRASP)
    assert (v['link'], v['reason']) == (1, 'box inside the validated reach region')
    assert v['margins']['box_region_m'] < 0


def test_box_without_an_attempt_is_unclassified(tmp_path):
    commands, steps, scene = box_episode()
    v = judge(write(tmp_path, 'pp_box', scene, commands[:1], steps, False, 'command_budget'), THRESHOLDS, grasp_config=GRASP)
    assert v['link'] == 'unclassified' and v['reason'].startswith('no attempt') and v['end_reason'] == 'command_budget'


def test_collision_on_the_way_without_an_attempt_is_link_one(tmp_path):
    commands, steps, scene = box_episode()
    for s in steps:
        if 2. <= s['time'] <= 3.:
            s['other_contacts_n'] = {'left_elbow_link': 90.}
    v = judge(write(tmp_path, 'pp_box', scene, commands[:1], steps, False), THRESHOLDS, grasp_config=GRASP)
    assert v['link'] == 1 and v['phase'] == 'approach' and 'left_elbow_link' in v['reason']


def test_a_brief_touch_is_not_a_collision(tmp_path):
    v = verdict_box(tmp_path, contacts={'torso_link': 80.}, success=False, contact_window=(4., 4.1))
    assert v['link'] != 1


def test_earliest_and_last_attempt_rules(tmp_path):
    commands, steps, scene = box_episode(lift=lambda t: 0., placed=False)
    # A second attempt after the first: its reach misses by 15 cm. Earliest: link 4 (first attempt). Last: link 3.
    at = commands[1][0]['measured']
    bad = {'left_wrist_m': .15, 'left_wrist_deg': 5., 'right_wrist_m': .02, 'right_wrist_deg': 5.}
    more = [command(7, 'hand', 15., 15.5, {'left': 'open', 'right': 'open'}, {}, at, at),
            command(8, 'hand_pose', 15.5, 17., {'hands': {}}, commands[1][0]['target'], at, at, bad),
            command(9, 'hand', 17., 17.6, {'left': 'close', 'right': 'close'}, {}, at, at)]
    episode = write(tmp_path, 'pp_box', scene, commands[:3]+more, steps, False)
    assert judge(episode, THRESHOLDS, grasp_config=GRASP)['link'] == 4
    assert judge(episode, THRESHOLDS, grasp_config=GRASP, attribution='last_attempt')['link'] == 3


# ---- SitSofa ----------------------------------------------------------------------------
SEAT = {'name': 'sofa_seat', 'kind': 'region', 'min': [-1., -1., 0.], 'max': [-.4, 1., .45]}


def sofa_episode(tmp_path, success=False, **change):
    pelvis = change.get('pelvis', [-.25, 0., .78])
    yaw = change.get('yaw', 0.)
    start = state(pelvis, yaw)
    finish = state(change.get('end_root', [-.44, 0., .56]), yaw)
    sit = {'height_m': .55, 'behind_feet_m': change.get('behind', .2)}
    commands = [command(0, 'turn', .6, 3., {'left_deg': 180}, {}, start, start),
                command(1, 'pelvis_height', 3., 5., sit, {'pelvis_height_m': .55}, start, finish)]
    contact = change.get('contact', lambda t: 200. if t >= 4.8 else 0.)
    steps = [{'time': round(float(t), 2), 'hip_contact_n': contact(t), 'tilt_deg': 3.,
              'fallen': bool(change.get('fall_at') is not None and t >= change['fall_at']), 'objects': []}
             for t in np.arange(0., 9., .1)]
    episode = write(tmp_path, 'sit_sofa', {'objects': [SEAT]}, commands, steps, success)
    return judge(episode, THRESHOLDS)


def test_sofa_success(tmp_path):
    v = sofa_episode(tmp_path, success=True)
    assert v['success'] and v['link'] is None


def test_sofa_all_checks_pass_on_the_reference_episode(tmp_path):
    v = sofa_episode(tmp_path)
    assert v['link'] == 'unclassified' and all(c['passed'] for c in v['checks'])
    assert v['margins']['pelvis_target_seat_m'] > 0


UNSEATED = {'contact': lambda t: 0.}    # a sit that went wrong does not end with the seat carrying the hips


@pytest.mark.parametrize('change, link, check', [
    ({'yaw': 180., **UNSEATED}, 1, 'back turned to the seat'),
    ({'pelvis': [.3, 0., .78], 'end_root': [.1, 0., .56], **UNSEATED}, 1, 'seat edge within reach behind the pelvis'),
    ({'behind': 0., **UNSEATED}, 2, 'pelvis target over the seat'),
    ({'end_root': [-.44, 0., .70], **UNSEATED}, 3, 'pelvis arrived (height)'),
    ({'contact': lambda t: 0.}, 4, 'seat carries the hips'),
    ({'pelvis': [2.0, 0., .78], 'yaw': 180., 'end_root': [2.2, 0., .5]}, 1, 'back turned to the seat'),   # sat on the floor
    ({'contact': lambda t: 200. if 4.8 <= t < 7. else 0., 'fall_at': 7.5}, 5, 'stayed seated')])
def test_sofa_one_episode_per_link(tmp_path, change, link, check):
    v = sofa_episode(tmp_path, **change)
    assert (v['link'], v['reason']) == (link, check)


# ---- margins and the thresholds file ----------------------------------------------------
def test_rectangle_margin_signs():
    assert rectangle_margin([0, 0], [-1, -2], [1, 2]) == 1.
    assert rectangle_margin([2, 0], [-1, -2], [1, 2]) == -1.
    assert np.isclose(rectangle_margin([2, 3], [-1, -2], [1, 2]), -np.sqrt(2))


def test_reach_region_matches_the_first_batch():
    forward, lateral = box_reach_region(GRASP)
    assert lateral == GRASP['stand']['region']['lateral_m']
    assert .3 < forward[0] < forward[1] < .55


def test_every_threshold_is_named_and_marked():
    used = set()
    import re
    import kimodo_vla.agent.judge as module
    used = set(re.findall(r"value\(t, '([a-z_]+)'\)", Path(module.__file__).read_text()))
    assert used <= set(THRESHOLDS)
    for name in used:
        entry = THRESHOLDS[name]
        assert (entry['status'] == 'fact' or entry['status'].startswith(('fixed 2026-11-28', 'decided 2026-11-28'))) and entry['meaning']
    assert THRESHOLDS['place_hold_s']['value'] == 5.0 and 'place_settle_s' not in THRESHOLDS


def test_a_lifted_box_supersedes_stance_and_arrival_checks(tmp_path):
    """The box was lifted, then lost: links 1-3 of that attempt are noted, the failure is link 5."""
    commands, steps, scene = box_episode(lift=lambda t: .1 if 7. <= t < 9. else 0., hands_up=lambda t: 0. if t < 7. else .1,
                                         placed=False, reach_error={'left_wrist_m': .15, 'left_wrist_deg': 5.,
                                                                    'right_wrist_m': .02, 'right_wrist_deg': 5.})
    commands[1][0]['measured']['root'][0] -= .4      # also outside the validated reach region
    v = judge(write(tmp_path, 'pp_box', scene, commands, steps, False), THRESHOLDS, grasp_config=GRASP)
    assert v['link'] == 5
    noted = [c for c in v['checks'] if c.get('superseded')]
    assert {c['link'] for c in noted} == {1, 3}


def test_unfinished_and_refused_sits(tmp_path):
    """A sit the episode ended in the middle of has no end record; a sit the planner refused is no attempt."""
    pelvis = [-.25, 0., .78]
    start = state(pelvis, 0.)
    sit = {'height_m': .55, 'behind_feet_m': .2}
    unfinished = command(0, 'pelvis_height', 1., 2., sit, {'pelvis_height_m': .55}, start, start)[:1]
    refused = command(1, 'pelvis_height', .5, .5, sit, {}, start, start)
    refused[1]['status'] = 'planner_refused'
    steps = [{'time': round(float(t), 2), 'hip_contact_n': 200. if t > 1.5 else 0., 'tilt_deg': 3., 'fallen': False,
              'objects': []} for t in np.arange(0., 4., .1)]
    v = judge(write(tmp_path, 'sit_sofa', {'objects': [SEAT]}, [refused, unfinished], steps, False), THRESHOLDS)
    assert v['attempts'] == 1 and v['link'] in (5, 'unclassified', 1, 2, 3, 4)


def quat_z(deg):
    a = np.radians(deg)/2
    return [float(np.cos(a)), 0., 0., float(np.sin(a))]


def test_turning_with_the_box_held_is_not_a_drop(tmp_path):
    """Link 5 measures the box's offset from the wrists in the body's heading frame (2026-11-28): a 90 degree turn
    in place with the box held 14 cm in front of the wrists is not a drop; the box falling out still is."""
    from kimodo_vla.agent.judge import held_offset
    def step(yaw, box_ahead=.14, drop=0.):
        a = np.radians(yaw)
        fwd, left = np.array([np.cos(a), np.sin(a), 0.]), np.array([-np.sin(a), np.cos(a), 0.])
        mid = np.array([0., 0., 1.])
        return {'root_quat': quat_z(yaw), 'wrist_pos': [(mid+.15*left).tolist(), (mid-.15*left).tolist()],
                'objects': [{'name': 'box', 'position': (mid+box_ahead*fwd-[0, 0, drop]).tolist()}]}
    held = held_offset(step(0.))
    assert np.linalg.norm(held_offset(step(90.))-held) < 1e-9          # turned with it: same offset
    world = lambda s: np.subtract(s['objects'][0]['position'], np.mean(s['wrist_pos'], axis=0))
    assert np.linalg.norm(world(step(90.))-world(step(0.))) > .19         # the world frame would have moved 0.2 m
    assert np.linalg.norm(held_offset(step(90., drop=.5))-held) > .4      # fell out: still a drop
    assert np.allclose(held_offset({k: v for k, v in step(30.).items() if k != 'root_quat'}), world(step(30.)))


def test_a_carrying_turn_in_an_episode_is_not_link_five(tmp_path):
    """The reference episode, but the robot turns 90 degrees while carrying (the box turns with the hands)."""
    commands, steps, scene = box_episode()
    for s in steps:
        yaw = 0. if s['time'] < 9. else min(90., (s['time']-9.)*45.) if s['time'] < 12. else 0.
        if 8. <= s['time'] < 12.:
            a = np.radians(yaw)
            mid = np.mean(s['wrist_pos'], axis=0)
            box = np.asarray(s['objects'][0]['position'])
            d = box-mid
            rot = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
            s['objects'][0]['position'] = (mid+rot @ d).tolist()
            s['root_quat'] = quat_z(yaw)
    v = judge(write(tmp_path, 'pp_box', scene, commands, steps, False), THRESHOLDS, grasp_config=GRASP)
    assert v['success'] and not any(c['link'] == 5 and not c['passed'] for c in v['checks'])


def test_the_place_stage_does_not_check_the_root(tmp_path):
    """2026-11-28 (user, departing from the plan's text): the walk before placing is not a link-3 check; the pick
    stage keeps its root check."""
    v = verdict_box(tmp_path/'a', carry_error=.3)
    assert v['success'] and not any(c.get('phase') == 'place' and 'root' in c['check'] for c in v['checks'])
    v = verdict_box(tmp_path/'b', walk_error=.2, **NO_LIFT)
    assert (v['link'], v['reason']) == (3, 'pick: root arrived (last walk)')
