"""The command set, the posture solver and the step loop, without Kimodo or the simulator."""
import json
from pathlib import Path

import numpy as np
import pytest

from kimodo_vla.agent.commands import CommandError, extract_json, palm_and_fingers, parse, validate, wrist_rotation
from kimodo_vla.agent.episode_log import load_episode
from kimodo_vla.agent.loop import CommandLoopPolicy
from kimodo_vla.agent.planner import Planner, hold_rows
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.model.features import G1_XML_PATH
from kimodo_vla.oracle.geometry import XmlFK

HERE = Path(__file__).resolve().parent
FK = XmlFK(G1_XML_PATH, G1_JOINT_NAMES)


def standing():
    q = np.zeros(29)
    names = list(G1_JOINT_NAMES)
    for side in ('left', 'right'):
        q[names.index(f'{side}_hip_pitch_joint')], q[names.index(f'{side}_knee_joint')] = -.1, .3
        q[names.index(f'{side}_ankle_pitch_joint')] = -.2
    root = np.array([0., 0., .784])
    return q, root


def sample(t, q, root):
    wp, wr = FK(q, root, np.eye(3))
    feet = FK(q, root, np.eye(3), body_names=['left_ankle_roll_link', 'right_ankle_roll_link'])
    return {'time': t, 'root_pos': root.tolist(), 'root_quat': [1, 0, 0, 0], 'joints': q.tolist(),
            'reference_root_pos': root.tolist(), 'reference_root_quat': [1, 0, 0, 0], 'reference_joints': q.tolist(),
            'wrist_pos': wp.tolist(), 'wrist_rot': wr.tolist(), 'foot_pos': feet[0].tolist(), 'foot_rot': feet[1].tolist(),
            'gripper_command': [0, 0]}


# ---- commands ---------------------------------------------------------------------------
def test_every_type_validates():
    assert validate({'type': 'walk', 'forward_m': .4})['left_m'] == 0.
    assert validate({'type': 'turn', 'left_deg': -90})['left_deg'] == -90.
    assert validate({'type': 'pelvis_height', 'height_m': .55, 'behind_feet_m': .15})['behind_feet_m'] == .15
    assert validate({'type': 'torso_pitch', 'pitch_deg': 20})['pitch_deg'] == 20.
    c = validate({'type': 'hand_pose', 'hands': {'left': {'position': [.3, .2, .9], 'palm': [0, -2, 0], 'fingers': [1, 0, 0]}}})
    assert c['hands']['left']['palm'] == [0., -1., 0.] and c['duration_s'] == 1.5
    assert validate({'type': 'hand', 'left': 'close'}) == {'type': 'hand', 'left': 'close'}
    assert validate({'type': 'done'}) == {'type': 'done'}


@pytest.mark.parametrize('bad, code', [
    ({'type': 'fly'}, 'unknown_type'), ({'type': 'walk', 'forward_m': 3.}, 'out_of_range'),
    ({'type': 'walk', 'forward_m': 0.}, 'out_of_range'), ({'type': 'walk', 'forward_m': 'far'}, 'bad_number'),
    ({'type': 'walk', 'forward_m': float('nan')}, 'bad_number'), ({'type': 'turn', 'left_deg': 270}, 'out_of_range'),
    ({'type': 'pelvis_height', 'height_m': .2}, 'out_of_range'), ({'type': 'torso_pitch', 'pitch_deg': 80}, 'out_of_range'),
    ({'type': 'hand_pose', 'hands': {}}, 'bad_hands'), ({'type': 'hand_pose', 'hands': {'middle': {}}}, 'bad_hands'),
    ({'type': 'hand_pose', 'hands': {'left': {'position': [1, 2]}}}, 'bad_vector'),
    ({'type': 'hand_pose', 'hands': {'left': {'position': [0, 0, 1], 'palm': [1, 0, 0]}}}, 'bad_hands'),
    ({'type': 'hand_pose', 'hands': {'left': {'position': [0, 0, 1], 'palm': [1, 0, 0], 'fingers': [1, 0, .1]}}}, 'bad_vector'),
    ({'type': 'hand', 'left': 'squeeze'}, 'bad_hands'), ({'type': 'walk', 'forward_m': .3, 'speed': 1}, 'unknown_field')])
def test_refusals(bad, code):
    with pytest.raises(CommandError) as info:
        validate(bad)
    assert info.value.code == code


def test_workspace_needs_pelvis():
    far = {'type': 'hand_pose', 'hands': {'right': {'position': [2., 0., 1.]}}}
    validate(far)
    with pytest.raises(CommandError):
        validate(far, pelvis=[0., 0., .78])


def test_reply_parsing():
    assert parse('Sure.\n```json\n{"type": "turn", "left_deg": 45}\n```')['left_deg'] == 45.
    assert parse('I will walk {"type": "walk", "forward_m": 0.5} now.')['forward_m'] == .5
    for text in ('', 'no command', '{"type": "walk", "forward_m": }'):
        with pytest.raises(CommandError):
            parse(text)
    assert extract_json('{"a": {"b": 1}} trailing') == {'a': {'b': 1}}


def test_palm_and_fingers_round_trip():
    for side in ('left', 'right'):
        palm, fingers = np.array([0., -1., 0.]) * (1 if side == 'left' else -1), np.array([1., 0., -.2])
        rotation = wrist_rotation(side, palm, fingers)
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(rotation), 1.)
        back_palm, back_fingers = palm_and_fingers(side, rotation)
        assert np.allclose(back_palm, palm/np.linalg.norm(palm))
        assert back_fingers @ (fingers/np.linalg.norm(fingers)) > .99
    # Level wrists with palms facing each other are the identity: the box grasp's jaws.
    assert np.allclose(wrist_rotation('left', [0, -1, 0], [1, 0, 0]), np.eye(3))
    assert np.allclose(wrist_rotation('right', [0, 1, 0], [1, 0, 0]), np.eye(3))


def test_schema_documents_every_type():
    from kimodo_vla.agent.commands import SCHEMA, TYPES
    assert [o['properties']['type']['const'] for o in SCHEMA['oneOf']] == list(TYPES)


# ---- posture ----------------------------------------------------------------------------
def test_posture_keeps_feet_and_hits_targets():
    from kimodo_vla.agent.posture import posture_pose, torso_pitch_deg
    q, root = standing()
    feet = lambda q, r, R: FK(q, r, R, body_names=['left_ankle_roll_link', 'right_ankle_roll_link'])[0]
    before = feet(q, root, np.eye(3))
    for height, pitch, behind in [(.65, None, None), (None, 20., None), (.6, 25., None), (.5, None, .2)]:
        q2, r2, R2, info = posture_pose(FK, q, root, np.eye(3), height, pitch, behind)
        assert np.allclose(feet(q2, r2, R2), before, atol=2e-3)
        if height is not None:
            assert abs(r2[2]-height) < .01
        if pitch is not None:
            assert abs(torso_pitch_deg(FK, q2, r2, R2)-pitch) < 1.
        if behind is not None:
            assert abs(info['pelvis_behind_feet_m']-behind) < .01
    with pytest.raises(ValueError):
        posture_pose(FK, q, root, np.eye(3), .30)


# ---- the loop with a stand-in planner and commander -------------------------------------
class StandInPlanner(Planner):
    """Each command lasts two chunks of five rows; the reference never moves."""
    name = 'stand-in'
    fk = FK

    def start(self, command, context):
        self.left, self.context = 2, context
        self.hands = np.array(context['hands'], dtype=float)
        if command['type'] == 'hand':
            for i, s in enumerate(('left', 'right')):
                if s in command:
                    self.hands[i] = command[s] == 'close'
        return {'target': {'root_xy': [0., 0.]} if command['type'] == 'walk' else {}, 'plan': {},
                'hands_closed': [bool(h) for h in self.hands]}

    def chunk(self, request):
        if self.left == 0:
            return None
        self.left -= 1
        obs = request['observation']['oracle_history'][-1]
        return hold_rows(obs, np.eye(3), 5, self.hands)


class ListCommander:
    def __init__(self, replies, needs_image=False):
        self.replies, self.needs_image, self.seen = list(replies), needs_image, []

    def next(self, observation, image=None, error=None):
        from kimodo_vla.agent.commands import parse, validate
        self.seen.append((observation, image, error))
        raw = self.replies.pop(0) if self.replies else {'type': 'done'}
        record = {'source': 'list', 'raw': raw, 'command': None, 'error': None}
        try:
            record['command'] = parse(raw) if isinstance(raw, str) else validate(raw)
        except Exception as exc:
            record['error'] = str(exc)
        return record


def drive(policy, steps=200, vision=False):
    q, root = standing()
    t, responses, capture = 0., [], False
    pending = [sample(t, q, root)]
    for _ in range(steps):
        observation = {'oracle_history': pending, 'scene_truth': {'objects': [{'name': 'box', 'position': [.5, 0, .85]}]}}
        if capture:
            observation['vision'] = {'capture_path': 'none', 'time': pending[-1]['time']}
        response = policy({'observation': observation, 'task': 'x'})
        responses.append(response)
        rows = np.asarray(response['action_chunk'])
        assert rows.shape[1] == 40
        capture = response['command_loop']['capture']
        if response['command_loop']['end']:
            break
        pending = []
        for row in rows:
            t = round(t+.02, 6)
            s = sample(t, q, root)
            s['gripper_command'] = [int(row[38] >= .5), int(row[39] >= .5)]
            pending.append(s)
    return responses


def test_loop_runs_commands_and_logs(tmp_path):
    commander = ListCommander([{'type': 'walk', 'forward_m': .3}, 'garbage', {'type': 'hand', 'left': 'close'}, {'type': 'done'}])
    policy = CommandLoopPolicy(None, None, tmp_path, config={'budgets': {}}, task_text='t', planner=StandInPlanner(),
                               commander=commander, fk=FK, meta={'task': 'pp_box'})
    responses = drive(policy)
    assert responses[-1]['command_loop']['end']
    episode = load_episode(tmp_path)
    assert [c['type'] for c in episode['commands']] == ['walk', 'hand']
    assert all(c['end']['status'] == 'completed' for c in episode['commands'])
    assert episode['commands'][0]['target'] == {'root_xy': [0., 0.]}
    assert 'root_xy_m' in episode['commands'][0]['end']['measured_error']
    assert [r['error'] is None for r in episode['vlm']] == [True, False, True, True]
    assert commander.seen[2][2] is not None          # the refusal is passed back with the next request
    assert episode['summary']['end_reason'] == 'commander_done'
    sent = [json.loads(x) for x in (tmp_path/'loop/sent.jsonl').read_text().splitlines()]
    assert sum(len(x['rows']) for x in sent) > 0
    hand_rows = [r for x in sent if x['command'] == 1 for r in x['rows']]
    assert hand_rows and all(r[38] == 1. and r[39] == 0. for r in hand_rows)
    first = episode['commands'][0]
    assert first['time'] >= .6-1e-9                   # the start hold before the first command


def test_loop_budgets_and_refusals(tmp_path):
    policy = CommandLoopPolicy(None, None, tmp_path, config={'budgets': {'max_refused_in_row': 2}}, task_text='t',
                               planner=StandInPlanner(), commander=ListCommander(['x', 'y', 'z']), fk=FK)
    drive(policy)
    assert load_episode(tmp_path)['summary']['end_reason'] == 'refused_replies'
    policy = CommandLoopPolicy(None, None, tmp_path/'b', config={'budgets': {'max_commands': 2}}, task_text='t',
                               planner=StandInPlanner(), commander=ListCommander([{'type': 'turn', 'left_deg': 10}]*5), fk=FK)
    drive(policy)
    assert load_episode(tmp_path/'b')['summary']['end_reason'] == 'command_budget'


def test_loop_asks_for_a_capture_before_a_vlm_decision(tmp_path):
    commander = ListCommander([{'type': 'done'}], needs_image=True)
    policy = CommandLoopPolicy(None, None, tmp_path, config={'budgets': {}}, task_text='t', planner=StandInPlanner(),
                               commander=commander, fk=FK)
    policy.image = lambda vision: 'image' if vision else None
    responses = drive(policy, vision=True)
    assert any(r['command_loop']['capture'] for r in responses)
    assert commander.seen and commander.seen[0][1] == 'image'


# ---- scripted commanders ----------------------------------------------------------------
def test_smoke_script_is_valid_throughout():
    from exp3_scripted import SmokeScript
    script = SmokeScript()
    q, root = standing()
    observation = {'robot': {'pelvis': root.tolist(), 'heading_deg': 30., 'wrists': {}}, 'objects': []}
    seen = []
    while (command := script.next(observation)) is not None:
        seen.append(validate(command, pelvis=observation['robot']['pelvis'])['type'])
    assert set(seen) == {'walk', 'turn', 'torso_pitch', 'pelvis_height', 'hand_pose', 'hand'}


def test_box_script_grasp_commands_are_valid():
    from exp3_scripted import TaskScript
    box = {'name': 'box', 'position': [-.5, -4.8, .85], 'quat_wxyz': [1, 0, 0, 0], 'yaw_deg': 0., 'size': [.21]*3}
    table = {'name': 'table', 'min': [-.641, -6.24, 0.], 'max': [.12, -3.76, .745]}
    observation = {'robot': {'pelvis': [-1.6, -5.0, .78], 'heading_deg': 0.}, 'objects': [box, table]}
    plain = TaskScript('pp_box')                     # default: walk in with the hands in the walking pose (d7)
    assert plain.next(observation)['type'] == 'walk'
    observation['robot']['pelvis'] = [-.641-.20, -4.75, .78]
    assert plain.next(observation) == {'type': 'walk', 'forward_m': pytest.approx(-.04, abs=.011), 'left_m': 0.}
    observation['robot']['pelvis'] = [-1.6, -5.0, .78]
    script = TaskScript('pp_box')
    script.raise_before_walk = True
    raised = validate(script.next(observation), pelvis=observation['robot']['pelvis'])     # hands up before walking
    assert raised['type'] == 'hand_pose' and min(h['position'][2] for h in raised['hands'].values()) > .85+.105+.05
    assert script.next(observation) == {'type': 'hand', 'left': 'close', 'right': 'close'}
    first = script.next(observation)
    assert first['type'] == 'walk' and first['forward_m'] > .3
    final = script.final_stance(observation, box, raised=True)
    assert final[0] <= -.641-.17
    observation['robot']['pelvis'] = [-.641-.12, final[1], .78]     # walked too close to the table edge
    back = script.next(observation)
    assert back['type'] == 'walk' and back['forward_m'] < -.05
    observation['robot']['pelvis'] = [final[0], final[1], .78]
    commands = [validate(script.next(observation), pelvis=observation['robot']['pelvis']) for _ in range(7)]
    assert [c['type'] for c in commands] == ['hand', 'hand_pose', 'hand_pose', 'hand', 'hand_pose', 'hand_pose', 'walk']
    assert commands[0] == {'type': 'hand', 'left': 'open', 'right': 'open'}
    over, pre = commands[1]['hands']['left']['position'], commands[2]['hands']['left']['position']
    assert over[2] == pytest.approx(pre[2]+.10) and over[:2] == pytest.approx(pre[:2])


def test_config_names_dev_seeds_and_commanders():
    config = json.loads((HERE/'exp3_harness.json').read_text())
    seeds = config['tasks']['pp_box']['dev_seeds']+config['tasks']['sit_sofa']['dev_seeds']
    seeds += config['tasks']['vision_navi']['dev_seeds']
    assert len(set(seeds)) == len(seeds) == 30
    for spec in config['commanders'].values():
        assert (HERE/spec.get('file', spec.get('page', ''))).is_file()


def test_prompt_block_fills():
    from kimodo_vla.agent.commander import fill, load_prompt
    template = load_prompt(HERE/'vlm-config.md')
    text = fill(template, {'task': 'Sit down.', 'robot': {'pelvis': [1.23456, 0, .78]}}, error='no_json: x')
    assert 'Sit down.' in text and '1.235' in text and 'refused (no_json: x)' in text
    assert '{observation}' not in text and '{task}' not in text and '{error}' not in text
    assert 'refused' not in fill(template, {'task': 't'})


def test_task_scripts_on_recorded_scenes():
    """Both rules on the geometry the smoke runs recorded (seed 90000 and 91000): valid commands, sensible targets."""
    from exp3_scripted import TaskScript, clearance, nearest_face
    shelf = {'name': 'shelf', 'min': [-3.316, -3.209, 0.], 'max': [-1.236, -2.146, 2.18]}
    layer = {'name': 'shelf_target_layer', 'min': [-3.316, -3.209, .648], 'max': [-1.236, -2.146, .648]}
    normal, _, inside = nearest_face(shelf, [-1.3, -5.2])
    assert normal.tolist() == [0., -1.] and inside(.22, .25)[1] == pytest.approx(-3.209+.22)
    script = TaskScript('pp_box')
    script.stage = 'grasp'
    box = {'name': 'box', 'position': [-.85, -5.2, 1.07], 'quat_wxyz': [1, 0, 0, 0]}
    wrists = {s: {'position': [-.95, -5.2+(.16 if s == 'left' else -.16), 1.1]} for s in ('left', 'right')}
    observation = {'robot': {'pelvis': [-1.3, -5.2, .78], 'heading_deg': 0., 'wrists': wrists}, 'objects': [box, shelf, layer]}
    first = script.next(observation)
    assert first == {'type': 'turn', 'left_deg': 90.}
    assert script.place_yaw == pytest.approx(90.) and script.place_z == pytest.approx(.648+.105+.02)
    assert script.approach_xy[1] == pytest.approx(-3.209-.05-.105) and script.place_xy[1] == pytest.approx(-3.209+.15)
    seat = {'name': 'sofa_seat', 'min': [-4.306, -1.690, -.31], 'max': [-2.326, -.785, .31]}
    cartons = [{'name': 'carton_obstacle_01', 'position': [-1.86, .60, 0.], 'size': [.41, .41, .42]},
               {'name': 'carton_obstacle_02', 'position': [-1.92, .08, 0.], 'size': [.41, .41, .42]}]
    sofa = TaskScript('sit_sofa')
    observation = {'robot': {'pelvis': [-1.6, 2.0, .78], 'heading_deg': -90.}, 'objects': [seat]+cartons}
    turn = sofa.next(observation)
    assert turn['type'] == 'turn' and abs(turn['left_deg']) == pytest.approx(180.)
    assert sofa.sit_yaw == pytest.approx(90.) and sofa.stand[1] == pytest.approx(-.785+.10)
    tea = {'name': 'model_teatable', 'kind': 'region', 'min': [-3.92, -.19, 0.], 'max': [-2.64, .51, .4]}
    around = TaskScript('sit_sofa')
    around.next({**observation, 'objects': [seat, tea]+cartons})
    legs = [[-1.6, 2.0]]+around.route
    assert len(around.route) > 1 and all(clearance(np.asarray(a)+(np.asarray(b)-np.asarray(a))*u, tea) > .3
                                         for a, b in zip(legs, legs[1:]) for u in np.linspace(0, 1, 30))
    for leg_start, leg_end in zip([[-1.6, 2.0]]+sofa.route, sofa.route):
        for u in np.linspace(0, 1, 20):
            p = np.asarray(leg_start)+(np.asarray(leg_end)-np.asarray(leg_start))*u
            assert all(clearance(p, c) > .3 for c in cartons)


def test_planner_settles_before_reaching():
    """A hand_pose holds the body until it is still (at least the minimum), then arms the reach once."""
    from kimodo_vla.agent.planner import KimodoPlanner, SETTINGS

    class Reacher:
        armed, run = [], None

        def arm(self, obs, prior, heading, goals, duration, settle, feedback):
            self.armed.append((obs['time'], [p['time'] for p in prior]))

        def step(self, request):
            return np.zeros((25, 40))
    planner = KimodoPlanner.__new__(KimodoPlanner)
    planner.settings, planner.fk, planner.reacher, planner.active, planner.settle = dict(SETTINGS), FK, Reacher(), None, None
    q, root = standing()
    history = [sample(round(.02*i, 2), q, root) for i in range(30)]
    moving = [dict(s, root_pos=[s['time']*.1, 0., .784]) for s in history]      # still walking at 0.1 m/s
    command = validate({'type': 'hand_pose', 'hands': {'left': {'position': [.3, .2, .9]}}})
    import tempfile
    planner.start(command, {'history': moving, 'hands': [0, 0], 'heading': np.eye(3), 'folder': tempfile.mkdtemp(),
                            'seed': 1, 'new_samples': 5})
    request = lambda samples: {'observation': {'oracle_history': samples}}
    rows = planner.chunk(request(moving[-5:]))
    assert rows.shape == (5, 40) and not planner.reacher.armed            # the minimum hold
    t, armed_at = moving[-1]['time'], None
    for _ in range(40):
        new = [sample(round(t+.02*(i+1), 2), q, root) for i in range(5)]
        t = new[-1]['time']
        rows = planner.chunk(request(new))
        if planner.reacher.armed:
            armed_at = t
            break
    assert armed_at is not None and armed_at-moving[-1]['time'] >= SETTINGS['reach_settle_s'][0]-1e-6
    at, prior = planner.reacher.armed[0]
    assert at == armed_at and prior[-1] < at and len(prior) == len(set(prior))   # prior excludes this request, no repeats


def test_loop_reports_a_target_before_the_first_reach_is_armed(tmp_path):
    """While the planner holds the body still before its first hand_pose, the reacher has no target yet."""
    class Settling(StandInPlanner):
        reacher = object()
    commander = ListCommander([{'type': 'hand_pose', 'hands': {'left': {'position': [.3, .2, .9]}}}, {'type': 'done'}])
    policy = CommandLoopPolicy(None, None, tmp_path, config={'budgets': {}}, task_text='t', planner=Settling(),
                               commander=commander, fk=FK)
    responses = drive(policy)
    assert responses[-1]['command_loop']['end'] and all('targets_world' in r['oracle_target'] for r in responses)


def test_a_command_the_planner_cannot_carry_out_is_logged_and_asked_again(tmp_path):
    class Refusing(StandInPlanner):
        def start(self, command, context):
            if command['type'] == 'pelvis_height':
                raise ValueError('Posture out of reach')
            return super().start(command, context)
    commander = ListCommander([{'type': 'pelvis_height', 'height_m': .45, 'behind_feet_m': .3}, {'type': 'done'}])
    policy = CommandLoopPolicy(None, None, tmp_path, config={'budgets': {}}, task_text='t', planner=Refusing(),
                               commander=commander, fk=FK)
    drive(policy)
    episode = load_episode(tmp_path)
    assert episode['commands'][0]['end']['status'] == 'planner_refused'
    assert episode['summary']['end_reason'] == 'commander_done'
    assert commander.seen[1][0]['last_command']['status'] == 'planner_refused'


def test_development_seeds_are_not_evaluation_episodes():
    """Plan v3: debugging stays off the 60 official evaluation episodes of each task."""
    from exp3_eval_episodes import derive_episode_seed, official_episodes
    config = json.loads((HERE/'exp3_harness.json').read_text())
    for task in ('pp_box', 'sit_sofa', 'vision_navi'):
        episodes = official_episodes(task)
        assert len(episodes) == 60 and len({e['episode_seed'] for e in episodes}) == 60
        evaluation = {e['episode_seed'] for e in episodes} | {e['group_seed'] for e in episodes}
        dev = set(config['tasks'][task]['dev_seeds'])
        assert not dev & evaluation       # a development run uses seed = episode seed = its dev seed
    assert derive_episode_seed('X', 0, 0) == derive_episode_seed('X', 0, 0) < 2**31
    # Experiment 2's exp2_replay.official_episode_seed, which
    # restored and replayed P&PBox (group 0, repeat 0) with this episode seed; and the value checked here before.
    assert official_episodes('pp_box')[0]['episode_seed'] == 575085118
    assert official_episodes('pp_box')[27]['episode_seed'] == 274596493     # group 1, repeat 7


def test_planner_moves_avoid_rules_and_corrections():
    """A sidestep beyond the tracker's rule becomes turn + forward walk + turn back; afterwards the measured
    root error is walked off again, at most move_corrections times."""
    from kimodo_vla.agent.planner import KimodoPlanner, SETTINGS
    planner = KimodoPlanner.__new__(KimodoPlanner)
    planner.settings = {**SETTINGS, 'avoid_rules': {'sonic': {'sidestep': {'max_m': .3}}}}
    planner.fk = FK
    planner.move_target = {'root_xy': [0., .8], 'yaw_deg': 0.}
    now = {'root': [0., 0., .78], 'yaw_deg': 0.}
    moves = planner.plan_moves('walk', now)
    assert [m[0] for m in moves] == ['face', 'walk_to', 'face'] and moves[0][1] == pytest.approx(90.)
    planner.move_target = {'root_xy': [.8, .1], 'yaw_deg': 0.}
    assert [m[0] for m in planner.plan_moves('walk', now)] == ['walk_to']
    planner.settings['avoid_rules'] = {'sonic': {}}
    planner.move_target = {'root_xy': [0., .8], 'yaw_deg': 0.}
    assert [m[0] for m in planner.plan_moves('walk', now)] == ['walk_to']
    assert planner.plan_moves('turn', now) == [('face', 0.)]

    class Walker:
        native_settings = {'prime_samples': 30, 'speed_m_s': .35, 'tolerance_m': .05}
        yaw_gain = 1.
    started = []
    import kimodo_vla.agent.planner as module
    original = module.prime
    module.prime = lambda walker, **kw: started.append(kw['goal'])
    try:
        import tempfile
        q, root = standing()
        planner.walker, planner.heading, planner.folder, planner.seed = Walker(), np.eye(3), Path(tempfile.mkdtemp()), 1
        planner.recent = [sample(0., q, root)]
        planner.move_target, planner.corrections, planner.moves_done = {'root_xy': [.5, 0.], 'yaw_deg': 0.}, 0, []
        planner.moves = [('walk_to', [.5, 0.])]
        assert planner.next_move(sample(0., q, root)) and 'xy' in started[-1]['root']
        short = sample(1., q, root+[.3, 0., 0.])           # stopped 20 cm short: walk again, twice at most
        assert planner.next_move(short) and planner.corrections == 1
        assert planner.next_move(short) and planner.corrections == 2
        assert not planner.next_move(short)
        there = sample(2., q, root+[.48, 0., 0.])
        planner.corrections, planner.moves = 0, []
        assert not planner.next_move(there)
    finally:
        module.prime = original


def test_navi_script_routes_around_low_obstacles_only():
    from exp3_scripted import TaskScript, clearance
    target = {'name': 'target', 'kind': 'region', 'min': [-1.9, -1.3, 0.], 'max': [-1.3, -.7, 0.], 'center': [-1.6, -1., 0.]}
    wall = {'name': 'obstacle01_a', 'kind': 'region', 'min': [-2.4, -3.3, 0.], 'max': [-.8, -3.0, .3]}
    beam = {'name': 'obstacle02_c', 'kind': 'region', 'min': [-2.4, -2.3, 1.6], 'max': [-.8, -2.0, 2.0]}
    script = TaskScript('vision_navi')
    observation = {'robot': {'pelvis': [-1.9, -5.2, .78], 'heading_deg': 90.}, 'objects': [target, wall, beam]}
    first = validate(script.next(observation))
    assert first['type'] == 'walk'
    legs = [[-1.9, -5.2]]+script.route
    assert len(script.route) > 1 and script.route[-1] == pytest.approx([-1.6, -1.])
    # The 0.3 m high slab needs only the legs' clearance; the hands pass over it.
    assert all(clearance(np.asarray(a)+(np.asarray(b)-np.asarray(a))*u, wall) > script.legs_clearance_m-.01
               for a, b in zip(legs, legs[1:]) for u in np.linspace(0, 1, 30))


def test_navi_verdict(tmp_path):
    from kimodo_vla.agent.judge import judge
    loop = tmp_path/'loop'
    loop.mkdir()
    (loop/'meta.json').write_text(json.dumps({'task': 'vision_navi'}))
    (loop/'scene.json').write_text(json.dumps({'objects': [{'name': 'target', 'center': [0., 0., 0.], 'min': [-.3, -.3, 0.],
                                                             'max': [.3, .3, 0.]}]}))
    (loop/'commands.jsonl').write_text('')
    (loop/'steps.jsonl').write_text(json.dumps({'time': 1., 'root_pos': [1., 0., .78], 'fallen': False})+'\n')
    (tmp_path/'episode.json').write_text(json.dumps({'success': False, 'failure_reason': 'timeout'}))
    v = judge(load_episode(tmp_path), {})
    assert v['link'] == 'self-check failed' and v['reason'] == 'ended outside the target'
    assert v['margins']['final_pelvis_to_target_m'] == pytest.approx(1.)


def test_runner_accepts_every_configured_task():
    import subprocess, sys
    config = json.loads((HERE/'exp3_harness.json').read_text())
    text = subprocess.run([sys.executable, str(HERE/'exp3_run.py'), '--help'], capture_output=True, text=True).stdout
    assert all(task in text for task in config['tasks'])


def test_true_shelf_face_carry_height_and_insert_check():
    """true_shelf: the turned shelf's long side, the carried box under the board above, its corners checked before it
    goes in; carry_turn_max_deg splits a carrying turn."""
    from exp3_scripted import TaskScript, corner_heights, open_face
    yaw = 15.
    oriented = {'center': [-2.27, -2.7, 1.14], 'yaw_deg': yaw, 'size': [1.93, .6, 2.08]}
    normal, along, inside = open_face(oriented, [-1.3, -5.2])
    r = np.radians(yaw)
    assert normal == pytest.approx([np.sin(r), -np.cos(r)]) and along == pytest.approx([np.cos(r), np.sin(r)])
    face_point = np.asarray(inside(0., 0., .5))
    assert face_point == pytest.approx(np.asarray(oriented['center'][:2])+normal*.3)
    assert np.dot(np.asarray(inside(.35, .4))-face_point, normal) == pytest.approx(-.35)
    tilt = np.radians(18.)
    tilted = {'position': [0, 0, 1.05], 'quat_wxyz': [np.cos(tilt/2), np.sin(tilt/2), 0, 0]}
    low, high = corner_heights(tilted, .105)
    assert high-1.05 == pytest.approx(.105*(np.cos(tilt)+np.sin(tilt))) and low == pytest.approx(2.1-high)

    shelf = {'name': 'shelf', 'min': [-3.3, -3.2, 0.], 'max': [-1.2, -2.2, 2.18], 'oriented': oriented}
    layer = {'name': 'shelf_target_layer', 'min': [-3.3, -3.2, .648], 'max': [-1.2, -2.2, .648], 'clear_height_m': .456}
    script = TaskScript('pp_box', place_by_walk=True, place_depth_m=.35, place_clearance_m=.1, place_end_margin_m=.3,
                        true_shelf=True, carry_turn_max_deg=45)
    script.stage = 'grasp'
    box = {'name': 'box', 'position': [-.85, -5.2, 1.07], 'quat_wxyz': [np.cos(tilt/2), 0, np.sin(tilt/2), 0]}
    wrists = {s: {'position': [-.95, -5.2+(.15 if s == 'left' else -.15), 1.1]} for s in ('left', 'right')}
    observation = {'robot': {'pelvis': [-1.3, -5.2, .78], 'heading_deg': 0., 'wrists': wrists}, 'objects': [box, shelf, layer]}
    turn = script.next(observation)        # first a turn of at most 45 degrees toward the shelf
    assert turn['type'] == 'turn' and turn['left_deg'] == pytest.approx(45.)
    assert script.place_yaw == pytest.approx(np.degrees(np.arctan2(-normal[1], -normal[0])))
    facing = {**observation, 'robot': {**observation['robot'], 'heading_deg': script.place_yaw}}
    down = script.next(facing)             # facing the shelf, carried at 1.07 m: down under the board above (1.104 m)
    ceiling = .648+.456-script.carry_clear_m
    assert down['type'] == 'hand_pose'
    assert down['hands']['left']['position'][2]-1.1 == pytest.approx(ceiling-corner_heights(box, .105)[1])
    walk = script.next(facing)             # then the carrying walk, at most 0.9 m
    assert walk['type'] == 'walk' and np.hypot(walk['forward_m'], walk['left_m']) == pytest.approx(.9)
    refused = {**facing, 'last_command': {'status': 'planner_refused', 'command': walk}}
    shorter = script.next(refused)
    assert np.hypot(shorter['forward_m'], shorter['left_m']) == pytest.approx(.45)
    # in front of the shelf, the lowering puts the lowest corner 0.10 m over the board, and the check leaves a box
    # that is in the room alone
    placed = {**box, 'position': [-.85, -5.2, .9]}
    assert script.insert_dz({**observation, 'objects': [placed, shelf, layer]}) == pytest.approx(
        .748-corner_heights(placed, .105)[0])
    level = {**box, 'quat_wxyz': [1, 0, 0, 0], 'position': [-.85, -5.2, .648+.105+.08]}
    assert script.insert_check({**observation, 'objects': [level, shelf, layer]}) is None
    sagged = {**level, 'position': [-.85, -5.2, .648+.105+.02]}
    fix = script.insert_check({**observation, 'objects': [sagged, shelf, layer]})
    assert fix['hands']['left']['position'][2]-1.1 == pytest.approx(.08)
    resting = {**level, 'position': [-.85, -5.2, .648+.105+.10]}
    put = script.set_down({**observation, 'objects': [resting, shelf, layer]})
    assert put['hands']['left']['position'][2]-1.1 == pytest.approx(0.-.10)
    assert script.set_down({**observation, 'objects': [{**level, 'position': [-.85, -5.2, .648+.105+.005]}, shelf, layer]}) is None
