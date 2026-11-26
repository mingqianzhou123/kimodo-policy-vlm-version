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
    script = TaskScript('pp_box')
    box = {'name': 'box', 'position': [-.5, -4.8, .85], 'quat_wxyz': [1, 0, 0, 0], 'yaw_deg': 0., 'size': [.21]*3}
    table = {'name': 'table', 'min': [-.641, -6.24, 0.], 'max': [.12, -3.76, .745]}
    observation = {'robot': {'pelvis': [-1.6, -5.0, .78], 'heading_deg': 0.}, 'objects': [box, table]}
    first = script.next(observation)
    assert first['type'] == 'walk' and first['forward_m'] > .3
    stance = script.box_stance(box['position'])
    observation['robot']['pelvis'] = [-.641-.12, stance[1], .78]     # walked too close to the table edge
    back = script.next(observation)
    assert back['type'] == 'walk' and back['forward_m'] < -.05
    observation['robot']['pelvis'] = [min(stance[0], -.641-.24), stance[1], .78]
    kinds = [validate(script.next(observation), pelvis=observation['robot']['pelvis'])['type'] for _ in range(6)]
    assert kinds == ['hand_pose', 'hand_pose', 'hand', 'hand_pose', 'hand_pose', 'walk']


def test_config_names_dev_seeds_and_commanders():
    config = json.loads((HERE/'exp3_harness.json').read_text())
    seeds = config['tasks']['pp_box']['dev_seeds']+config['tasks']['sit_sofa']['dev_seeds']
    assert len(set(seeds)) == len(seeds) == 20
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
