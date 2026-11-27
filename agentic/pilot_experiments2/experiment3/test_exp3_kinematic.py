"""The kinematic planner (kimodo_vla/agent/kinematic.py) without HITCH: a fake helper replies with world frames."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kimodo_vla.agent.kinematic import FROM_HITCH, HITCH_JOINTS, KinematicPlanner, TO_HITCH, world_to_rows
from kimodo_vla.core.types import G1_JOINT_NAMES, yaw_matrix


def quat(yaw):
    return Rotation.from_euler('z', yaw).as_quat(scalar_first=True)


def test_joint_orders_round_trip():
    joints = np.arange(29.)
    assert np.array_equal(joints[TO_HITCH][FROM_HITCH], joints)
    assert [G1_JOINT_NAMES[i] for i in TO_HITCH] == list(HITCH_JOINTS)


def test_world_frames_to_rows():
    yaw, anchor = np.radians(30.), np.radians(10.)
    heading = yaw_matrix(anchor)
    joints = np.linspace(-.5, .5, 29)
    t = np.arange(6)*.02
    world = np.zeros((6, 36))
    world[:, 0], world[:, 1] = 1.+.3*t*np.cos(yaw), 2.+.3*t*np.sin(yaw)
    world[:, 2] = .76
    world[:, 3:7] = quat(yaw)
    world[:, 7:] = joints[TO_HITCH]
    rows = world_to_rows(world, heading, [1., 0.])
    assert rows.shape == (5, 40)
    assert rows[:, 0] == pytest.approx(.3*.02) and rows[:, 1] == pytest.approx(0., abs=1e-12)   # body frame steps
    rel = heading.T @ Rotation.from_euler('z', yaw).as_matrix()
    assert rows[0, 3:9] == pytest.approx(rel[:, :2].ravel())
    assert rows[0, 9:38] == pytest.approx(joints) and rows[0, 2] == .76 and rows[0, 38:].tolist() == [1., 0.]


class FakeClient:
    def __init__(self):
        self.messages = []

    def plan(self, message):
        self.messages.append(message)
        start = np.asarray(message['start_pose'])
        world = np.repeat(start[None], 30, 0)
        if 'walk' in message['request']:
            x, y, yaw = message['frame']
            tx, ty = message['request']['walk']['target']
            end = np.array([x+np.cos(yaw)*tx-np.sin(yaw)*ty, y+np.sin(yaw)*tx+np.cos(yaw)*ty])
            world[:, :2] = np.linspace(start[:2], end, 30)
        return {'ok': True, 'passed': True, 'world': world.tolist(), 'end_torso': [0, 0], 'duration_s': .58}


def sample(xy, yaw, z=.77):
    q = quat(np.radians(yaw)).tolist()
    return {'time': 3., 'root_pos': [*xy, z], 'root_quat': q, 'reference_root_pos': [*xy, z], 'reference_root_quat': q,
            'joints': [0.]*29, 'reference_joints': [0.]*29}


class FK:
    def __call__(self, joints, root, rotation, body_names):
        return np.zeros((len(body_names), 3))+root, np.repeat(np.asarray(rotation)[None], len(body_names), 0)


def test_walk_request_and_correction(tmp_path):
    client = FakeClient()
    planner = KinematicPlanner(tmp_path, fk=FK(), client=client)
    obs = sample([1., 1.], 90.)
    info = planner.start({'type': 'walk', 'forward_m': .5, 'left_m': 0.},
                         {'history': [obs], 'hands': [0, 0], 'heading': np.eye(3), 'folder': tmp_path})
    assert info['target']['root_xy'] == pytest.approx([1., 1.5])
    assert client.messages[0]['request']['walk'] == {'target': [pytest.approx(.5), pytest.approx(0., abs=1e-9)], 'turn_deg': 0}
    rows = planner.chunk({'observation': {'oracle_history': [obs]}})
    assert rows.shape == (25, 40) and rows[0, 0] > 0 and abs(rows[0, 1]) < 1e-9   # forward in the body frame
    assert planner.chunk({'observation': {'oracle_history': [obs]}}).shape == (4, 40)
    # the true root ended 0.5 m short: walked off again, from the new reference
    far = sample([1., 1.], 90.)
    assert planner.chunk({'observation': {'oracle_history': [far]}}) is not None and len(client.messages) == 2
    # the reference anywhere (Arena's integrator drifts), the true root 0.2 m short: the plan starts at the root
    lag = sample([1., 1.3], 90.)
    lag['reference_root_pos'] = [1., 0.7, .77]
    planner.queue, planner.corrections, planner.last_off = None, 0, None
    planner.chunk({'observation': {'oracle_history': [lag]}})
    assert client.messages[-1]['request']['walk']['target'] == [pytest.approx(.2), pytest.approx(0., abs=1e-9)]
    assert client.messages[-1]['start_pose'][:2] == [1., 1.3] and client.messages[-1]['frame'][:2] == [1., 1.3]
    # a correction that did not reduce the error: no more
    sent = len(client.messages)
    planner.queue = None
    assert planner.chunk({'observation': {'oracle_history': [lag]}}) is None and len(client.messages) == sent
    near = sample([1., 1.49], 90.)
    planner.queue = None
    assert planner.chunk({'observation': {'oracle_history': [near]}}) is None


def test_refusals(tmp_path):
    planner = KinematicPlanner(tmp_path, fk=FK(), client=FakeClient())
    context = {'history': [sample([0., 0.], 0.)], 'hands': [0, 0], 'heading': np.eye(3), 'folder': tmp_path}
    with pytest.raises(ValueError):
        planner.start({'type': 'pelvis_height', 'height_m': .5, 'behind_feet_m': .1}, context)
    info = planner.start({'type': 'hand', 'right': 'close'}, context)
    assert info['hands_closed'] == [False, True]
    assert planner.chunk({'observation': {'oracle_history': context['history']}})[0, 39] == 1.


def test_box_rule_stops_after_the_hold_stage():
    from exp3_scripted import TaskScript
    script = TaskScript('pp_box', stop_after='hold')
    box = {'name': 'box', 'position': [-.5, -4.8, .85], 'quat_wxyz': [1, 0, 0, 0], 'yaw_deg': 0., 'size': [.21]*3}
    table = {'name': 'table', 'min': [-.641, -6.24, 0.], 'max': [.12, -3.76, .745]}
    stance = script.final_stance({'objects': [box, table]}, box)
    observation = {'robot': {'pelvis': [stance[0], stance[1], .78], 'heading_deg': 0.}, 'objects': [box, table]}
    kinds = [script.next(observation) for _ in range(8)]
    assert [k['type'] for k in kinds[:6]] == ['hand_pose', 'hand_pose', 'hand_pose', 'hand', 'hand_pose', 'hand_pose']
    assert kinds[6]['type'] == 'hand_pose' and kinds[6]['duration_s'] == 2.0 and kinds[7] is None


class LeanOnly(FakeClient):
    """Refuses a reach unless the torso leans (as HITCH did for k5heft 90000)."""

    def plan(self, message):
        if message.get('wrist_targets') and 'torso' not in message['request']:
            self.messages.append(message)
            return {'ok': False, 'code': 'PLAN_REJECTED', 'error': 'An arm cannot hold its palm pose'}
        reply = super().plan(message)
        reply['end_torso'] = [message['request'].get('torso', {}).get('pitch_deg', 0), 0]
        return reply


def test_refused_reach_leans_and_the_next_walk_straightens(tmp_path):
    client = LeanOnly()
    planner = KinematicPlanner(tmp_path, fk=FK(), client=client)
    context = {'history': [sample([0., 0.], 0.)], 'hands': [0, 0], 'heading': np.eye(3), 'folder': tmp_path}
    info = planner.start({'type': 'hand_pose', 'duration_s': 1., 'hands': {'right': {'position': [.4, -.2, .9]}}}, context)
    assert info['plan']['lean_deg'] == 10 and planner.torso == (10, 0)
    planner.start({'type': 'walk', 'forward_m': .3, 'left_m': 0.}, context)
    assert client.messages[-1]['request']['torso'] == {'pitch_deg': 0, 'yaw_deg': 0}


class SlowOnly(FakeClient):
    def plan(self, message):
        w = message.get('wrist_targets') or {}
        if w and any(t['duration_s'] < 4. for t in w.values()):
            self.messages.append(message)
            return {'ok': False, 'code': 'PLAN_REJECTED', 'error': 'The motion from 0.00 s to 1.20 s needs 4.84 s, more than twice'}
        return super().plan(message)


def test_hand_pose_retried_at_the_duration_hitch_needs(tmp_path):
    client = SlowOnly()
    planner = KinematicPlanner(tmp_path, fk=FK(), client=client)
    context = {'history': [sample([0., 0.], 0.)], 'hands': [1, 1], 'heading': np.eye(3), 'folder': tmp_path}
    info = planner.start({'type': 'hand_pose', 'duration_s': 1.2, 'hands': {'left': {'position': [.3, .1, 1.]}}}, context)
    assert client.messages[-1]['wrist_targets']['left']['duration_s'] == pytest.approx(4.84*1.05)
    assert info['plan']['duration_s_asked'] == 1.2
