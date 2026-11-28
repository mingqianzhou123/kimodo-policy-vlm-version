"""A kinematic planner behind the ``Planner`` seam: HITCH's whole-body planner (``wbc.planner.request.plan``).

Each body command becomes one HITCH request, planned from the reference the tracker is on (as HITCH's own session
plans from the reference tail), and HITCH's world frames become the loop's 40-d rows. Nothing here plans motion;
it translates. HITCH runs in its own environment (``hitch_server.py``), started on first use from
``$HITCH_PYTHON`` (its ``.venv/bin/python``, ``wbc`` importable) with ``WBC_HAND=dex3`` (Arena's G1 has Dex3 hands).

  walk        ``{"walk": {"target": [x, y], "turn_deg": t}}`` in the plan frame; the target is the commanded point
              from the measured root (as ``KimodoPlanner``), the turn keeps the measured heading. HITCH walks with the
              arms locked at their joint angles; with both hands open the request first moves the palms to
              ``walk_palms`` ([forward, out, height] from the pelvis: beside the body, above a table, outside a box
              in front), so the open hands are not carried in front at table height. HITCH's own ``home`` is the
              Dex3 ready pose (wrists 0.22 m in front at 0.92 m): it touched the box on the way in (k3heft 90004). With
              a hand closed the arms stay as they are, holding what they hold. After the plan, the remaining error of the true root pose is walked
              off again, at most ``move_corrections`` times (``walk_tolerance_m``, ``turn_tolerance_deg``)
  turn        a walk to where the robot stands with ``turn_deg`` (integer degrees)
  pelvis_height  ``{"height": h}`` (HITCH: 0.30-0.79 m; ``behind_feet_m`` is not supported: refused)
  torso_pitch ``{"torso": {"pitch_deg": p, "yaw_deg": 0}}`` (HITCH: multiples of 5 degrees)
  hand_pose   wrist targets, turned into HITCH palm waypoints by the helper (its hand's palm offset)
  hand        the hand bits change; the body holds the reference it is on

A plan HITCH refuses or whose checks fail raises ``ValueError``: the loop logs the command as refused by the planner
and asks the commander again. ``KimodoPlanner`` is unchanged; the planner is chosen by name (``make_planner``).
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.oracle.geometry import SIDES, rotation_wxyz
from .commands import wrist_rotation
from .planner import SETTINGS as KIMODO_SETTINGS, Planner, hold_rows
from .state import body_state, yaw_deg

# wbc.core.layout.JOINTS (HITCH's pose layout), copied so that this process needs no HITCH import
HITCH_JOINTS = (
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint',
    'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint',
    'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
    'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint',
    'right_wrist_pitch_joint', 'right_wrist_yaw_joint')
TO_HITCH = [G1_JOINT_NAMES.index(n) for n in HITCH_JOINTS]
FROM_HITCH = [HITCH_JOINTS.index(n) for n in G1_JOINT_NAMES]
WALK_HEIGHT_MAX = .78      # HITCH's walker range (wbc.helpers.gait.WALK_HEIGHT)
WALK_HEIGHT = .76          # a start above it walks after lowering the pelvis to this
CHUNK = 25

SETTINGS = {k: KIMODO_SETTINGS[k] for k in ('hand_toggle_s', 'walk_tolerance_m', 'turn_tolerance_deg', 'move_corrections',
                                             'tracker', 'avoid_rules')}
SETTINGS.update(hitch_python=None, hand='dex3', walk_palms=[.10, .30, .95],
                reach_lean_deg=[10, 20],      # a refused reach is retried with this torso lean (HITCH plans both together)
                slower_when_carrying=True)    # walks with a hand closed at HITCH's slower timing


def wrap_deg(angle):
    return (float(angle)+180.) % 360.-180.


def heading_rad(quat_wxyz):
    r = rotation_wxyz(quat_wxyz)
    return float(np.arctan2(r[1, 0], r[0, 0]))


def to_hitch_pose(root_pos, root_quat, joints_g1):
    return np.r_[np.asarray(root_pos, float), np.asarray(root_quat, float), np.asarray(joints_g1, float)[TO_HITCH]]


def world_to_rows(world, heading, hands):
    """HITCH world frames [T, 36] (frame 0 = the reference the tracker is on) -> 40-d rows [T-1, 40]: root steps in the
    body frame, root height, root rotation relative to the loop's ``heading`` (6-D), joints in G1 order, hand bits."""
    world = np.asarray(world, float)
    rel = np.einsum('ij,tjk->tik', heading.T, np.array([rotation_wxyz(q) for q in world[:, 3:7]]))
    step = np.diff(world[:, :3], axis=0) @ heading         # = heading.T @ step, row-wise
    rows = np.zeros((len(world)-1, 40))
    rows[:, :2] = np.einsum('tji,tj->ti', rel[1:], step)[:, :2]
    rows[:, 2] = world[1:, 2]
    rows[:, 3:9] = rel[1:, :, :2].reshape(-1, 6)
    rows[:, 9:38] = world[1:, 7:][:, FROM_HITCH]
    rows[:, 38:40] = hands
    return rows


class HitchClient:
    """The helper process (``hitch_server.py``), one JSON line each way."""

    def __init__(self, python, hand='dex3'):
        if not python:
            raise ValueError('The kinematic planner needs HITCH: set $HITCH_PYTHON (HITCH/.venv/bin/python)')
        env = {**os.environ, 'WBC_HAND': hand}
        self.process = subprocess.Popen([str(python), '-u', str(Path(__file__).with_name('hitch_server.py'))],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
        ready = self.read()
        if not ready.get('ready'):
            raise RuntimeError(f'HITCH helper did not start: {ready}')
        self.hand = ready.get('hand')

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'HITCH helper exited ({self.process.poll()})')
        return json.loads(line)

    def plan(self, message):
        self.process.stdin.write(json.dumps(message)+'\n')
        self.process.stdin.flush()
        return self.read()


class KinematicPlanner(Planner):
    name = 'hitch'

    def __init__(self, run, settings=None, fk=None, client=None):
        self.settings = {**SETTINGS, **(settings or {})}
        self.run = Path(run)
        if fk is None:
            from kimodo_vla.oracle.demo_walks import g1_fk
            fk = g1_fk()
        self.fk = fk
        self.client = client
        self.torso = (0, 0)
        self.queue = None
        self.log = []

    def connect(self):
        if self.client is None:
            self.client = HitchClient(self.settings['hitch_python'] or os.environ.get('HITCH_PYTHON'), self.settings['hand'])
        return self.client

    # ---- the interface -----------------------------------------------------------------
    def start(self, command, context):
        history, heading = context['history'], np.asarray(context['heading'], dtype=float)
        obs = history[-1]
        self.hands = np.asarray(context['hands'], dtype=float)
        self.heading, self.folder = heading, Path(context['folder'])
        kind = command['type']
        now = body_state(obs, self.fk)
        self.move, self.corrections, self.last_off = None, 0, None
        target, plan = {}, {}
        if kind in ('walk', 'turn'):
            yaw = np.radians(now['yaw_deg'])
            if kind == 'walk':
                c, s = np.cos(yaw), np.sin(yaw)
                goal = np.asarray(obs['root_pos'][:2])+np.array([c*command['forward_m']-s*command['left_m'],
                                                                 s*command['forward_m']+c*command['left_m']])
                target = {'root_xy': goal.tolist(), 'yaw_deg': now['yaw_deg']}
            else:
                target = {'root_xy': list(obs['root_pos'][:2]), 'yaw_deg': wrap_deg(now['yaw_deg']+command['left_deg'])}
            self.move = target
            plan = self.plan_walk(obs, target)
        elif kind == 'pelvis_height':
            if command.get('behind_feet_m') is not None:
                raise ValueError('The kinematic planner (HITCH) has no sit-back posture (behind_feet_m)')
            height = float(np.clip(command['height_m'], .30, .79))
            target = {'pelvis_height_m': command['height_m']}
            plan = self.plan_request(obs, {'height': round(height, 3)})
        elif kind == 'torso_pitch':
            pitch = int(5*round(float(command['pitch_deg'])/5))
            target = {'torso_pitch_deg': command['pitch_deg']}
            plan = self.plan_request(obs, {'torso': {'pitch_deg': max(0, pitch), 'yaw_deg': 0}})
        elif kind == 'hand_pose':
            wrists, goals = {}, {}
            for side, t in command['hands'].items():
                rotation = (wrist_rotation(side, t['palm'], t['fingers']) if 'palm' in t
                            else np.asarray(now['wrists'][side]['rotation']))
                goals[side] = {'position': np.asarray(t['position'], float).tolist(), 'rotation': np.asarray(rotation).tolist()}
                wrists[side] = {**goals[side], 'duration_s': float(command['duration_s'])}
            target = {'wrists': goals}
            try:
                try:
                    plan = self.plan_request(obs, {}, wrists)
                except ValueError as slow:
                    # "... needs 4.84 s, more than twice ..." (HITCH's timing check): once more with that duration
                    # (k5/m1heft 90005: the tuck after the lift asked for 1.2 s)
                    found = re.search(r'needs ([0-9.]+) s', str(slow))
                    if not found:
                        raise
                    seconds = min(8., float(found.group(1))*1.05)
                    wrists = {side: {**w, 'duration_s': seconds} for side, w in wrists.items()}
                    plan = {**self.plan_request(obs, {}, wrists), 'duration_s_asked': float(command['duration_s'])}
            except ValueError as refused:
                # HITCH's arm reach is judged on an upright torso; lean as its agent would (k5heft 90000: the last
                # reach beside the box was refused after the body settled 4 cm back, and planned with a 10 degree lean)
                plan = None
                for lean in self.settings['reach_lean_deg']:
                    try:
                        plan = {**self.plan_request(obs, {'torso': {'pitch_deg': int(lean), 'yaw_deg': 0}}, wrists),
                                'lean_deg': int(lean)}
                        break
                    except ValueError:
                        continue
                if plan is None:
                    raise refused
        elif kind == 'hand':
            for i, side in enumerate(SIDES):
                if side in command:
                    self.hands[i] = 1. if command[side] == 'close' else 0.
            count = int(round(self.settings['hand_toggle_s']*50))
            self.queue = hold_rows(obs, heading, count, self.hands)
            target = {'hands_closed': [bool(h) for h in self.hands]}
        else:
            raise ValueError(f'The planner does not execute {kind!r}')
        return {'target': target, 'plan': plan, 'hands_closed': [bool(h) for h in self.hands]}

    def chunk(self, request):
        obs = request['observation']['oracle_history'][-1]
        if self.queue is not None and len(self.queue):
            rows, self.queue = self.queue[:CHUNK], self.queue[CHUNK:]
            return rows
        if self.move is not None and self.corrections < self.settings['move_corrections']:
            off = float(np.linalg.norm(np.subtract(obs['root_pos'][:2], self.move['root_xy'])))
            turn = abs(wrap_deg(yaw_deg(rotation_wxyz(obs['root_quat']))-self.move['yaw_deg']))
            # A tracker that does not follow (SONIC stepping in place on HITCH's short walks, k3sonic) gains nothing
            # from more walking: no further correction once one did not reduce the error by a fifth.
            stalled = self.last_off is not None and off > .8*self.last_off
            self.last_off = off
            if (off > self.settings['walk_tolerance_m'] or turn > self.settings['turn_tolerance_deg']) and not stalled:
                self.corrections += 1
                try:
                    self.plan_walk(obs, self.move)
                except ValueError:
                    return None
                rows, self.queue = self.queue[:CHUNK], self.queue[CHUNK:]
                return rows
        return None

    def hold(self, obs, count):
        return hold_rows(obs, self.heading, count, self.hands)

    # ---- HITCH ---------------------------------------------------------------------------
    def frame(self, obs):
        """The plan frame O: on the ground below the measured root, along the reference's heading. HITCH's session
        plans from the reference tail, which its controller keeps on the robot; Arena's integrator does not: its
        absolute position drifts from the robot while the tracker follows the reference's motion (k4heft 90005:
        robot at x -0.76, reference at -1.26, and HITCH judged the box out of reach from the reference). So the plan
        starts from the reference posture placed at the measured root; its heading stays the reference's, so the rows
        continue the reference without a turn."""
        return [float(obs['root_pos'][0]), float(obs['root_pos'][1]), heading_rad(obs['reference_root_quat'])]

    def plan_walk(self, obs, target):
        """A walk of the reference by what the true root still lacks: the plan starts from the reference, which may be
        ahead of (or behind) the measured root, so the reference goes to target + (reference - measured)."""
        frame = self.frame(obs)
        c, s = np.cos(frame[2]), np.sin(frame[2])
        d = np.asarray(target['root_xy'], float)-np.asarray(obs['root_pos'][:2], float)
        local = [float(c*d[0]+s*d[1]), float(-s*d[0]+c*d[1])]
        measured_yaw = yaw_deg(rotation_wxyz(obs['root_quat']))
        turn = int(round(wrap_deg(target['yaw_deg']-measured_yaw)))
        request = {'walk': {'target': [round(v, 4) for v in local], 'turn_deg': int(np.clip(turn, -180, 180))}}
        if obs['reference_root_pos'][2] > WALK_HEIGHT_MAX:
            request['height'] = WALK_HEIGHT
        palms = self.settings['walk_palms']
        if tuple(self.torso) != (0, 0):   # HITCH walks with the torso in its standard posture (a reach may have leaned it)
            request['torso'] = {'pitch_deg': 0, 'yaw_deg': 0}
        if palms and not np.any(self.hands >= .5):
            request['hands'] = {side: {'waypoints': [{'position': [palms[0], palms[1]*(1 if side == 'left' else -1), palms[2]],
                                                      'duration_s': 1.2, 'orientation': {'palm': 'inward'}}]} for side in SIDES}
        return self.plan_request(obs, request)

    def plan_request(self, obs, request, wrists=None):
        start = to_hitch_pose([obs['root_pos'][0], obs['root_pos'][1], obs['reference_root_pos'][2]],
                              obs['reference_root_quat'], obs['reference_joints'])
        message = {'request': request, 'start_pose': start.tolist(), 'frame': self.frame(obs), 'torso': list(self.torso),
                   'hands': None, 'wrist_targets': wrists,
                   # carrying: HITCH's slower timing (k6heft 90005 lost the box in a 90 degree turn at the default one)
                   'slower': bool(self.settings['slower_when_carrying'] and 'walk' in request and np.any(self.hands >= .5))}
        reply = self.connect().plan(message)
        if not reply.get('ok') and reply.get('code') == 'PLAN_REJECTED' or reply.get('ok') and not reply.get('passed'):
            # as HITCH's own session does: once more with its slower default timing (limits.SLOWER)
            reply = self.connect().plan({**message, 'slower': True})
        self.log.append({'time': obs['time'], 'message': message, 'reply': {k: v for k, v in reply.items() if k != 'world'}})
        if not reply.get('ok'):
            details = reply.get('details')
            raise ValueError(f"HITCH refused: {reply.get('code')}: {reply.get('error')}" + (f' {json.dumps(details)}' if details else ''))
        if not reply.get('passed'):
            raise ValueError(f"HITCH checks failed: {[f.get('name') for f in reply.get('failures', [])]}")
        self.torso = tuple(reply.get('end_torso') or (0, 0))
        self.queue = world_to_rows(reply['world'], self.heading, self.hands)
        return {'hitch_request': reply.get('request', request), 'duration_s': reply.get('duration_s'),
                'frames': len(self.queue)}


def make_planner(name, **kwargs):
    """``kimodo`` (default) or ``hitch``."""
    if name in (None, 'kimodo'):
        from .planner import KimodoPlanner
        return KimodoPlanner(kwargs['checkpoint'], kwargs['text_cache'], kwargs['run'], settings=kwargs.get('settings'))
    if name == 'hitch':
        return KinematicPlanner(kwargs['run'], settings=kwargs.get('settings'))
    raise ValueError(f'Unknown planner {name!r} (kimodo, hitch)')


__all__ = ['KinematicPlanner', 'HitchClient', 'world_to_rows', 'to_hitch_pose', 'make_planner', 'HITCH_JOINTS']
