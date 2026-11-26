"""Scripted commanders of experiment 3: no VLM, commands computed from the ground-truth observation.

``SmokeScript`` issues one of every command type and stops; it checks the planner end to end and
records the scene. ``TaskScript`` is the harness's acceptance test: a fixed rule per task that
should complete it on development layouts if the loop, the planner and the tracker work. Each
``next(observation)`` returns one command dict, or None when the script is finished. The rules
reuse the box grasp geometry the first pilot batch validated (``kimodo_vla/oracle/grasp_policy``
and ``agentic/configs/grasp-loop.json``); nothing here was tuned on development outcomes beyond
what the smoke runs show about the scene.
"""
import json
from pathlib import Path

import numpy as np

from kimodo_vla.agent.commands import palm_and_fingers
from kimodo_vla.oracle.grasp_policy import stance, waypoints

GRASP_CONFIG = Path(__file__).resolve().parents[2]/'configs'/'grasp-loop.json'


def heading(observation):
    return np.radians(observation['robot']['heading_deg'])


def to_robot(observation, world_xy):
    """World ground offset from the pelvis, in the robot's frame: (forward, left)."""
    yaw = heading(observation)
    d = np.asarray(world_xy, dtype=float)-np.asarray(observation['robot']['pelvis'][:2])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c*d[0]+s*d[1], -s*d[0]+c*d[1]])


def from_robot(observation, forward, left, up):
    """World point at (forward, left) from the pelvis on the ground plane and ``up`` above the floor."""
    yaw = heading(observation)
    c, s = np.cos(yaw), np.sin(yaw)
    p = np.asarray(observation['robot']['pelvis'][:2])
    return [float(p[0]+c*forward-s*left), float(p[1]+s*forward+c*left), float(up)]


def obj(observation, name):
    return next((o for o in observation['objects'] if o['name'] == name), None)


def wrap(angle):
    return (angle+180.) % 360.-180.


def pose(side, position, rotation):
    palm, fingers = palm_and_fingers(side, rotation)
    return {'position': [float(x) for x in position], 'palm': palm.tolist(), 'fingers': fingers.tolist()}


class SmokeScript:
    """Walk, turn both ways, lean and straighten, lower and raise the pelvis, move one hand, close
    and open it. Every value is relative to where the robot stands, so it runs in either scene."""

    def __init__(self, task=None):
        self.step = 0

    def next(self, observation):
        r = observation['robot']
        k, self.step = self.step, self.step+1
        if k == 0:
            self.height = r['pelvis'][2]
            return {'type': 'walk', 'forward_m': 0.3, 'left_m': 0.0}
        if k == 1:
            return {'type': 'turn', 'left_deg': 30}
        if k == 2:
            return {'type': 'turn', 'left_deg': -30}
        if k == 3:
            return {'type': 'torso_pitch', 'pitch_deg': 15}
        if k == 4:
            return {'type': 'torso_pitch', 'pitch_deg': 0}
        if k == 5:
            return {'type': 'pelvis_height', 'height_m': round(self.height-0.12, 3)}
        if k == 6:
            return {'type': 'pelvis_height', 'height_m': round(self.height, 3)}
        if k == 7:
            yaw = heading(observation)
            forward, left = np.array([np.cos(yaw), np.sin(yaw), 0.]), np.array([-np.sin(yaw), np.cos(yaw), 0.])
            return {'type': 'hand_pose', 'duration_s': 1.5, 'hands': {'right': {
                'position': from_robot(observation, 0.30, -0.22, r['pelvis'][2]+0.15),
                'palm': left.tolist(), 'fingers': forward.tolist()}}}
        if k == 8:
            return {'type': 'hand', 'right': 'close'}
        if k == 9:
            return {'type': 'hand', 'right': 'open'}
        return None


class TaskScript:
    """The fixed rule for one task (``task``: 'pp_box' or 'sit_sofa')."""

    def __init__(self, task):
        if task not in ('pp_box', 'sit_sofa'):
            raise ValueError(task)
        self.task = task
        self.queue, self.stage, self.tries = [], 'start', 0
        config = json.loads(GRASP_CONFIG.read_text())
        self.grasp, self.wide = config['policy'], config['wide']

    def next(self, observation):
        if not self.queue:
            self.plan(observation)
        if not self.queue:
            return None
        item = self.queue.pop(0)
        return item(observation) if callable(item) else item

    # ---- P&PBox -------------------------------------------------------------------------
    def box_stance(self, box_pos):
        """The walk-to-the-box stance of the first batch (stance rule moved back and sideways, ``wide``)."""
        shift = np.array([-self.wide.get('stance_extra_m', 0.), -self.wide.get('box_lateral_m', 0.)])
        return stance(np.asarray(box_pos), self.grasp)+shift

    def walk_to(self, observation, goal_xy, tolerance=.06):
        d = to_robot(observation, goal_xy)
        if np.linalg.norm(d) <= tolerance:
            return None
        scale = min(1., 2.0/np.linalg.norm(d))
        return {'type': 'walk', 'forward_m': float(d[0]*scale), 'left_m': float(d[1]*scale)}

    def face(self, observation, yaw_deg, tolerance=4.):
        turn = wrap(yaw_deg-observation['robot']['heading_deg'])
        return None if abs(turn) <= tolerance else {'type': 'turn', 'left_deg': float(turn)}

    def plan(self, observation):
        if self.task == 'pp_box':
            self.plan_box(observation)
        else:
            self.plan_sofa(observation)

    def plan_box(self, observation):
        box = obj(observation, 'box')
        if self.stage == 'start':
            self.stage, self.tries = 'approach', 0
        if self.stage == 'approach':
            turn = self.face(observation, 0.)
            walk = self.walk_to(observation, self.box_stance(box['position']))
            if (turn or walk) and self.tries < 4:
                self.tries += 1
                self.queue = [turn or walk]
                return
            self.stage = 'grasp'
            quat = box['quat_wxyz']
            w = waypoints(box['position'], quat, observation['robot']['pelvis'][:2], self.grasp)
            both = lambda key, rot: {s: pose(s, w[key][i], w[rot][i]) for i, s in enumerate(('left', 'right'))}
            self.queue = [
                {'type': 'hand_pose', 'hands': both('via', 'via_rotation'), 'duration_s': self.grasp['raise_s']},
                {'type': 'hand_pose', 'hands': both('pre', 'rotation'), 'duration_s': self.grasp['advance_s']},
                {'type': 'hand', 'left': 'close', 'right': 'close'},
                {'type': 'hand_pose', 'hands': both('squeeze', 'rotation'), 'duration_s': self.grasp['squeeze_s']},
                {'type': 'hand_pose', 'hands': both('lift', 'rotation'), 'duration_s': self.grasp['lift_s']}]
            return
        if self.stage == 'grasp':
            self.stage = 'done'
            self.queue = []

    # ---- SitSofa ------------------------------------------------------------------------
    def plan_sofa(self, observation):
        self.queue = []   # written after the scene probe shows where the seat is
