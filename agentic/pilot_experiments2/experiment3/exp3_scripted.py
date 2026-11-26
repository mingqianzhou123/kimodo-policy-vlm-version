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
    """The fixed rule for one task (``task``: 'pp_box', 'sit_sofa' or 'vision_navi')."""

    def __init__(self, task):
        if task not in ('pp_box', 'sit_sofa', 'vision_navi'):
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

    def final_stance(self, observation, box):
        """The first batch's stance, never closer to the table edge than the stance it validated (stance_clearance_m
        plus the wide setting's stance_extra_m, 0.24 m; 1 cm tolerance). In development episodes 90000 and 90002
        reaches started 17-21 cm from the edge and the hanging fingers caught it."""
        table = obj(observation, 'table')
        closest = table['min'][0]-(self.grasp['stance_clearance_m']+self.wide.get('stance_extra_m', 0.)-.01)
        final = self.box_stance(box['position'])
        final[0] = min(final[0], closest-.02)
        return final

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
        elif self.task == 'vision_navi':
            self.plan_navi(observation)
        else:
            self.plan_sofa(observation)

    # ---- VisionNavi: the interface self-check -------------------------------------------
    overhead_m = 1.3   # obstacles whose underside is higher than this do not block the walk
    # Obstacles lower than this only need the legs to clear them: the walking reference carries the hands at
    # 0.72-0.79 m. VisionNavi's slabs (2.0 x 0.4 x 0.25 m) leave gaps that a 0.32 m clearance cannot use: no route
    # existed on development layouts 92000 and 92001 above 0.22 / 0.25 m. The feet are about 0.15 m from the centre line.
    low_obstacle_top_m = .45
    legs_clearance_m = .22

    def plan_navi(self, observation):
        target = obj(observation, 'target')
        goal = np.asarray(target['center'][:2])
        if self.stage == 'start':
            low = [o for o in observation['objects'] if o['name'] not in ('target', 'room_floor')
                   and o.get('min', [0, 0, 0])[2] < self.overhead_m]
            self.route, self.stage, self.tries = route(observation['robot']['pelvis'][:2], goal, low,
                                                       room=floor_area(observation), low_body=self.legs_clearance_m,
                                                       low_top_m=self.low_obstacle_top_m), 'walk', 0
        while self.stage == 'walk' and self.route:
            step = self.walk_to(observation, self.route[0], .06 if len(self.route) == 1 else .2)
            if step and self.tries < 8:
                self.tries += 1
                self.queue = [step]
                return
            self.route.pop(0)
        self.queue = []

    def plan_box(self, observation):
        box = obj(observation, 'box')
        if self.stage == 'start':
            self.stage, self.tries = 'approach', 0
        if self.stage in ('approach', 'step_in'):
            final = self.final_stance(observation, box)
        if self.stage == 'approach':
            # Stop short of the stance first. While it walks or stands the walking reference carries both hands in
            # front of the body at 0.72-0.79 m, the height of the table top (0.745 m); in development episode 90000
            # (d9) the fingers hit the table edge (up to 131 N) on every walk that ended within 0.3 m of it.
            step = self.face(observation, 0.) or self.walk_to(observation, final-[self.pre_stance_m, 0.], .08)
            if step and self.tries < 4:
                self.tries += 1
                self.queue = [step]
                return
            # Hands up above the box, then closed: with closed hands the planner holds the arms still while it walks
            # (hold_arms_when_closed), so the last step in keeps them clear of the table.
            top = box['position'][2]+self.box_half()+self.raised_above_box_m
            w = waypoints(box['position'], box['quat_wxyz'], final, self.grasp)
            lateral = to_robot(observation, box['position'][:2])[1]
            self.queue = [
                {'type': 'hand_pose', 'duration_s': self.grasp['raise_s']+0.4, 'hands': {
                    s: pose(s, from_robot(observation, self.raised_forward_m, lateral+side*self.raised_half_width_m, top), w['rotation'][i])
                    for i, (s, side) in enumerate((('left', 1.), ('right', -1.)))}},
                {'type': 'hand', 'left': 'close', 'right': 'close'}]
            self.stage, self.tries = 'step_in', 0
            return
        if self.stage == 'step_in':
            step = self.walk_to(observation, final, .06)
            if step and self.tries < 3:
                self.tries += 1
                self.queue = [step]
                return
            self.stage = 'grasp'
            w = waypoints(box['position'], box['quat_wxyz'], observation['robot']['pelvis'][:2], self.grasp)
            both = lambda key, rot: {s: pose(s, w[key][i], w[rot][i]) for i, s in enumerate(('left', 'right'))}
            # Down beside the box from above: in every development reach from the hanging pose (90000 twice, 90002,
            # 90004, 2026-11-26/27) the open fingers caught the table edge on the way to 'pre'.
            w['over'] = np.asarray(w['pre']).copy()
            w['over'][:, 2] += self.over_m
            self.queue = [
                {'type': 'hand', 'left': 'open', 'right': 'open'},
                {'type': 'hand_pose', 'hands': both('over', 'rotation'), 'duration_s': self.grasp['advance_s']},
                {'type': 'hand_pose', 'hands': both('pre', 'rotation'), 'duration_s': 1.0},
                {'type': 'hand', 'left': 'close', 'right': 'close'},
                {'type': 'hand_pose', 'hands': both('squeeze', 'rotation'), 'duration_s': self.grasp['squeeze_s']},
                {'type': 'hand_pose', 'hands': both('lift', 'rotation'), 'duration_s': self.grasp['lift_s']},
                {'type': 'walk', 'forward_m': -0.35, 'left_m': 0.0}]
            return
        if self.stage == 'grasp':
            # Where the held box sits relative to the pelvis, and where it should go on the shelf.
            self.hold = to_robot(observation, box['position'][:2])
            layer, shelf = obj(observation, 'shelf_target_layer'), obj(observation, 'shelf')
            normal, along, inside = nearest_face(shelf, observation['robot']['pelvis'][:2])
            # Stop with the held box just short of the shelf's face, then move it in with the hands: in development
            # episode 90000 (2026-11-27) walking until the box was 22 cm inside the shelf knocked it out of the hands.
            self.place_xy = inside(self.place_depth_m, self.box_half()+0.15)
            self.approach_xy = inside(-self.approach_gap_m, self.box_half()+0.15)
            self.place_z = layer['max'][2]+self.box_half()+0.02
            self.place_yaw = float(np.degrees(np.arctan2(-normal[1], -normal[0])))
            self.stage, self.tries = 'carry', 0
        if self.stage == 'carry':
            yaw = np.radians(self.place_yaw)
            c, s_ = np.cos(yaw), np.sin(yaw)
            stand = np.asarray(self.approach_xy)-np.array([c*self.hold[0]-s_*self.hold[1], s_*self.hold[0]+c*self.hold[1]])
            step = self.face(observation, self.place_yaw, 6.) or self.walk_to(observation, stand, .08)
            if step and self.tries < 6:
                self.tries += 1
                self.queue = [step]
                return
            self.stage = 'place'
            delta = np.r_[np.asarray(self.place_xy)-np.asarray(box['position'][:2]), self.place_z-box['position'][2]]
            wrists = observation['robot']['wrists']
            yawv = np.array([np.cos(yaw), np.sin(yaw), 0.])
            self.queue = [
                {'type': 'hand_pose', 'duration_s': 2.0,
                 'hands': {s: {'position': (np.asarray(wrists[s]['position'])+delta).tolist()} for s in ('left', 'right')}},
                {'type': 'hand', 'left': 'open', 'right': 'open'},
                lambda o: {'type': 'hand_pose', 'duration_s': 1.5, 'hands': {s: {'position': (
                    np.asarray(o['robot']['wrists'][s]['position'])-0.15*yawv+[0, 0, 0.05]).tolist()} for s in ('left', 'right')}},
                {'type': 'walk', 'forward_m': -0.3, 'left_m': 0.0}]
            return
        self.queue = []

    over_m = .10          # the open hands travel this far above the grasp height before they come down beside the box
    pre_stance_m = .25    # the first walk stops this much short of the stance (hands still in the walking pose)
    raised_above_box_m = .08   # then the wrists go this far above the box top,
    raised_forward_m = .12     # this far in front of the pelvis,
    raised_half_width_m = .22  # and this far to either side of the box's line
    approach_gap_m = .05  # the carried box stops this far short of the shelf's face
    place_depth_m = .15   # and the hands put its centre this far inside it (Arena needs at least half the box, 0.105)

    def box_half(self):
        return 0.105

    # ---- SitSofa ------------------------------------------------------------------------
    def plan_sofa(self, observation):
        seat = obj(observation, 'sofa_seat')
        if self.stage == 'start':
            pelvis = observation['robot']['pelvis'][:2]
            normal, along, inside = nearest_face(seat, pelvis)
            # Everything but the seat and the sofa it belongs to (any box around the seat's centre) is in the way.
            centre = (np.asarray(seat['min'][:2])+np.asarray(seat['max'][:2]))/2
            obstacles = [o for o in observation['objects'] if o['name'] not in ('sofa_seat', 'room_floor') and clearance(centre, o) > 0]
            # The spot along the seat front with the most room in front of it, nearest the robot on ties.
            best = None
            for u in np.linspace(0.1, 0.9, 17):
                spot = np.asarray(inside(-0.10, None, u))
                room = min((clearance(spot+normal*d, o) for o in obstacles for d in (0.2, 0.5, 0.8)), default=9.)
                score = (round(min(room, 1.0), 2), -np.linalg.norm(spot-pelvis))
                if best is None or score > best[0]:
                    best = (score, spot)
            self.stand = best[1]
            self.sit_yaw = float(np.degrees(np.arctan2(normal[1], normal[0])))   # back to the seat
            sofa = [o for o in observation['objects'] if o['name'] not in ('sofa_seat', 'room_floor') and clearance(centre, o) == 0]
            self.route = route(pelvis, self.stand, obstacles, near=sofa, room=floor_area(observation))
            self.stage, self.tries = 'turn', 0
        if self.stage == 'turn':
            step = self.face(observation, self.sit_yaw, 6.)
            if step and self.tries < 3:
                self.tries += 1
                self.queue = [step]
                return
            self.stage, self.tries = 'walk', 0
        if self.stage == 'walk':
            while self.route:
                step = self.walk_to(observation, self.route[0], .08 if len(self.route) == 1 else .2)
                if step and self.tries < 8:
                    self.tries += 1
                    self.queue = [step]
                    return
                self.route.pop(0)
            self.stage = 'sit'
            step = self.face(observation, self.sit_yaw, 8.)
            self.queue = ([step] if step else [])+[{'type': 'pelvis_height', 'height_m': 0.48, 'behind_feet_m': 0.25}]
            return
        self.queue = []


def nearest_face(region, point):
    """The side of an axis-aligned region that faces ``point``: (outward normal, unit vector along it,
    inside(depth, margin, u)) where inside gives the ground point ``depth`` inside the face (negative:
    outside), at fraction ``u`` along it, or (u None) at the point's own place clamped ``margin`` from the ends."""
    low, high = np.asarray(region['min'][:2], dtype=float), np.asarray(region['max'][:2], dtype=float)
    centre, half = (low+high)/2, (high-low)/2
    d = np.asarray(point, dtype=float)-centre
    axis = int(np.argmax(np.abs(d)/half))
    normal = np.zeros(2)
    normal[axis] = np.sign(d[axis]) or 1.
    other = 1-axis
    along = np.zeros(2)
    along[other] = 1.
    face = centre+normal*half

    def inside(depth, margin, u=None):
        p = face.copy()-normal*depth
        if u is None:
            p[other] = np.clip(point[other], low[other]+margin, high[other]-margin)
        else:
            p[other] = low[other]+u*(high[other]-low[other])
        return p.tolist()
    return normal, along, inside


def clearance(point, region):
    """Ground distance from a point to an object's footprint (0 inside): its turned rectangle when the object
    has a position, a size and a yaw, otherwise its axis-aligned box."""
    if all(k in region for k in ('position', 'size', 'yaw_deg')) and region.get('size'):
        yaw = np.radians(region['yaw_deg'])
        d = np.asarray(point, dtype=float)[:2]-np.asarray(region['position'][:2])
        local = np.array([np.cos(yaw)*d[0]+np.sin(yaw)*d[1], -np.sin(yaw)*d[0]+np.cos(yaw)*d[1]])
        return float(np.linalg.norm(np.maximum(0., np.abs(local)-np.asarray(region['size'][:2])/2)))
    if 'min' in region:
        low, high = np.asarray(region['min'][:2]), np.asarray(region['max'][:2])
    else:
        half = np.asarray(region.get('size') or [.2, .2])[:2]/2
        low, high = np.asarray(region['position'][:2])-half, np.asarray(region['position'][:2])+half
    return float(np.linalg.norm(np.maximum(0., np.maximum(low-point, point-high))))


def floor_area(observation):
    """The walkable floor (``room_floor``: the walls run along its edges), or None."""
    return next((o for o in observation['objects'] if o['name'] == 'room_floor'), None)


def route(start, goal, obstacles, body=.32, near=None, step=.1, pad=4., room=None, low_body=None, low_top_m=.45):
    """Waypoints from ``start`` to ``goal`` that keep ``body`` clear of every obstacle (``near``: obstacles
    the robot may come right up to, such as the sofa it will sit on) and of the walls (``room``: the floor
    area inside them). Obstacles whose top is below ``low_top_m`` need only ``low_body`` (the legs) when given.
    A* on a ``step`` grid, then shortened to the fewest straight legs that stay clear.
    Falls back to the straight line."""
    import heapq
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    near = near or []
    walls = lambda p: (room is not None and (np.any(np.asarray(p) < np.asarray(room['min'][:2])+body)
                                             or np.any(np.asarray(p) > np.asarray(room['max'][:2])-body)))
    radius = lambda o: low_body if low_body is not None and o.get('max', [0, 0, 9])[2] < low_top_m else body
    blocked = lambda p: (walls(p) or any(clearance(p, o) <= radius(o) for o in obstacles)
                         or any(clearance(p, o) <= .02 for o in near))

    def free(a, b):
        n = max(2, int(np.linalg.norm(b-a)/(step/2)))
        return all(not blocked(a+(b-a)*u) for u in np.linspace(0, 1, n)[1:-1])
    if free(start, goal):
        return [goal.tolist()]
    low = np.minimum(start, goal)-pad
    cell = lambda p: tuple(np.round((np.asarray(p)-low)/step).astype(int))
    point = lambda c: low+np.asarray(c)*step
    s, g = cell(start), cell(goal)
    shape = tuple(cell(np.maximum(start, goal)+pad)[i]+1 for i in range(2))
    frontier, came, cost = [(0., s)], {s: None}, {s: 0.}
    while frontier:
        _, c = heapq.heappop(frontier)
        if c == g:
            break
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                n = (c[0]+dx, c[1]+dy)
                if n == c or not (0 <= n[0] < shape[0] and 0 <= n[1] < shape[1]):
                    continue
                if n != g and blocked(point(n)):
                    continue
                new = cost[c]+step*np.hypot(dx, dy)
                if new < cost.get(n, np.inf):
                    cost[n], came[n] = new, c
                    heapq.heappush(frontier, (new+np.linalg.norm(point(n)-point(g)), n))
    if g not in came:
        return [goal.tolist()]
    path, c = [], g
    while c is not None:
        path.append(point(c))
        c = came[c]
    path = [start]+path[::-1][1:-1]+[goal]
    legs, i = [], 0
    while i < len(path)-1:
        j = next(k for k in range(len(path)-1, i, -1) if k == i+1 or free(path[i], path[k]))
        legs.append(path[j])
        i = j
    return [p.tolist() for p in legs]
