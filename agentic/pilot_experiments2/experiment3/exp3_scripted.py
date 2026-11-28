"""Scripted commanders of experiment 3: no VLM, commands computed from the ground-truth observation.

``SmokeScript`` issues one of every command type and stops; it checks the planner end to end and
records the scene. ``TaskScript`` is the harness's acceptance test: a fixed rule per task that
should complete it on development layouts if the loop, the planner and the tracker work. Each
``next(observation)`` returns one command dict, or None when the script is finished. The rules
reuse the box grasp geometry the first pilot batch validated (``kimodo_vla/oracle/grasp_policy``
and ``agentic/configs/grasp-loop.json``); nothing here was tuned on development outcomes beyond
what the smoke runs show about the scene.
"""
import copy
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

    def __init__(self, task, stop_after=None, stance_closer_m=0., min_clearance=False, clearance_m=None,
                 place_by_walk=False, tuck_m=0., place_depth_m=None, place_clearance_m=.02, place_end_margin_m=.15,
                 approach_gap_m=None, true_shelf=False, carry_turn_max_deg=None, loosen_before_open_m=None,
                 loosen_clear_m=None, loosen_back_m=None, loosen_min_gap_m=None, loosen_brace_m=None,
                 loosen_probe_m=None):
        """``stop_after`` (P&PBox, staged development runs): 'hold' ends after lifting and holding the box 2 s,
        'carry' after carrying it to the shelf front; None runs the whole task.

        ``stance_closer_m`` / ``min_clearance`` (P&PBox, for the kinematic planner): stand this much closer to the box
        than the first batch's stance rule, never closer to the table edge than its smallest validated clearance
        (stance_min_clearance_m + stance_extra_m, 0.17 m) instead of its stance (0.24 m). HITCH plans palms with its
        own arm-reach model: from the first batch's stance it refused the reach beside the box ("farther from its
        shoulder than the arm reaches") and suggested 4-20 cm closer (k1heft 90000, 90002, 2026-11-28); its walk keeps
        the arms at their rest posture, so the hanging fingers no longer set the distance."""
        if task not in ('pp_box', 'sit_sofa', 'vision_navi') or stop_after not in (None, 'hold', 'carry'):
            raise ValueError((task, stop_after))
        self.task, self.stop_after = task, stop_after
        self.stance_closer_m, self.min_clearance = float(stance_closer_m), bool(min_clearance)
        # an explicit pelvis-to-table-edge clearance overrides both (k5heft: 0.12 m, below the validated range)
        self.clearance_m = None if clearance_m is None else float(clearance_m)
        # place: lower the box in front of the shelf and walk it in, instead of pushing it in with the hands
        self.place_by_walk = bool(place_by_walk)
        # after lifting, the wrists come this much toward the body before the walk back: the box was carried 17 cm in
        # front of the wrists and slid out sideways in a 90 degree turn (k6heft 90005)
        self.tuck_m = float(tuck_m)
        # the box bottom this far above the target board while it goes in: at 0.02 m the grip's sag left it level with the
        # board, its front caught the board's edge and it was left propped and slid off after release (m3heft 90002); the
        # box also sags 4-7 cm in the grip while lowered, so at 0.05 it still struck the edge (m4heft 90000, 90004)
        self.place_clearance_m = float(place_clearance_m)
        # the box centre stays this much plus half the box from either end of the shelf, along its face: a box carried
        # turned 45 degrees reaches 0.15 m out, and at 0.15 + 0.105 its corner struck the end post (m5heft 90004)
        self.place_end_margin_m = float(place_end_margin_m)
        # with place_by_walk: how far short of the shelf's face the carried box stops before it is lowered (class
        # default 0.05 m). Tried 0.25 m (m7heft) against the box striking what stands on the board above while carried
        # at 1.05 m (m6heft 90000): worse - the lowered box, sagging in the grip, struck the shelf front on the longer
        # walk in (280 N, 90004). Not used.
        if approach_gap_m is not None:
            self.approach_gap_m = float(approach_gap_m)
        if place_depth_m is not None:   # how far inside the shelf's face the box centre goes (class default 0.15 m)
            self.place_depth_m = float(place_depth_m)
        # Place against the shelf as it stands (its turned frame and the room under the board above, from the
        # observation's 'oriented' / 'clear_height_m'), not its axis-aligned box. Arena turns the shelf up to +-15
        # degrees per layout; the axis-aligned face is then up to 0.25 m off at the ends. With it: the box goes in
        # through the long side facing the robot, square to it; after the walk back the carried box comes down until
        # its highest corner is carry_clear_m under the board above the target (carried at 1.05 m its top corner
        # reached 1.16-1.20 m and struck that board's underside at 1.10 m, m6heft 90000); and before it is walked in,
        # its lowest corner (the box sits tilted 8-18 degrees in the grip) is put place_clearance_m above the target
        # board and checked again after the lowering.
        self.true_shelf = bool(true_shelf)
        # Turns while carrying, at most this many degrees per command: the box slipped out at the end of one 89
        # degree turn (f1heft 90008, grip 7-17 N per hand, gone as the turn stopped).
        self.carry_turn_max_deg = None if carry_turn_max_deg is None else float(carry_turn_max_deg)
        # With the box set down on the board, before the hands open: each wrist this far outside where it is measured,
        # fists still closed, so the palms stop pressing. Opening the fingers while the wrists were still commanded 3.5 cm
        # inside the box let the palms' squeeze pop it out toward the back of the shelf, the wrists closing 5-8 cm
        # behind it (7 of the first 60 official layouts thrown off the back; 2026-11-28). The box weighs about 50 g
        # (0.49 N on the board): at 2 cm the opening fingers still flicked it 0.21 m inward (r1heft 90000), so the fists
        # go far enough out that the opening fingers clear it.
        self.loosen_before_open_m = None if loosen_before_open_m is None else float(loosen_before_open_m)
        # ... and further if needed, until each wrist is at least this far outside the box's side along the line
        # between the wrists (from the box's pose): the set-down box is not always centred between the hands, and
        # a wrist still 2-3 cm inside its side flicked it 0.17-0.2 m inward on opening (a1heft 90017, 90022).
        self.loosen_clear_m = None if loosen_clear_m is None else float(loosen_clear_m)
        # After that, checked against the box's pose: a wrist still closer than loosen_min_gap_m to the box's side
        # goes out to loosen_clear_m again and loosen_back_m back toward the pelvis, before the hands open. Going only
        # outward, a closed fist dragged the box 7 of its 9.5 cm along (which part of the hand holds it is not in the
        # logs), was still against it, and the opening flung it off the back (r3heft 90022). Both hands always going
        # back as well (r4heft) pressed a hand that could not reach its target against something and left it 2 cm from
        # the box, which then got flicked 8 cm in (90029), so only the hand that needs it goes back.
        self.loosen_back_m = None if loosen_back_m is None else float(loosen_back_m)
        self.loosen_min_gap_m = None if loosen_min_gap_m is None else float(loosen_min_gap_m)
        # With loosen_brace_m, a wrist the box came along with is not taken further: the other fist, still closed,
        # comes back against the box's other side (this far inside it, as when holding), the stuck hand opens alone
        # and goes out, then the other fist goes out before it opens. Taken further out (r5heft) or back (r4heft),
        # the stuck fist dragged the box 9-12 cm, and opening it threw the box 0.19 m inward in 0.3 s (r3heft 90022).
        self.loosen_brace_m = None if loosen_brace_m is None else float(loosen_brace_m)
        # With loosen_probe_m, the loosening starts with the closed fists only this far out. If the box came along
        # with one wrist (at least half as far), both fists go back where they were, so the other one still holds
        # the box, and the stuck hand opens alone and goes out; then the other fist goes out and opens. Otherwise the
        # loosening goes on to where it would have gone without the probe. A brace from the other side (h1heft) did
        # not get there: the other arm stopped 6 cm short, and opening the stuck hand threw the box as before.
        self.loosen_probe_m = None if loosen_probe_m is None else float(loosen_probe_m)
        self._before_probe = None
        self.queue, self.stage, self.tries = [], 'start', 0
        config = json.loads(GRASP_CONFIG.read_text())
        self.grasp, self.wide = config['policy'], config['wide']

    def next(self, observation):
        last = observation.get('last_command') or {}
        if self.task == 'pp_box' and last.get('status') == 'planner_refused' and last.get('command', {}).get('type') == 'hand_pose':
            # A reach the planner cannot carry out: stop rather than close the hands on nothing.
            self.stage, self.queue = 'stop', []
            return None
        if not self.queue:
            self.plan(observation)
        while self.queue:
            item = self.queue.pop(0)
            item = item(observation) if callable(item) else item
            if item is not None:   # a step computed when it is due may find nothing to do
                return item
            if not self.queue:
                self.plan(observation)
        return None

    # ---- P&PBox -------------------------------------------------------------------------
    def box_stance(self, box_pos):
        """The walk-to-the-box stance of the first batch (stance rule moved back and sideways, ``wide``)."""
        shift = np.array([-self.wide.get('stance_extra_m', 0.), -self.wide.get('box_lateral_m', 0.)])
        return stance(np.asarray(box_pos), self.grasp)+shift

    def final_stance(self, observation, box, raised=False):
        """The first batch's stance, never closer to the table edge than the stance it validated (stance_clearance_m
        plus the wide setting's stance_extra_m, 0.24 m; 1 cm tolerance). In development episodes 90000 and 90002
        reaches started 17-21 cm from the edge and the hanging fingers caught it.

        ``raised``: the hands are already above the box, so the hanging fingers are no longer the limit; the
        clearance is the first batch's smallest validated one (stance_min_clearance_m) plus stance_extra_m,
        0.17 m. The walk ends with the pelvis about 8 cm ahead of the feet and the body settles back over them,
        and reaching forward moves the hips back another 5 cm: in b1-90000 / 90002 (2026-11-27) a step-in to
        0.25 m left the pelvis 0.30-0.36 m from the edge and the wrists 15-27 cm short of the box."""
        table = obj(observation, 'table')
        raised = raised or self.min_clearance
        clearance_m = self.grasp['stance_min_clearance_m'] if raised else self.grasp['stance_clearance_m']-.01
        closest = table['min'][0]-(clearance_m+self.wide.get('stance_extra_m', 0.))
        if self.clearance_m is not None:
            closest = table['min'][0]-self.clearance_m
        final = self.box_stance(box['position'])
        final[0] += self.stance_closer_m
        final[0] = min(final[0], closest-.02)
        return final

    def walk_to(self, observation, goal_xy, tolerance=.06, max_m=2.0):
        d = to_robot(observation, goal_xy)
        if np.linalg.norm(d) <= tolerance:
            return None
        scale = min(1., max_m/np.linalg.norm(d))
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
            # No route at these clearances (development layout 92000: the only gap, between a slab's end and a crate,
            # is narrower): the rule gives up rather than walk into the obstacles, as it did with the straight line.
            self.no_route = self.route is None
            self.route = self.route or []
        while self.stage == 'walk' and self.route:
            step = self.walk_to(observation, self.route[0], .06 if len(self.route) == 1 else .2)
            if step and self.tries < 8:
                self.tries += 1
                self.queue = [step]
                return
            self.route.pop(0)
        self.queue = []

    def plan_box(self, observation):
        if self.stage == 'stop':
            self.queue = []
            return
        box = obj(observation, 'box')
        if self.stage == 'start':
            self.stage, self.tries = 'approach', 0
        if self.stage == 'approach' and not self.raise_before_walk:
            # The walk-to-the-stance of d7 (hands in the walking pose, never closer than 0.24 m to the edge), then the
            # reach from the hanging pose. See raise_before_walk for why the other variant is off.
            final = self.final_stance(observation, box)
            step = self.face(observation, 0.) or self.walk_to(observation, final)
            if step is None and observation['robot']['pelvis'][0] > final[0]+.02:
                step = {'type': 'walk', 'forward_m': float(final[0]-observation['robot']['pelvis'][0]), 'left_m': 0.}
            if step and self.tries < 4:
                self.tries += 1
                self.queue = [step]
                return
            self.stage = 'grasp'
            w = waypoints(box['position'], box['quat_wxyz'], observation['robot']['pelvis'][:2], self.grasp)
            both = lambda key, rot: {s: pose(s, w[key][i], w[rot][i]) for i, s in enumerate(('left', 'right'))}
            w['over'] = np.asarray(w['pre']).copy()
            w['over'][:, 2] += self.over_m
            self.queue = [
                {'type': 'hand_pose', 'hands': both('via', 'via_rotation'), 'duration_s': self.grasp['raise_s']},
                {'type': 'hand_pose', 'hands': both('over', 'rotation'), 'duration_s': self.grasp['advance_s']},
                {'type': 'hand_pose', 'hands': both('pre', 'rotation'), 'duration_s': 1.0},
                {'type': 'hand', 'left': 'close', 'right': 'close'},
                {'type': 'hand_pose', 'hands': both('squeeze', 'rotation'), 'duration_s': self.grasp['squeeze_s']},
                {'type': 'hand_pose', 'hands': both('lift', 'rotation'), 'duration_s': self.grasp['lift_s']},
                {'type': 'walk', 'forward_m': -0.35, 'left_m': 0.0}]
            if self.tuck_m:
                # never closer than 0.15 m in front of the pelvis: on a near stance (m2heft 90005, wrists 0.17 m ahead)
                # a full 10 cm tuck pressed the box's back face against the torso, the wrists spread from 0.30 to
                # 0.40 m apart and the box slipped out
                ahead = np.array([np.cos(heading(observation)), np.sin(heading(observation)), 0.])
                forward = float(np.dot(np.mean(np.asarray(w['lift']), axis=0)[:2]-observation['robot']['pelvis'][:2], ahead[:2]))
                w['tuck'] = np.asarray(w['lift'])-ahead*float(np.clip(forward-.15, 0., self.tuck_m))
                self.queue.insert(-1, {'type': 'hand_pose', 'hands': both('tuck', 'rotation'), 'duration_s': 1.2})
            if self.stop_after == 'hold':   # hold the lifted box 2 s, then stop
                self.queue[-1] = {'type': 'hand_pose', 'hands': both('lift', 'rotation'), 'duration_s': 2.0}
                self.stage = 'stop'
            return
        if self.stage == 'approach':
            # Face the table, then hands up and closed before walking: while it walks or stands the walking reference
            # carries both hands in front of the body at 0.72-0.79 m, the height of the table top (0.745 m), and in
            # development episode 90000 (d9) the fingers hit the table edge (up to 131 N) on every walk that ended
            # within 0.3 m of it. With closed hands the planner holds the arms still while it walks
            # (hold_arms_when_closed). One whole walk to the stance, not a stop short and a step in: on a 0.25 m step-in
            # (b2-90004) SONIC left the feet 10-14 cm behind the reference's and the pelvis settled back over them.
            step = self.face(observation, 0.)
            if step and self.tries < 2:
                self.tries += 1
                self.queue = [step]
                return
            top = box['position'][2]+self.box_half()+self.raised_above_box_m
            w = waypoints(box['position'], box['quat_wxyz'], observation['robot']['pelvis'][:2], self.grasp)
            self.queue = [
                {'type': 'hand_pose', 'duration_s': self.grasp['raise_s']+0.4, 'hands': {
                    s: pose(s, from_robot(observation, self.raised_forward_m, side*self.raised_half_width_m, top), w['rotation'][i])
                    for i, (s, side) in enumerate((('left', 1.), ('right', -1.)))}},
                {'type': 'hand', 'left': 'close', 'right': 'close'}]
            self.stage, self.tries = 'walk_in', 0
            return
        if self.stage == 'walk_in':
            step = self.walk_to(observation, self.final_stance(observation, box, raised=True), .06)
            if step and self.tries < 4:
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
            if self.true_shelf and shelf.get('oriented'):
                normal, along, inside = open_face(shelf['oriented'], observation['robot']['pelvis'][:2])
            else:
                normal, along, inside = nearest_face(shelf, observation['robot']['pelvis'][:2])
            # Stop with the held box just short of the shelf's face, then move it in with the hands: in development
            # episode 90000 (2026-11-27) walking until the box was 22 cm inside the shelf knocked it out of the hands.
            self.place_xy = inside(self.place_depth_m, self.box_half()+self.place_end_margin_m)
            # the box's front face (not its centre) stops approach_gap_m short of the shelf's face: with the centre
            # there, the front was already 5.5 cm inside, and on a layout whose shelf stands nearer (k9heft 90004,
            # face at y -3.355) the box was carried into the frame at 1.05 m
            self.approach_xy = inside(-(self.approach_gap_m+self.box_half()), self.box_half()+self.place_end_margin_m)
            self.place_z = layer['max'][2]+self.box_half()+self.place_clearance_m
            self.place_yaw = float(np.degrees(np.arctan2(-normal[1], -normal[0])))
            self.stage, self.tries = 'carry', 0
            self.carry_lowered, self.carry_walk_m = not self.true_shelf, self.carry_walk_max_m
        if self.stage == 'carry':
            yaw = np.radians(self.place_yaw)
            c, s_ = np.cos(yaw), np.sin(yaw)
            stand = np.asarray(self.approach_xy)-np.array([c*self.hold[0]-s_*self.hold[1], s_*self.hold[0]+c*self.hold[1]])
            step = self.face(observation, self.place_yaw, 6.)
            if step and self.carry_turn_max_deg:
                step['left_deg'] = float(np.clip(step['left_deg'], -self.carry_turn_max_deg, self.carry_turn_max_deg))
            if step is None and not self.carry_lowered:
                # facing the shelf, away from the table: bring the box under the board above (lowered right after the
                # walk back, the box struck the table edge in the following turn and fell, n1heft 90005)
                self.carry_lowered = True
                self.queue = [self.carry_height]
                return
            last = observation.get('last_command') or {}
            if self.true_shelf and last.get('status') == 'planner_refused' and last.get('command', {}).get('type') == 'walk':
                # HITCH refuses a long carrying walk ("exceeds the path, duration or collection budget"): shorter
                self.carry_walk_m = max(.3, self.carry_walk_m/2)
            step = step or self.walk_to(observation, stand, .08, max_m=self.carry_walk_m if self.true_shelf else 2.0)
            if step and self.tries < (6 if self.carry_turn_max_deg is None else 10):
                self.tries += 1
                self.queue = [step]
                return
            if self.stop_after == 'carry':
                self.stage, self.queue = 'stop', []
                return
            self.stage = 'place'
            delta = np.r_[np.asarray(self.place_xy)-np.asarray(box['position'][:2]), self.place_z-box['position'][2]]
            wrists = observation['robot']['wrists']
            yawv = np.array([np.cos(yaw), np.sin(yaw), 0.])
            if self.place_by_walk:
                # Lower the box in front of the shelf, then walk it in with the arms held (closed hands), open, back
                # off. Pushing it in with the hands asked HITCH for wrists 0.49 m in front of the pelvis (k7heft 90002).
                forward = float(np.dot(delta[:2], yawv[:2]))
                # The measured wrists sit wider than the squeeze commanded them to (the box holds them apart):
                # commanding them as they are released the grip (k8heft 90002: 13 N -> 0, the box fell while
                # lowered). Each target keeps the first batch's squeeze inward.
                left, right = (np.asarray(wrists[s]['position'], float) for s in ('left', 'right'))
                inward = (right-left)*[1, 1, 0]
                inward /= max(np.linalg.norm(inward), 1e-9)
                squeeze = self.grasp['squeeze_m']*inward
                lowered = {'left': left+squeeze+[0, 0, delta[2]], 'right': right-squeeze+[0, 0, delta[2]]}
                # After opening: hands out and up off the box before stepping back (walking back with the hands at
                # its sides could drag it), then stand a moment so the log shows whether the box stays.
                def release(o):
                    l, r = (np.asarray(o['robot']['wrists'][s]['position'], float) for s in ('left', 'right'))
                    out = (l-r)*[1, 1, 0]
                    out = .06*out/max(np.linalg.norm(out), 1e-9)
                    return {'type': 'hand_pose', 'duration_s': 1.0, 'hands': {'left': {'position': (l+out+[0, 0, .04]).tolist()},
                                                                              'right': {'position': (r-out+[0, 0, .04]).tolist()}}}
                first = {'type': 'hand_pose', 'duration_s': 2.0, 'hands': {s: {'position': lowered[s].tolist()} for s in ('left', 'right')}}
                if self.true_shelf and obj(observation, 'shelf_target_layer').get('clear_height_m') is not None:
                    first = self.held_move(observation, self.insert_dz(observation), 2.0)
                self.queue = [
                    first] + ([self.insert_check] if self.true_shelf else []) + [
                    {'type': 'walk', 'forward_m': max(forward, .05), 'left_m': 0.0}] + (
                    [self.set_down] if self.true_shelf else []) + (
                    [self.probe, self.probe_check] if self.loosen_probe_m is not None else []) + (
                    [self.loosen] if self.loosen_before_open_m is not None else []) + (
                    [self.loosen_check] if self.loosen_min_gap_m is not None else []) + [
                    {'type': 'hand', 'left': 'open', 'right': 'open'},
                    release,
                    {'type': 'walk', 'forward_m': -0.3, 'left_m': 0.0},
                    {'type': 'hand', 'left': 'open', 'right': 'open'},
                    {'type': 'hand', 'left': 'open', 'right': 'open'}]
                return
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
    # Hands up and closed before the walk in (no finger on the table edge: 0 hand contacts above 40 N on the approach
    # in the seven b1-b3 episodes of 2026-11-27), then down beside the box. Off: from that start the walk ended 4-12 cm
    # short and SONIC stepped or sat back 12-30 cm during the reach (wrists 15-50 cm off, reference 3-9 cm), 0 of 7
    # grasped; the d7 reach from the hanging pose stayed within 1-3 cm. Kept for the record and for a planner that
    # carries the hands differently.
    raise_before_walk = False
    raised_above_box_m = .08   # before the walk in the wrists go this far above the box top,
    raised_forward_m = .12     # this far in front of the pelvis,
    raised_half_width_m = .22  # and this far to either side of it
    approach_gap_m = .05  # the carried box stops this far short of the shelf's face
    place_depth_m = .15   # and the hands put its centre this far inside it (Arena needs at least half the box, 0.105)

    def box_half(self):
        return 0.105

    carry_clear_m = .05     # the carried box's highest corner stays this far under the board above the target
    carry_walk_max_m = .9   # carrying walks at most this long per command (HITCH refused 1.6-1.9 m sideways ones)
    # before the hands open, the box's lowest corner comes down to this far above the board: at 0.02 m (resting 3.6 cm
    # up, tilted) the opening fists flung it 1.5 m through the shelf (n2heft 90000); at 0 it sits on the board
    release_above_m = 0.
    hold_extra_m = .01      # moving the held box, each wrist this much further inward than the first batch's squeeze
    insert_min_m = .05      # before walking in, the lowest corner at least this far above the target board

    def held_move(self, observation, dz, duration_s=None):
        """Move the held box up or down by ``dz`` with the measured wrists, each target keeping the first batch's
        squeeze inward (commanding the wrists as measured releases the grip, k8heft 90002)."""
        wrists = observation['robot']['wrists']
        left, right = (np.asarray(wrists[s]['position'], float) for s in ('left', 'right'))
        inward = (right-left)*[1, 1, 0]
        inward /= max(np.linalg.norm(inward), 1e-9)
        # 7-12 N per hand while carried; the box slipped out as the first walk after lowering started (n2heft 90008)
        squeeze = (self.grasp['squeeze_m']+self.hold_extra_m)*inward
        duration = duration_s if duration_s is not None else max(1.0, abs(dz)/.1)
        return {'type': 'hand_pose', 'duration_s': float(duration),
                'hands': {'left': {'position': (left+squeeze+[0, 0, dz]).tolist()},
                          'right': {'position': (right-squeeze+[0, 0, dz]).tolist()}}}

    def window(self, observation):
        """Target board top and the highest the carried box may reach under the board above (None if unknown)."""
        layer = obj(observation, 'shelf_target_layer')
        top = layer['max'][2]
        clear = layer.get('clear_height_m')
        return top, (None if clear is None else top+clear-self.carry_clear_m)

    def carry_height(self, observation):
        """After the walk back: bring the box down until its highest corner clears the board above the target."""
        _, ceiling = self.window(observation)
        low, high = corner_heights(obj(observation, 'box'), self.box_half())
        if ceiling is None or high <= ceiling:
            return None
        return self.held_move(observation, ceiling-high)

    def insert_dz(self, observation):
        """How far to move the box so its lowest corner is place_clearance_m above the target board, without its
        highest corner rising above the ceiling (if both cannot hold, the middle of the room)."""
        top, ceiling = self.window(observation)
        low, high = corner_heights(obj(observation, 'box'), self.box_half())
        dz = top+self.place_clearance_m-low
        if ceiling is not None and high+dz > ceiling:
            dz = (top+ceiling)/2-(low+high)/2
        return float(dz)

    def set_down(self, observation):
        """Inside the shelf, before opening: the box down onto the board. Opening the hands pushes the box (53 N on
        each fist); released 12 cm above the board it was thrown off the back of the shelf (n1heft 90000, 90002)."""
        top, _ = self.window(observation)
        low, _ = corner_heights(obj(observation, 'box'), self.box_half())
        dz = top+self.release_above_m-low
        return self.held_move(observation, dz, 1.5) if dz < -.01 else None

    def box_gaps(self, observation):
        """Line between the wrists (outward for the left wrist) and how far each wrist is outside the box's side."""
        wrists = observation['robot']['wrists']
        left, right = (np.asarray(wrists[s]['position'], float) for s in ('left', 'right'))
        out = (left-right)*[1, 1, 0]
        out = out/max(np.linalg.norm(out), 1e-9)
        box = obj(observation, 'box')
        centre = np.asarray(box['position'], float)
        reach = box_reach(box, self.box_half(), out)
        gaps = {side: sign*float(np.dot(wrist-centre, out))-reach
                for side, wrist, sign in (('left', left, 1.), ('right', right, -1.))}
        return left, right, out, gaps

    def back_toward_pelvis(self, observation, left, right, out):
        toward = (np.asarray(observation['robot']['pelvis'], float)-(left+right)/2)*[1, 1, 0]
        toward -= np.dot(toward, out)*out
        return self.loosen_back_m*toward/max(np.linalg.norm(toward), 1e-9)

    def loosen(self, observation):
        """Before the hands open on the set-down box: each wrist loosen_before_open_m outside its measured position,
        or further until it is loosen_clear_m outside the box's side, hands still closed, so the palms no longer
        squeeze it and the opening fingers clear it. Without loosen_min_gap_m, both hands also go loosen_back_m back
        toward the pelvis (r4heft)."""
        if self._before_probe is not None:   # from where the fists were before the probe, as without it
            observation = self._before_probe
        wrists = observation['robot']['wrists']
        left, right = (np.asarray(wrists[s]['position'], float) for s in ('left', 'right'))
        out = (left-right)*[1, 1, 0]
        out = out/max(np.linalg.norm(out), 1e-9)
        shift = {'left': self.loosen_before_open_m, 'right': self.loosen_before_open_m}
        if self.loosen_clear_m is not None:
            _, _, _, gaps = self.box_gaps(observation)
            for side in ('left', 'right'):
                shift[side] = max(shift[side], self.loosen_clear_m-gaps[side])
        back = np.zeros(3)
        if self.loosen_back_m and self.loosen_min_gap_m is None:
            back = self.back_toward_pelvis(observation, left, right, out)
        return {'type': 'hand_pose', 'duration_s': 0.8,
                'hands': {'left': {'position': (left+shift['left']*out+back).tolist()},
                          'right': {'position': (right-shift['right']*out+back).tolist()}}}

    def loosen_check(self, observation):
        """After the loosening: a wrist still closer than loosen_min_gap_m to the box's side (the box came along with
        it, or the hand was stopped) goes out to loosen_clear_m and loosen_back_m back toward the pelvis; None if
        both are clear."""
        left, right, out, gaps = self.box_gaps(observation)
        if min(gaps.values()) >= self.loosen_min_gap_m:
            return None
        stuck = [side for side in ('left', 'right') if gaps[side] < self.loosen_min_gap_m]
        if self.loosen_brace_m is not None and len(stuck) == 1:
            return self.brace_release(stuck[0], left, right, out, gaps)
        back = self.back_toward_pelvis(observation, left, right, out) if self.loosen_back_m else np.zeros(3)
        target = {'left': left, 'right': right}
        for side, sign in (('left', 1.), ('right', -1.)):
            if gaps[side] < self.loosen_min_gap_m:
                target[side] = target[side]+sign*max(0., (self.loosen_clear_m or 0.)-gaps[side])*out+back
        return {'type': 'hand_pose', 'duration_s': 0.8,
                'hands': {s: {'position': target[s].tolist()} for s in ('left', 'right')}}

    def brace_release(self, stuck, left, right, out, gaps):
        """loosen_brace_m: the free fist back against the box's other side, then the stuck hand opens alone and goes
        out to loosen_clear_m (and 4 cm up), then the free fist goes out to loosen_clear_m; the usual opening follows."""
        free = 'right' if stuck == 'left' else 'left'
        sign = {'left': 1., 'right': -1.}
        now = {'left': left, 'right': right}
        brace = now[free]-sign[free]*(gaps[free]+self.loosen_brace_m)*out
        self.queue[0:0] = [{'type': 'hand', stuck: 'open'}, self.away(stuck, .04), self.away(free, 0.)]
        return {'type': 'hand_pose', 'duration_s': 0.8,
                'hands': {stuck: {'position': now[stuck].tolist()}, free: {'position': brace.tolist()}}}

    def away(self, side, up):
        """A step, computed when due: this wrist out until it is loosen_clear_m outside the box's side, and up by
        ``up``; the other wrist stays where it is."""
        sign = {'left': 1., 'right': -1.}

        def step(o):
            l, r, o_out, g = self.box_gaps(o)
            target = {'left': l, 'right': r}
            target[side] = target[side]+sign[side]*max(0., self.loosen_clear_m-g[side])*o_out+[0, 0, up]
            return {'type': 'hand_pose', 'duration_s': 0.8,
                    'hands': {s: {'position': target[s].tolist()} for s in ('left', 'right')}}
        return step

    def probe(self, observation):
        """loosen_probe_m: both closed fists this far out along the line between the wrists (0.5 s, the shortest a
        command may take)."""
        self._before_probe = copy.deepcopy({'robot': {'wrists': observation['robot']['wrists']},
                                            'objects': observation['objects']})
        left, right, out, _ = self.box_gaps(observation)
        return {'type': 'hand_pose', 'duration_s': 0.5,
                'hands': {'left': {'position': (left+self.loosen_probe_m*out).tolist()},
                          'right': {'position': (right-self.loosen_probe_m*out).tolist()}}}

    def probe_check(self, observation):
        """After the probe: if the box came along with exactly one wrist (at least half the probe), both fists go back
        where they were, the stuck hand opens alone and goes out (and 4 cm up), then the other fist goes out, instead
        of the plain loosening; None otherwise (the loosening goes on)."""
        before = self._before_probe
        left, right, out, _ = self.box_gaps(before)
        moved = (np.asarray(obj(observation, 'box')['position'], float)-np.asarray(obj(before, 'box')['position'], float))*[1, 1, 0]
        follow = {'left': float(np.dot(moved, out)), 'right': -float(np.dot(moved, out))}
        stuck = [side for side in ('left', 'right') if follow[side] >= self.loosen_probe_m/2]
        if len(stuck) != 1:
            return None
        free = 'right' if stuck[0] == 'left' else 'left'
        self.queue = [{'type': 'hand', stuck[0]: 'open'}, self.away(stuck[0], .04), self.away(free, 0.)] + [
            item for item in self.queue if item != self.loosen]
        return {'type': 'hand_pose', 'duration_s': 0.5,
                'hands': {'left': {'position': left.tolist()}, 'right': {'position': right.tolist()}}}

    def insert_check(self, observation):
        """After the lowering (the box sags 4-7 cm in the grip): put it right again if a corner is out of the room."""
        top, ceiling = self.window(observation)
        low, high = corner_heights(obj(observation, 'box'), self.box_half())
        if low >= top+self.insert_min_m and (ceiling is None or high <= ceiling):
            return None
        return self.held_move(observation, self.insert_dz(observation))

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
            self.route = route(pelvis, self.stand, obstacles, near=sofa, room=floor_area(observation)) or [list(self.stand)]
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


def open_face(oriented, point):
    """The long side of a turned shelf that faces ``point``, as ``nearest_face`` gives it: (outward normal, unit
    vector along it, inside(depth, margin, u)). The shelf asset's ends are braced at every height and its long
    sides are open, so boxes go in through a long side. ``oriented``: centre, yaw_deg (of the long axis), size."""
    centre = np.asarray(oriented['center'][:2], dtype=float)
    yaw = np.radians(oriented['yaw_deg'])
    length, depth = float(oriented['size'][0]), float(oriented['size'][1])
    axis_x, axis_y = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    if depth > length:   # the long axis is the prim's y axis
        axis_x, axis_y, length, depth = axis_y, -axis_x, depth, length
    offset = np.asarray(point, dtype=float)[:2]-centre
    normal = axis_y*(np.sign(np.dot(offset, axis_y)) or 1.)
    face = centre+normal*depth/2

    def inside(depth_m, margin, u=None):
        if u is None:
            t = float(np.clip(np.dot(offset, axis_x), -length/2+margin, length/2-margin))
        else:
            t = -length/2+u*length
        return (face-normal*depth_m+axis_x*t).tolist()
    return normal, axis_x, inside


def corner_heights(box, half):
    """Lowest and highest corner of a cube of half-size ``half`` at the box's position and orientation."""
    w, x, y, z = (float(v) for v in box['quat_wxyz'])
    row = np.array([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)])   # the world z row of the rotation
    reach = half*float(np.abs(row).sum())
    centre = float(box['position'][2])
    return centre-reach, centre+reach


def box_reach(box, half, direction):
    """How far a cube of half-size ``half`` at the box's orientation reaches from its centre along ``direction``."""
    w, x, y, z = (float(v) for v in box['quat_wxyz'])
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                         [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                         [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    d = np.asarray(direction, float)
    return half*float(np.abs(rotation.T@(d/max(np.linalg.norm(d), 1e-9))).sum())


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
    None when no route keeps those clearances."""
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
        return None
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
