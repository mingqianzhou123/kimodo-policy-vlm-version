"""Legs from recorded gait cycles instead of Kimodo: one stride of clean motion capture, looped.

A cycle is one stride cut from a recorded walking segment (``demo_walks``) between two frames
where both feet are down and the pose nearly repeats; what is left of the mismatch is spread
over the cycle so it loops without a jump. Its net turn is removed unless it is a turning cycle.

``ClipWalker`` sends cycles one after another, closing the loop only at cycle boundaries: from the
measured root it picks the direction that most reduces the remaining distance (forward, backward,
left, right, or a turn until the heading is reached) and stops, blending back to the standing pose,
once the next cycle would carry the body past the goal. How far a cycle carries the body is
learned from the cycles already walked (the tracker covers only part of a reference's distance).
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from kimodo_vla.core.types import yaw_of
from .demo_walks import RATE, g1_fk, heading_frame_motion, integrate_root, load_segment, reference_feet, rot6d_rows, CONTACT_M

LEGS_AND_WAIST = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 17, 18)  # hips, waist, knees, ankles (G1_JOINT_NAMES order)
DIRECTION_UNITS = {'forward': (1., 0.), 'backward': (-1., 0.), 'left': (0., 1.), 'right': (0., -1.)}
FIRST_GUESS = .6       # share of a cycle's reference advance the tracker is assumed to cover before it is measured
START_S, SWITCH_S, STOP_S, SETTLE_S = 1., .4, 1., .5


def rz(deg):
    return Rotation.from_euler('z', np.radians(deg)).as_matrix()


def find_cycle(actions, begin, end, fk=None, min_s=.7, max_s=1.6, direction=None, min_advance_m=.25, max_side=.35,
               min_turn_deg=30.):
    """The best looping stride ``(i, j)`` in ``actions[begin:end+1]``: both feet down at both ends, pose nearly repeating.

    With ``direction`` the stride must also go that way: at least ``min_advance_m`` along it and no
    more than ``max_side`` of that sideways, or for 'turn' at least ``min_turn_deg``.
    """
    a = np.asarray(actions, dtype=float)[begin:end+1]
    feet = reference_feet(a, fk)
    down = feet[..., 2]-np.percentile(feet[..., 2], 5, axis=0) < CONTACT_M
    both = np.flatnonzero(down.all(axis=1))
    rot = rot6d_rows(a)
    tilt = np.stack([r[:, 2] for r in rot])           # body up axis: pitch and roll, not heading
    root = integrate_root(a, rot)
    best, pair = np.inf, None
    for i in both:
        for j in both[(both >= i+min_s*RATE) & (both <= i+max_s*RATE)]:
            if direction is not None:
                d, turn = heading_frame_motion(root, rot, i, j)
                if direction == 'turn':
                    if abs(turn) < min_turn_deg:
                        continue
                else:
                    unit = np.array(DIRECTION_UNITS[direction])
                    along, side = d @ unit, abs(unit[0]*d[1]-unit[1]*d[0])
                    if along < min_advance_m or side > max_side*along:
                        continue
            cost = (np.linalg.norm(a[i, 9:38][list(LEGS_AND_WAIST)]-a[j, 9:38][list(LEGS_AND_WAIST)])
                    + 5*abs(a[i, 2]-a[j, 2]) + 2*np.linalg.norm(tilt[i]-tilt[j]))
            if cost < best:
                best, pair = cost, (int(i), int(j))
    if pair is None:
        return None
    return begin+pair[0], begin+pair[1], float(best)


def make_cycle(actions, i, j, turning):
    """Frames i..j-1 as a loop: rotations relative to the start heading, mismatch spread over the cycle."""
    a = np.asarray(actions, dtype=float)[i:j+1].copy()
    n = j-i
    u = np.arange(n+1)/n
    rot = rot6d_rows(a)
    yaw = np.degrees(np.unwrap(yaw_of(rot)))
    rel = np.einsum('ij,tjk->tik', rz(-yaw[0]), rot)
    if not turning:   # no net turn: take the heading change out linearly
        rel = np.einsum('tij,tjk->tik', np.stack([rz(-(yaw[-1]-yaw[0])*x) for x in u]), rel)
    a[:, 9:38] += (a[0, 9:38]-a[-1, 9:38])[None]*u[:, None]
    a[:, 2] += (a[0, 2]-a[-1, 2])*u
    a[:, 3:9] = rel[:, :, :2].reshape(n+1, 6)
    a = a[:-1]
    root = integrate_root(a)
    d, turn = heading_frame_motion(root, rot6d_rows(a), 0, n-1)
    return {'actions': a.tolist(), 'advance_m': d.tolist(), 'yaw_deg': turn if turning else 0., 'seconds': n/RATE}


def build_library(segment_paths):
    """Cycles by direction ('forward', 'backward', 'left', 'right', 'turn-left', 'turn-right') from segment files."""
    fk = g1_fk()
    out = {}
    for path in segment_paths:
        seg = load_segment(path)
        a = np.asarray(seg['actions'], dtype=float)
        b, e = seg['window']
        found = find_cycle(a, b, e, fk, direction=seg['direction'])
        if found is None:
            continue
        i, j, cost = found
        cycle = make_cycle(a, i, j, seg['direction'] == 'turn')
        name = seg['direction'] if seg['direction'] != 'turn' else ('turn-left' if cycle['yaw_deg'] > 0 else 'turn-right')
        cycle.update(source=seg['name'], frames=[i, j], loop_cost=cost)
        out.setdefault(name, []).append(cycle)
    return out


def blend(a, b, n):
    """n rows easing from action row a to action row b (no root steps)."""
    u = np.arange(1, n+1)/n
    w = u**3*(10+u*(-15+6*u))
    ra, rb = rot6d_rows(np.stack([a, b]))
    rot = Slerp([0., 1.], Rotation.from_matrix(np.stack([ra, rb])))(w).as_matrix()
    out = np.zeros((n, 40))
    out[:, 2] = a[2]+w*(b[2]-a[2])
    out[:, 3:9] = rot[:, :, :2].reshape(n, 6)
    out[:, 9:38] = a[9:38][None]+w[:, None]*(b[9:38]-a[9:38])[None]
    return out


class ClipWalker:
    """Sends looped gait cycles toward a root goal; see the module docstring.

    Everything is in the episode's start frame (x forward, y left, the heading actions are relative
    to). ``stance`` is the standing action row the walk starts from and ends in.
    """
    def __init__(self, library, goal_xy, goal_yaw_deg, stance, variant=0, tolerance_m=.10, yaw_tolerance_deg=8.):
        self.library = library
        self.goal_xy, self.goal_yaw = np.asarray(goal_xy, dtype=float), float(goal_yaw_deg)
        self.stance = np.asarray(stance, dtype=float)
        self.variant, self.tolerance, self.yaw_tolerance = int(variant), tolerance_m, yaw_tolerance_deg
        self.heading = 0.                      # commanded heading, degrees
        self.queue, self.last, self.finished = [], self.stance.copy(), False
        self.current = None                    # name of the cycle being walked
        self.expected = {}                     # measured advance per cycle, by name
        self.mark = None                       # (cycle name, measured xy, measured yaw) when it was queued
        self.log = []

    def cycle(self, name):
        cycles = self.library[name]
        return cycles[self.variant % len(cycles)]

    def heading_rows(self, rows, heading):
        rel = rot6d_rows(rows)
        out = np.array(rows, dtype=float)
        out[:, 3:9] = np.einsum('ij,tjk->tik', rz(heading), rel)[:, :, :2].reshape(len(rows), 6)
        return out

    def choose(self, xy, yaw):
        """The next cycle's name, or None to stop."""
        if self.goal_yaw:
            left = self.goal_yaw-yaw
            if abs(left) > self.yaw_tolerance:
                name = 'turn-left' if left > 0 else 'turn-right'
                if name in self.library:
                    step = abs(self.expected.get(name, FIRST_GUESS*self.cycle(name)['yaw_deg']))
                    return name if abs(left) > step/2 else None
        r = self.goal_xy-xy
        if np.linalg.norm(r) <= self.tolerance:
            return None
        local = rz(-self.heading)[:2, :2] @ r
        name = max(DIRECTION_UNITS, key=lambda k: local @ np.array(DIRECTION_UNITS[k]) if k in self.library else -np.inf)
        along = float(local @ np.array(DIRECTION_UNITS[name]))
        ref = float(np.array(self.cycle(name)['advance_m']) @ np.array(DIRECTION_UNITS[name]))
        step = self.expected.get(name, FIRST_GUESS*ref)
        return name if along > step/2 else None

    def plan(self, xy, yaw, time):
        """Queue the next cycle (or the stop) given the body's predicted position when the queue runs out."""
        if self.mark is not None:                  # what the last cycle covered (both ends predicted the same way)
            name, xy0, yaw0 = self.mark
            if name.startswith('turn'):
                self.expected[name] = yaw-yaw0
            else:
                moved = float(rz(-self.heading)[:2, :2] @ (xy-xy0) @ np.array(DIRECTION_UNITS[name]))
                if moved > .02:
                    self.expected[name] = moved
        name = self.choose(xy, yaw)
        self.log.append({'time': time, 'xy': xy.tolist(), 'yaw_deg': yaw, 'next': name, 'expected': dict(self.expected)})
        if name is None:
            stand = self.heading_rows(self.stance[None], self.heading)[0]
            self.queue += list(blend(self.last, stand, int(STOP_S*RATE)))+[stand]*int(SETTLE_S*RATE)
            self.finished, self.mark, self.last = True, None, stand
            return
        cycle = self.heading_rows(np.array(self.cycle(name)['actions']), self.heading)
        if name != self.current:
            self.queue += list(blend(self.last, cycle[0], int((START_S if self.current is None else SWITCH_S)*RATE)))
        self.queue += list(cycle)
        self.last, self.current, self.mark = cycle[-1], name, (name, xy.copy(), yaw)
        self.heading += self.cycle(name)['yaw_deg']

    def chunk(self, xy, yaw, time, size=25):
        """Up to ``size`` action rows; ``xy``/``yaw`` are the measured root in the start frame."""
        while len(self.queue) < size and not self.finished:
            # Decide from where the body will be once the rows already queued have been walked.
            ahead_xy, ahead_yaw = np.asarray(xy, dtype=float), float(yaw)
            if self.current is not None and self.queue:
                share = len(self.queue)/len(self.cycle(self.current)['actions'])
                if self.current.startswith('turn'):
                    ahead_yaw += share*self.expected.get(self.current, self.cycle(self.current)['yaw_deg'])
                else:
                    unit = np.array(DIRECTION_UNITS[self.current])
                    ref = float(np.array(self.cycle(self.current)['advance_m']) @ unit)
                    ahead_xy = ahead_xy+share*self.expected.get(self.current, FIRST_GUESS*ref)*(rz(self.heading)[:2, :2] @ unit)
            self.plan(ahead_xy, ahead_yaw, time)
        rows, self.queue = self.queue[:size], self.queue[size:]
        return np.array(rows if rows else [self.last])

    @property
    def done(self):
        return self.finished and not self.queue


def load_library(path):
    library = json.loads(Path(path).read_text())
    if not library or not all(isinstance(v, list) and v for v in library.values()):
        raise ValueError(f'{path}: expected cycles by direction')
    return library
