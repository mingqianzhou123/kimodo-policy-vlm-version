"""Physical grasp success and per-attempt attribution. Pure logic, no simulator imports.

Success follows the experiment plan: the box centre of mass is at least 10 cm above its
initial resting height while both hands grip it and nothing but the robot touches it,
continuously for 0.5 s. Forces come from the box contact report (see ``grasp_scene``).
"""
from dataclasses import dataclass, field
import math

import numpy as np

LABELS = ('success', 'no_contact', 'touched_not_held', 'slipped', 'knocked_over', 'held_at_cutoff')


@dataclass(frozen=True)
class Criteria:
    lift_m: float = .10          # centre of mass above the initial resting height
    hold_s: float = .5           # continuous duration of lifted + gripped + off the support
    hand_force_n: float = .05    # each hand must press at least this hard to count as gripping
    support_force_n: float = .05  # more non-robot contact force than this means still supported
    grip_s: float = .1           # gripped and clear of the support this long is a valid hold
    clear_m: float = .01         # ... while at least this far above the resting height
    tilt_deg: float = 30.        # tipped further than this, or
    shove_m: float = .10         # pushed this far sideways without being held, or
    drop_m: float = .05          # fallen this far below the resting height: knocked over


def gripped(forces, c):
    """Both hands press on the box and nothing except the robot touches it."""
    return (forces['left_n'] >= c.hand_force_n and forces['right_n'] >= c.hand_force_n
            and forces['external_n'] <= c.support_force_n)


class Sustained:
    """Start time of a condition's current unbroken run, and whether it has lasted long enough."""
    def __init__(self, duration):
        self.duration, self.since = duration, None

    def update(self, time, holds):
        self.since = (time if self.since is None else self.since) if holds else None
        return self.since is not None and time - self.since >= self.duration - 1e-9


class GraspJudge:
    """Episode-level success. ``rest_height`` is the box height measured before any attempt."""
    def __init__(self, rest_height, criteria=Criteria()):
        if not math.isfinite(rest_height):
            raise ValueError('rest height must be finite')
        self.rest_height, self.criteria = float(rest_height), criteria
        self._run = Sustained(criteria.hold_s)
        self.success_time = None

    def update(self, time, box_height, forces):
        c = self.criteria
        lifted = box_height - self.rest_height >= c.lift_m
        if self._run.update(time, lifted and gripped(forces, c)) and self.success_time is None:
            self.success_time = time
        return self.success_time is not None


@dataclass
class Attempt:
    """One round: what happened to the box between this latch and the next."""
    index: int
    start_time: float
    latched_box_pos: tuple
    rest_height: float
    criteria: Criteria = field(default_factory=Criteria)
    events: dict = field(default_factory=dict)
    end_time: float | None = None
    holding: bool = False
    notes: dict = field(default_factory=dict)   # reach and box-motion measurements, see GraspLoop

    def __post_init__(self):
        self._grip = Sustained(self.criteria.grip_s)
        self._loss = Sustained(self.criteria.grip_s)

    def _mark(self, name, time):
        self.events.setdefault(name, time)

    def update(self, time, box_pos, box_up_z, forces, hands_closed, success=False):
        """Feed one control step. ``box_up_z`` is the world-z component of the box's own up axis."""
        c = self.criteria
        for side in ('left', 'right'):
            if forces[side + '_n'] >= c.hand_force_n:
                self._mark(side + '_contact', time)
        if forces['robot_other_n'] >= c.hand_force_n:
            self._mark('other_contact', time)
        if hands_closed:
            self._mark('closed', time)
        clear = box_pos[2] - self.rest_height >= c.clear_m
        grip = gripped(forces, c) and clear
        if forces['external_n'] <= c.support_force_n and clear:
            self._mark('off_support', time)
        if self._grip.update(time, grip):
            self._mark('held', self._grip.since)
            self.holding = True
        if self.holding and self._loss.update(time, not grip) and not success:
            self._mark('slipped', self._loss.since)
            self.holding = False
        if success:
            self._mark('success', time)
        if not self.holding:
            tilt = math.degrees(math.acos(max(-1., min(1., box_up_z))))
            shove = float(np.linalg.norm(np.asarray(box_pos[:2]) - np.asarray(self.latched_box_pos[:2])))
            if tilt > c.tilt_deg or shove > c.shove_m or self.rest_height - box_pos[2] > c.drop_m:
                self._mark('knocked_over', time)

    def close(self, time):
        self.end_time = time

    @property
    def touched(self):
        return any(k in self.events for k in ('left_contact', 'right_contact', 'other_contact'))

    @property
    def label(self):
        e = self.events
        if 'success' in e:
            return 'success'
        if 'held' in e:
            return 'slipped' if 'slipped' in e else 'held_at_cutoff'
        if 'knocked_over' in e:
            return 'knocked_over'
        return 'touched_not_held' if self.touched else 'no_contact'

    def summary(self):
        return {'attempt': self.index, 'label': self.label, 'start_time': self.start_time,
                'end_time': self.end_time, 'latched_box_pos': list(self.latched_box_pos),
                'touched': self.touched, 'held': 'held' in self.events, 'events': dict(self.events),
                **({'notes': dict(self.notes)} if self.notes else {})}


def summarize(attempts, termination, success_time, clock_start):
    """Episode record: per-attempt labels plus the two rates the plan asks for."""
    rows = [a.summary() for a in attempts]
    held = [r for r in rows if r['held']]
    return {'success': success_time is not None, 'termination': termination,
            'attempts': rows, 'attempt_count': len(rows),
            'first_attempt_success': bool(rows and rows[0]['label'] == 'success'),
            'attempts_to_success': next((r['attempt'] + 1 for r in rows if r['label'] == 'success'), None),
            'completion_s': None if success_time is None else success_time - clock_start,
            'touched_attempts': sum(r['touched'] for r in rows),
            'held_attempts': len(held), 'slipped_attempts': sum(r['label'] == 'slipped' for r in held)}
