"""Measured-pose gripper events for execution-side use (not chunk prediction).

One gate per hand. Call every control tick using measured TCP pose/speed, in
the event target's stable coordinate frame. Command state is NOT grasp success.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class GripperEvent:
    event_id: str
    close: bool
    target_position: tuple[float, float, float]
    target_rotation: tuple[tuple[float, float, float], ...]
    deadline_s: float  # absolute episode simulation time, not time since the latest replan


class ArrivalGripperGate:
    def __init__(self, *, position_tolerance_m=0.03, angle_tolerance_rad=0.25,
                 speed_tolerance_mps=0.1, dwell_s=0.1, max_observation_age_s=0.05):
        for value in (position_tolerance_m, angle_tolerance_rad, speed_tolerance_mps,
                      dwell_s, max_observation_age_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("gate tolerances and durations must be finite and positive")
        self.position_tolerance = position_tolerance_m
        self.angle_tolerance = angle_tolerance_rad
        self.speed_tolerance = speed_tolerance_mps
        self.dwell_s = dwell_s
        self.max_age = max_observation_age_s
        self.reset()

    def reset(self, *, closed=False):
        self.closed = bool(closed)
        self.event = None
        self.status = "idle"
        self._seen_ids = set()
        self._inside_since = None
        self._last_time = None
        self._last_observation_time = None

    def submit(self, event: GripperEvent):
        if not isinstance(event.event_id, str) or not event.event_id:
            raise ValueError("event_id must be nonempty")
        if not isinstance(event.close, bool) or not math.isfinite(event.deadline_s):
            raise ValueError("event needs a bool close command and finite deadline")
        target = np.asarray(event.target_position, dtype=float)
        rotation = np.asarray(event.target_rotation, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all() or not self._rotation_valid(rotation):
            raise ValueError("invalid target pose")
        # Repeated HTTP responses / replans do not reset dwell or rearm an event.
        if event.event_id in self._seen_ids:
            return False
        self._seen_ids.add(event.event_id)
        self.event = event
        self.status = "pending"
        self._inside_since = None
        return True

    def cancel(self):
        self.event = None
        self._inside_since = None
        self.status = "cancelled"  # preserve current command, including a held object

    @staticmethod
    def _rotation_valid(rotation):
        return (rotation.shape == (3, 3) and np.isfinite(rotation).all()
                and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
                and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4))

    def update(self, *, now_s, observation_time_s, position, rotation, speed_mps):
        if not math.isfinite(now_s) or (self._last_time is not None and now_s < self._last_time):
            raise ValueError("simulation time must be finite and monotonic; reset on each episode")
        if self._last_time is not None and now_s - self._last_time > self.max_age:
            self._inside_since = None
        self._last_time = now_s
        if self.event is None or self.status != "pending":
            return self.closed
        if now_s > self.event.deadline_s:
            self.status = "timed_out"
            self._inside_since = None
            return self.closed
        pos, rot = np.asarray(position, dtype=float), np.asarray(rotation, dtype=float)
        fresh = (math.isfinite(observation_time_s)
                 and 0 <= now_s - observation_time_s <= self.max_age
                 and (self._last_observation_time is None
                      or observation_time_s > self._last_observation_time))
        if fresh:
            self._last_observation_time = observation_time_s
        valid = (fresh and pos.shape == (3,) and np.isfinite(pos).all()
                 and self._rotation_valid(rot) and math.isfinite(speed_mps) and speed_mps >= 0)
        arrived = False
        if valid:
            error = np.linalg.norm(pos - self.event.target_position)
            cos_angle = (np.trace(np.asarray(self.event.target_rotation).T @ rot) - 1) / 2
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            arrived = (error <= self.position_tolerance and angle <= self.angle_tolerance
                       and speed_mps <= self.speed_tolerance)
        if not arrived:
            self._inside_since = None
        elif self._inside_since is None:
            self._inside_since = now_s
        elif now_s - self._inside_since >= self.dwell_s - 1e-9:
            self.closed = self.event.close
            self.status = "fired"
        return self.closed
