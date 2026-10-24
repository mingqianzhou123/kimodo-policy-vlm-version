"""GR00T lower-body commands <-> the pelvis path of a whole-body motion.

GR00T's whole-body controllers (GR00T-WBC in BiGym 2.0, GR00T Balance/Walk) own
the legs and take the lower body as four commands: planar velocity in the pelvis
heading frame, yaw rate and pelvis height. Kimodo plans a pelvis path instead, so
every benchmark on one of these controllers needs both directions:

* ``BaseCommander``: online, frame 0 of a planned ``ActionChunk`` -> the command
  for the next step (evaluation);
* ``pelvis_path``: a recorded command sequence -> the pelvis path the operator
  asked for (training targets, round-trip checks).

Each is one control step of difference or sum, and they invert each other
(``tests/test_groot_wbc.py``). Lessons from replaying BiGym demos through this
interface (``docs/ARCHITECTURE.md`` section 10):

* Velocity and yaw rate are both backward differences into frame 0: the yaw rate
  runs from the previous step's target heading. A forward difference leads the
  operator by one step and cost pick_box 12 points (90% -> 78% at 50 Hz).
* The heading is pure feed-forward (``yaw_gain`` 0). Feedback on the measured
  heading fights the controller's own heading drift and walks a different route:
  gain 2 cut move_plate from 97% to 23% success and pick_box from 77% to 2%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from kimodo_vla.core.types import ActionChunk, yaw_matrix, yaw_of


class BaseCommand(NamedTuple):
    """Lower-body command: scalars for one step, ``(T,)`` arrays for a sequence."""

    vx: np.ndarray        # m/s, forward in the pelvis heading frame
    vy: np.ndarray        # m/s, to the left
    yaw_rate: np.ndarray  # rad/s
    height: np.ndarray    # m, absolute pelvis height


def base_command(target: ActionChunk, root_rot: np.ndarray, previous_yaw: float | None = None,
                 yaw_gain: float = 0.0) -> BaseCommand:
    """Frame 0 of ``target`` as one command; ``root_rot`` is the measured pelvis orientation.

    The yaw rate runs from ``previous_yaw`` (the last step's target heading, or the
    measured one at the start) to frame 0, like ``root_delta_xy``.
    """
    fps = target.fps
    vx, vy = target.root_delta_xy[0] * fps
    yaw_now = float(yaw_of(root_rot))
    yaw_target = float(yaw_of(target.root_rot[0]))
    yaw_from = yaw_now if previous_yaw is None else previous_yaw
    yaw_rate = _wrap(yaw_target - yaw_from) * fps + yaw_gain * _wrap(yaw_target - yaw_now)
    return BaseCommand(vx, vy, yaw_rate, target.root_height[0])


@dataclass
class BaseCommander:
    """``base_command`` across an episode: remembers the last target heading."""

    # rad/s of yaw-rate command per rad of measured heading error; keep 0 (see module docstring).
    yaw_gain: float = 0.0
    _previous_yaw: float | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self._previous_yaw = None

    def __call__(self, target: ActionChunk, root_rot: np.ndarray) -> BaseCommand:
        command = base_command(target, root_rot, self._previous_yaw, self.yaw_gain)
        self._previous_yaw = float(yaw_of(target.root_rot[0]))
        return command


def pelvis_path(commands: BaseCommand, start_pos: np.ndarray, start_yaw: float, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Integrate ``(T,)`` commands into the pelvis path they ask for: ``(T, 3)`` positions, ``(T, 3, 3)`` yaw rotations.

    Row ``t`` moves the pelvis from pose ``t - 1`` to pose ``t`` (the backward
    difference of ``base_command``), so row 0 is the start pose and its velocity and
    yaw rate are unused; the height is taken as commanded on every row.
    """
    dt = 1.0 / fps
    vx, vy, yaw_rate, height = commands
    yaw = start_yaw + np.concatenate([[0.0], np.cumsum(yaw_rate[1:]) * dt])
    root_pos = np.zeros((len(height), 3), np.float32)
    root_pos[0] = start_pos
    for t in range(1, len(height)):
        heading = yaw_matrix(float(yaw[t]))[:2, :2]
        root_pos[t, :2] = root_pos[t - 1, :2] + heading @ np.array([vx[t], vy[t]]) * dt
    root_pos[:, 2] = height
    return root_pos, np.stack([yaw_matrix(float(y)) for y in yaw])


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)
