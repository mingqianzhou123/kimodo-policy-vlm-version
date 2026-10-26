"""SIMPLE's Dex3 hands: closure <-> joint targets, and the debounce for binary hand checkpoints.

Ported from the original repository (``data/simple_hand.py`` and
``_SimpleRuntimeMixin`` in ``evaluation/simple_server.py``). Joint vectors use
the natural order thumb (3), index (2), middle (2) unless marked MuJoCo order
(thumb, middle, index), which is how SIMPLE reports measured hand joints.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

# Binary checkpoints: the closed pose sent when the hand is closed.
BINARY_LEFT_CLOSE = np.asarray([0.3523, -0.0964, 0.2790, -0.5058, -1.1950, -0.5389, -0.9835], np.float32)
BINARY_RIGHT_CLOSE = np.asarray([-0.5, -0.7, -0.7, 1.5, 1.5, 0.6, 1.5], np.float32)  # dominant closed target in SIMPLE demos
# Continuous checkpoints: the teleop demonstrations move each hand along closure * this pose.
CONTINUOUS_LEFT_CLOSE = np.asarray([-0.5, 0.7, 0.7, -1.5, -1.5, -0.6, -1.5], np.float32)
CONTINUOUS_RIGHT_CLOSE = np.asarray([-0.5, -0.7, -0.7, 1.5, 1.5, 0.6, 1.5], np.float32)
CONTINUOUS_CLOSE = np.stack([CONTINUOUS_LEFT_CLOSE, CONTINUOUS_RIGHT_CLOSE])

CLOSE_CONFIRM_FRAMES = 2
OPEN_CONFIRM_FRAMES = 6
OPEN_CONFIRM_REPLANS = 2
MIN_CLOSE_SECONDS = 1.0
MIN_OPEN_SECONDS = 0.4
MAX_HOLD_SECONDS = 2.0
OPEN_POSITION_TOL = 0.08
VELOCITY_TOL = 0.20
CLOSE_PROGRESS = 0.45
TORQUE_FRACTION = 0.75


def mujoco_to_natural(hand: np.ndarray) -> np.ndarray:
    """One hand's 7 joints, MuJoCo order (thumb, middle, index) <-> natural (the swap is its own inverse)."""
    hand = np.asarray(hand, np.float32).reshape(-1)
    if hand.size != 7:
        raise ValueError(f"expected 7 hand values, got {hand.size}")
    return np.concatenate([hand[:3], hand[5:7], hand[3:5]])


def binary_targets(closure: np.ndarray) -> np.ndarray:
    """``(T, 2)`` closure -> ``(T, 14)``: the closed pose at >= 0.5, else open."""
    closed = np.asarray(closure) >= 0.5
    return np.concatenate([np.where(closed[:, :1], BINARY_LEFT_CLOSE, 0.0),
                           np.where(closed[:, 1:], BINARY_RIGHT_CLOSE, 0.0)], axis=1).astype(np.float32)


def continuous_targets(closure: np.ndarray) -> np.ndarray:
    """``(T, 2)`` closure -> ``(T, 14)``: closure times the closed pose."""
    closure = np.clip(np.asarray(closure, np.float32), 0.0, 1.0)
    return (closure[:, :, None] * CONTINUOUS_CLOSE[None]).reshape(-1, 14).astype(np.float32)


def closure_of(hand_q: np.ndarray) -> np.ndarray:
    """``(T, 14)`` natural-order joints -> ``(T, 2)`` closure, the projection onto the closed poses."""
    hand_q = np.asarray(hand_q, np.float32).reshape(-1, 2, 7)
    closure = np.sum(hand_q * CONTINUOUS_CLOSE[None], axis=2) / np.sum(CONTINUOUS_CLOSE**2, axis=1)
    return np.clip(closure, 0.0, 1.0).astype(np.float32)


def measured_closure(proprio: Mapping[str, Any]) -> np.ndarray:
    """``(2,)`` closure from SIMPLE's measured ``left_hand_q`` / ``right_hand_q`` (MuJoCo order)."""
    q = np.concatenate([mujoco_to_natural(proprio["left_hand_q"]), mujoco_to_natural(proprio["right_hand_q"])])
    return closure_of(q[None])[0]


class HandDebounce:
    """SIMPLE's per-hand event controller for binary checkpoints.

    Closing needs ``CLOSE_CONFIRM_FRAMES`` agreeing frames; opening needs
    ``OPEN_CONFIRM_FRAMES`` frames and two consecutive replans whose whole
    prediction no longer contains a confirmed close. After a change the hand is
    held until the fingers settle (measured, at least the minimum time, at most
    ``MAX_HOLD_SECONDS``). A close confirmed in the unexecuted tail of a
    prediction is scheduled by model time, so replanning cannot move it.

    Use: ``advance(hand_observation)`` when a request arrives, ``__call__`` as
    the planner's ``hand_filter``, ``executed(steps)`` once the chunk is sent.
    """

    def __init__(self, control_fps: float = 50.0):
        self.control_fps = max(float(control_fps), 1.0)
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros(2, bool)
        self.phase = ["stable_open", "stable_open"]
        self.hold_steps = np.zeros(2, np.int64)
        self.elapsed_steps = np.zeros(2, np.int64)
        self.stable_votes = np.zeros(2, np.int64)
        self.candidate_state = np.zeros(2, bool)
        self.candidate_count = np.zeros(2, np.int64)
        self.open_votes = np.zeros(2, np.int64)
        self.pending_state = np.zeros(2, bool)
        self.pending_steps = np.full(2, -1, np.int64)
        self.last_chunk_steps = 0

    def executed(self, steps: int) -> None:
        self.last_chunk_steps = int(steps)

    def _close_pose_mujoco(self, side: int) -> np.ndarray:
        return mujoco_to_natural(BINARY_LEFT_CLOSE if side == 0 else BINARY_RIGHT_CLOSE)

    def _settled(self, side: int, closed: bool, hand_observation: Mapping[str, Any] | None) -> bool:
        """Has the hand reached its target? A grasped object can stop a closing hand early,
        so high effort at low velocity also counts as closed."""
        if not isinstance(hand_observation, Mapping):
            return False
        prefix = "left" if side == 0 else "right"
        q, dq = hand_observation.get(f"{prefix}_hand_q"), hand_observation.get(f"{prefix}_hand_dq")
        if q is None or dq is None:
            return False
        q, dq = np.asarray(q, np.float32).reshape(-1), np.asarray(dq, np.float32).reshape(-1)
        if q.size != 7 or dq.size != 7 or not (np.isfinite(q).all() and np.isfinite(dq).all()):
            return False
        if float(np.max(np.abs(dq))) > VELOCITY_TOL:
            return False
        if not closed:
            return bool(float(np.max(np.abs(q))) <= OPEN_POSITION_TOL)
        span = np.abs(self._close_pose_mujoco(side))
        progress = float(np.mean(np.abs(q) / np.maximum(span, 1e-3)))
        tau = hand_observation.get(f"{prefix}_hand_tau_est")
        effort_high = False
        if tau is not None:
            tau = np.asarray(tau, np.float32).reshape(-1)
            limits = np.asarray([2.45] + [0.7] * 6, np.float32)
            effort_high = bool(tau.size == 7 and np.isfinite(tau).all() and np.max(np.abs(tau) / limits) >= TORQUE_FRACTION)
        return bool(progress >= CLOSE_PROGRESS or effort_high)

    def advance(self, hand_observation: Mapping[str, Any] | None) -> None:
        """Account for the chunk executed since the last request, using the latest measurement."""
        elapsed, self.last_chunk_steps = self.last_chunk_steps, 0
        if elapsed <= 0:
            return
        for side, phase in enumerate(self.phase):
            if phase not in ("closing", "opening"):
                continue
            self.elapsed_steps[side] += elapsed
            self.hold_steps[side] = max(0, int(self.hold_steps[side]) - elapsed)
            if self.hold_steps[side] == 0 and self._settled(side, bool(self.state[side]), hand_observation):
                self.stable_votes[side] += 1
            else:
                self.stable_votes[side] = 0
            if self.stable_votes[side] >= 1 or self.elapsed_steps[side] >= int(round(MAX_HOLD_SECONDS * self.control_fps)):
                self.phase[side] = "stable_closed" if self.state[side] else "stable_open"
                self.stable_votes[side] = 0
                self.candidate_count[side] = 0

    def _switch(self, side: int, new_state: bool, at: int, effective: torch.Tensor, current: bool) -> None:
        self.state[side] = new_state
        self.phase[side] = "closing" if new_state else "opening"
        seconds = MIN_CLOSE_SECONDS if new_state else MIN_OPEN_SECONDS
        self.hold_steps[side] = max(1, int(round(seconds * self.control_fps)))
        self.elapsed_steps[side] = 0
        self.stable_votes[side] = 0
        self.candidate_count[side] = 0
        effective[:at, side] = float(current)
        effective[at:, side] = float(new_state)

    def __call__(self, prefix: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """The executed model-rate prefix ``[n, 2]`` after debouncing, given the whole prediction ``[T, 2]``."""
        hand = torch.as_tensor(prefix, dtype=torch.float32).cpu()
        detection = torch.as_tensor(prediction, dtype=torch.float32).cpu()
        if hand.ndim != 2 or hand.shape[1] != 2 or detection.ndim != 2 or detection.shape[0] < hand.shape[0]:
            raise ValueError(f"expected prefix [n, 2] within prediction [T, 2], got {tuple(hand.shape)}, {tuple(detection.shape)}")
        effective = torch.zeros_like(hand)
        for side in range(2):
            current = bool(self.state[side])
            if self.phase[side] in ("closing", "opening"):
                effective[:, side] = float(current)
                continue

            pending = int(self.pending_steps[side])
            if pending >= 0:
                pending_state = bool(self.pending_state[side])
                if pending_state == current:
                    self.pending_steps[side] = -1
                elif pending >= hand.shape[0]:
                    self.pending_steps[side] = pending - hand.shape[0]
                    effective[:, side] = float(current)
                    continue
                else:
                    self.pending_steps[side] = -1
                    self._switch(side, pending_state, pending, effective, current)
                    continue

            count, candidate = int(self.candidate_count[side]), bool(self.candidate_state[side])
            at, new_state = -1, current
            confirm = CLOSE_CONFIRM_FRAMES if not current else OPEN_CONFIRM_FRAMES
            for frame, value in enumerate(hand[:, side].tolist()):
                sample = bool(float(value) >= 0.5)
                if sample == current:
                    count, candidate = 0, current
                    continue
                if count == 0 or candidate != sample:
                    candidate, count = sample, 1
                else:
                    count += 1
                if count >= confirm:
                    at, new_state = frame - confirm + 1, sample
                    break

            if at >= 0:
                if not new_state:
                    # A prefix-only open is not enough to release while the whole
                    # prediction still has a confirmed close segment.
                    if _first_close(detection[:, side]) >= 0:
                        self.open_votes[side] = 0
                        self.candidate_state[side] = current
                        self.candidate_count[side] = 0
                        effective[:, side] = float(current)
                        continue
                    self.open_votes[side] += 1
                    if self.open_votes[side] < OPEN_CONFIRM_REPLANS:
                        self.candidate_state[side] = current
                        self.candidate_count[side] = 0
                        effective[:, side] = float(current)
                        continue
                self.open_votes[side] = 0
                self._switch(side, new_state, at, effective, current)
            else:
                if count == 0:
                    self.open_votes[side] = 0
                self.candidate_state[side] = candidate
                self.candidate_count[side] = count
                effective[:, side] = float(current)
                # Only closes are scheduled from the unexecuted tail; releases wait for executed evidence.
                if not current and detection.shape[0] > hand.shape[0]:
                    future = _first_close(detection[:, side])
                    if future >= 0:
                        self.pending_state[side] = True
                        self.pending_steps[side] = max(0, future - hand.shape[0])
        return effective


def _first_close(values: torch.Tensor) -> int:
    """Start of the first run of ``CLOSE_CONFIRM_FRAMES`` closed frames, or -1."""
    run = 0
    for frame, value in enumerate(values.tolist()):
        run = run + 1 if float(value) >= 0.5 else 0
        if run >= CLOSE_CONFIRM_FRAMES:
            return frame - CLOSE_CONFIRM_FRAMES + 1
    return -1
