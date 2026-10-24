"""Planners that replay a recorded motion, for checking a benchmark's action interface.

Handing the Agent a demo's own motion in place of a model prediction shows how
much success survives the interface if the model predicted perfectly:

* ``DemoPlanner`` goes through the exact evaluation path (30 Hz Kimodo chunks,
  resampling to the control rate, 40-D encoding), like ``KimodoPlanner``;
* ``DirectPlanner`` hands the motion to the adapter at the control rate, skipping
  Kimodo's representation: the gap between the two is the model-rate loss.

See ``scripts/check_bigym2_roundtrip.py`` and ARCHITECTURE section 10.
"""

from __future__ import annotations

import numpy as np

from kimodo_vla.core.types import ActionChunk
from kimodo_vla.data.build import DemoMotion
from kimodo_vla.model.features import KimodoFeatures
from kimodo_vla.motion.g1_reference import resample_hand_binary, resample_hand_binary_chunk, resample_motion_chunk


class DemoPlanner:
    """A Planner that 'predicts' a recorded motion, chunked exactly like KimodoPlanner."""

    history_steps = 2

    def __init__(self, motion: DemoMotion, control_fps=50.0, model_fps=30.0, chunk=50):
        self.control_fps, self.model_fps, self.chunk = control_fps, model_fps, chunk
        self.kimodo = KimodoFeatures(control_fps, model_fps)
        self.rot, self.root = self.kimodo.motion(motion.joint_pos, motion.root_pos, motion.root_rot)
        self.hand = resample_hand_binary(motion.hand, control_fps, model_fps)
        self.reset()

    def reset(self, seed=None):
        self._previous_rot = None
        self._previous_hand = None

    def plan(self, history):
        last = len(self.rot) - 1  # past the end of the demo, hold its final pose
        k = int(round(history[-1].step * self.model_fps / self.control_fps))
        before = min(max(k - 1, 0), last)
        idx = np.clip(np.arange(k, k + self.chunk), 0, last)
        rot, root, hand = self.rot[idx], self.root[idx], self.hand[idx]
        previous_rot = self.rot[before] if self._previous_rot is None else self._previous_rot
        previous_hand = self.hand[before] if self._previous_hand is None else self._previous_hand
        control_rot, control_root = resample_motion_chunk(
            rot, root, self.model_fps, self.control_fps,
            previous_local_rot_mat=previous_rot, previous_root_position=self.root[before],
        )
        control_hand = resample_hand_binary_chunk(hand, self.model_fps, self.control_fps, previous_hand_binary=previous_hand)
        actions = self.kimodo.codec.encode(control_rot, control_root, previous_root_position=self.root[before], hand_binary=control_hand)
        self._previous_rot, self._previous_hand = rot[-1], hand[-1]
        return ActionChunk.from_array40(actions.numpy(), fps=self.control_fps)


class DirectPlanner:
    """The recorded motion at the control rate; frame 0 is the next step's row."""

    history_steps = 2

    def __init__(self, motion: DemoMotion, control_fps=50.0, chunk=83):
        self.motion, self.control_fps, self.chunk = motion, control_fps, chunk

    def reset(self, seed=None):
        pass

    def plan(self, history):
        m, last = self.motion, len(self.motion.joint_pos) - 1
        idx = np.clip(np.arange(history[-1].step + 1, history[-1].step + 1 + self.chunk), 0, last)
        delta = m.root_pos[idx] - m.root_pos[np.clip(idx - 1, 0, last)]
        local = np.einsum("tji,tj->ti", m.root_rot[idx], delta)[:, :2]  # into each frame's pelvis frame
        return ActionChunk(local, m.root_pos[idx, 2], m.root_rot[idx], m.joint_pos[idx], m.hand[idx], self.control_fps)
