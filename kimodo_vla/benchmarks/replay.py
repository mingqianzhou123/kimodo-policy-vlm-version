"""Planners that replay a recorded motion, for checking a benchmark's action interface.

Handing the Agent a demo's own motion in place of a model prediction shows how
much success survives the interface if the model predicted perfectly:

* ``DemoPlanner`` goes through the evaluation path (30 Hz Kimodo chunks read at the
  true time of each control step, 40-D encoding), like ``KimodoPlanner``;
* ``DirectPlanner`` hands the motion to the adapter at the control rate, skipping
  Kimodo's representation: the gap between the two is the model-rate loss.

See ``scripts/check_bigym2_roundtrip.py`` and ARCHITECTURE section 10.
"""

from __future__ import annotations

import numpy as np
import torch

from kimodo_vla.core.types import ActionChunk
from kimodo_vla.data.build import DemoMotion
from kimodo_vla.model.features import KimodoFeatures, lerp_rows
from kimodo_vla.motion.g1_reference import _interpolate_motion, resample_hand_binary, resample_hand_continuous


class DemoPlanner:
    """A Planner that 'predicts' a recorded motion and samples it like KimodoPlanner.

    The recorded motion is resampled to the model rate, then read at the true time of
    each control step the plan covers (``chunk`` model frames per plan).
    """

    history_steps = 2

    def __init__(self, motion: DemoMotion, control_fps=50.0, model_fps=30.0, chunk=50, hand_mode="binary"):
        self.control_fps, self.model_fps, self.chunk = control_fps, model_fps, chunk
        self.continuous = hand_mode == "continuous"
        self.kimodo = KimodoFeatures(control_fps, model_fps)
        self.rot, self.root = self.kimodo.motion(motion.joint_pos, motion.root_pos, motion.root_rot)
        self.hand = (resample_hand_continuous if self.continuous else resample_hand_binary)(motion.hand, control_fps, model_fps)

    def reset(self, seed=None):
        pass

    def plan(self, history):
        ratio = self.model_fps / self.control_fps
        step, last = history[-1].step, len(self.rot) - 1  # past the end of the demo, hold its final pose
        steps = int(round(self.chunk / ratio))
        times = (torch.arange(step, step + steps + 1, dtype=torch.float32) * ratio).clamp(0, last)
        rot, root = _interpolate_motion(self.rot, self.root, times)  # row 0: now
        if self.continuous:
            hand = lerp_rows(self.hand, times[1:])
        else:
            hand = self.hand[times[1:].round().long()]
        actions = self.kimodo.codec.encode(rot[1:], root[1:], previous_root_position=root[0], hand_binary=hand)
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
