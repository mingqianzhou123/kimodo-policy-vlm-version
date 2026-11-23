"""KimodoPlanner, decoding plans the way the original repository's runtime did.

The original ``KimodoHumanoidArenaRuntime`` spread the first ``execution_frames``
predicted frames over the control steps from the previous plan's last frame, so
every plan ran about one model frame behind the clock (``tests/test_planner_timing.py``),
and it remembered each plan's executed hand prefix as the next hand history.
``KimodoPlanner`` reads predictions at the true time instead. This class keeps the
old decoding only to compare with the original runtime (``tests/test_port_equivalence.py``,
``test_arena_port.py``, ``test_simple_port.py``) and to reproduce its evaluations
(``--timing original`` of ``serve_arena.py`` / ``serve_simple.py``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from kimodo_vla.core.types import ActionChunk, Observation
from kimodo_vla.model.planner import KimodoPlanner
from kimodo_vla.motion.g1_reference import (
    resample_hand_binary_chunk,
    resample_hand_continuous_chunk,
    resample_motion_chunk,
)


class OriginalTimingPlanner(KimodoPlanner):
    def __init__(self, *args, **kwargs):
        if kwargs.get("blend_steps"):
            raise ValueError("blend_steps needs KimodoPlanner: the original decoding keeps no unexecuted targets")
        super().__init__(*args, **kwargs)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        # The last executed hand frames and pose, as the original runtime kept them.
        self._hand_history = torch.zeros(1, 0, 2)
        self._previous_rot: torch.Tensor | None = None
        self._previous_root: torch.Tensor | None = None

    def _condition_hand(self, history: Sequence[Observation], cut: int, frames: int) -> torch.Tensor:
        if self.hand_mode != "binary":
            return super()._condition_hand(history, cut, frames)
        hand = self._hand_history[:, -frames:] if frames else self._hand_history[:, :0]
        if hand.shape[1] < frames:
            hand = torch.cat((torch.zeros(1, frames - hand.shape[1], 2), hand), dim=1)
        return hand

    def _decode(self, output: dict, obs: Observation, cut: int, last_rot: torch.Tensor, last_root: torch.Tensor) -> ActionChunk:
        n = self.execution_frames
        all_roots = output["root_positions"].float().cpu()
        local_rot = output["local_rot_mats"][:n].float().cpu()
        root_pos = all_roots[:n]
        # Continue from the last target actually handed out; fall back to the measurement at reset.
        previous_rot = self._previous_rot if self._previous_rot is not None else last_rot
        previous_root = self._previous_root if self._previous_root is not None else last_root
        previous_root = self._root_before(output, all_roots, previous_root)

        control_rot, control_root = resample_motion_chunk(
            local_rot, root_pos,
            source_fps=self.model_fps, target_fps=self.control_fps,
            previous_local_rot_mat=previous_rot, previous_root_position=previous_root,
        )
        if self.hand_mode == "continuous":
            hand = output["hand_closure"][:n].clamp(0.0, 1.0).float().cpu()
            control_hand = resample_hand_continuous_chunk(
                hand, source_fps=self.model_fps, target_fps=self.control_fps,
                previous_hand_closure=torch.as_tensor(obs.hand, dtype=torch.float32),
            )
        else:
            hand = output["hand_binary"][:n].float().cpu()
            if self.hand_filter is not None:
                hand = self.hand_filter(hand, output["hand_binary"].float().cpu())
            previous_hand = self._hand_history[0, -1] if self._hand_history.shape[1] else torch.zeros(2)
            control_hand = resample_hand_binary_chunk(
                hand, source_fps=self.model_fps, target_fps=self.control_fps, previous_hand_binary=previous_hand
            )
        self._hand_history = torch.cat((self._hand_history, hand.unsqueeze(0)), dim=1)[:, -self.history_frames :]
        self._previous_rot = local_rot[-1].clone()
        self._previous_root = root_pos[-1].clone()
        actions = self.kimodo.codec.encode(control_rot, control_root, previous_root_position=previous_root, hand_binary=control_hand)

        chunk = ActionChunk.from_array40(actions.numpy(), fps=self.control_fps)
        chunk.root_rot = np.einsum("ij,tjk->tik", self._first_heading, chunk.root_rot).astype(np.float32)
        return chunk
