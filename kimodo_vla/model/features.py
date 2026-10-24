"""Pose trajectories -> Kimodo features, shared by the planner and by training data.

Training and evaluation must turn the same robot data into the same model
inputs; keeping the conversion in one place is what guarantees it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from kimodo_vla.core.types import G1_JOINT_INDEX, G1_JOINT_NAMES, Observation
from kimodo_vla.motion.g1_reference import HumanoidArenaActionDecoder, _interpolate_motion
from kimodo_vla.motion.representation.kimodo_motionrep import KimodoMotionRep
from kimodo_vla.skeleton.definitions import G1Skeleton34

G1_XML_PATH = Path(__file__).resolve().parents[1] / "skeleton/assets/g1skel34/xml/g1.xml"


def history_feature_mask(representation: KimodoMotionRep, joint_mask: np.ndarray) -> torch.Tensor:
    """Which Kimodo features of a history frame are known from observations.

    Root position is never observed (the policy only sees orientation), so the
    mask covers the root heading plus the global rotation of every skeleton joint
    the benchmark reports. Joints it does not report must still be filled with a
    plausible value by the adapter, since they shift their children's rotations.
    """
    mask = torch.zeros(representation.motion_rep_dim, dtype=torch.bool)
    mask[representation.slice_dict["global_root_heading"]] = True
    rot = representation.slice_dict["global_rot_data"]
    skeleton = representation.skeleton
    observed: dict[str, bool] = {}
    for j, name in enumerate(skeleton.bone_order_names):
        parent = skeleton.bone_parents[name]
        joint = name.replace("_skel", "_joint")
        if parent is None:
            observed[name] = True
        elif joint in G1_JOINT_INDEX:
            observed[name] = bool(joint_mask[G1_JOINT_INDEX[joint]])
        else:  # toe and hand end points carry no joint of their own
            observed[name] = observed[parent]
        if observed[name]:
            mask[rot.start + 6 * j : rot.start + 6 * j + 6] = True
    return mask


class KimodoFeatures:
    def __init__(self, control_fps: float, model_fps: float = 30.0):
        self.control_fps = float(control_fps)
        self.model_fps = float(model_fps)
        self.codec = HumanoidArenaActionDecoder(G1Skeleton34(), G1_XML_PATH, fps=self.control_fps)
        self.representation = KimodoMotionRep(G1Skeleton34(), fps=self.model_fps, stats_path=None)

    def motion(self, joint_pos, root_pos, root_rot, first_step: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """A control-rate MuJoCo pose trajectory as Kimodo local rotations and root at model rate.

        ``first_step`` is the episode step of row 0: the model-rate grid is anchored
        at the episode start, as when training resamples whole episodes.
        """
        pose = self.codec.decode_joint_configuration_pose(
            torch.as_tensor(np.asarray(joint_pos), dtype=torch.float32),
            torch.as_tensor(np.asarray(root_pos), dtype=torch.float32),
            root_rotation_matrices=torch.as_tensor(np.asarray(root_rot), dtype=torch.float32),
            joint_names=G1_JOINT_NAMES,
        )
        local_rot, root = pose["local_rot_mats"], pose["root_positions"]
        last_step = first_step + local_rot.shape[0] - 1
        ratio = self.model_fps / self.control_fps
        first_k = math.ceil(first_step * ratio - 1e-6)
        last_k = int(round(last_step * ratio))
        times = torch.arange(first_k, last_k + 1, dtype=torch.float32) / self.model_fps
        positions = (times * self.control_fps - first_step).clamp(0, local_rot.shape[0] - 1)
        return _interpolate_motion(local_rot, root, positions)

    def features(self, local_rot: torch.Tensor, root: torch.Tensor) -> torch.Tensor:
        """Unnormalized Kimodo features ``[T, 417]``."""
        frames = local_rot.shape[0]
        if frames < 5:  # the smooth-root solver needs a few samples; pad, then drop the padding
            pad = 5 - frames
            local_rot = torch.cat((local_rot, local_rot[-1:].expand(pad, -1, -1, -1)))
            root = torch.cat((root, root[-1:].expand(pad, -1)))
        return self.representation(local_rot, root, to_normalize=False).cpu()[:frames]

    def history(self, observations: Sequence[Observation], first_heading: np.ndarray):
        """What the policy is conditioned on: observed orientation and joints, no root position.

        Returns ``(features [T, D], mask [T, D], local_rot, root)`` at model rate;
        rotations are relative to the episode's first heading.
        """
        rots = np.stack([first_heading.T @ o.root_rot for o in observations])
        joints = np.stack([o.joint_pos for o in observations])
        local_rot, root = self.motion(joints, np.zeros((len(joints), 3), np.float32), rots, observations[0].step)
        features = self.features(local_rot, root)
        mask = history_feature_mask(self.representation, observations[-1].joint_mask)
        return features, mask.expand_as(features), local_rot, root

    def target(self, joint_pos, root_pos, root_rot, first_heading: np.ndarray) -> torch.Tensor:
        """What the policy should output, from a whole episode's commanded poses (world frame)."""
        relative = first_heading.T
        local_rot, root = self.motion(
            joint_pos,
            np.einsum("ij,tj->ti", relative, root_pos),
            np.einsum("ij,tjk->tik", relative, root_rot),
        )
        return self.features(local_rot, root)
