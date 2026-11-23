# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Motion representation helpers: velocity, heading, masks, and rotation of features."""

from typing import List, Optional, Union

import torch

from kimodo_vla.utils.geometry import cont6d_to_matrix, matrix_to_cont6d
from kimodo_vla.skeleton.base import SkeletonBase
from kimodo_vla.utils.tools import ensure_batched


def diff_angles(angles: torch.Tensor, fps: float) -> torch.Tensor:
    """Compute frame-to-frame angular differences in radians, scaled by fps.

    Args:
        angles: [..., T] batched sequences of rotation angles in radians.
        fps: Sampling rate used to convert frame differences to per-second rate.

    Returns:
        [..., T-1] difference between consecutive angles (rad/s).
    """

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    cos_diff = cos[..., 1:] * cos[..., :-1] + sin[..., 1:] * sin[..., :-1]
    sin_diff = sin[..., 1:] * cos[..., :-1] - cos[..., 1:] * sin[..., :-1]

    # should be close to angles.diff() but more robust
    # multiply by fps = 1 / dt
    angles_diff = fps * torch.arctan2(sin_diff, cos_diff)
    return angles_diff


def _resolve_valid_mask(
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    lengths: Optional[torch.Tensor],
    valid_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if lengths is not None and valid_mask is not None:
        raise ValueError("Provide either lengths or valid_mask, not both")
    if valid_mask is not None:
        valid_mask = torch.as_tensor(valid_mask, device=device, dtype=torch.bool)
        expected_shape = (batch_size, sequence_length)
        if valid_mask.shape != expected_shape:
            raise ValueError(
                f"Expected valid_mask shape {expected_shape}, got {tuple(valid_mask.shape)}"
            )
        return valid_mask
    if lengths is None:
        if batch_size != 1:
            raise ValueError("lengths or valid_mask is required for batched input")
        return torch.ones(batch_size, sequence_length, device=device, dtype=torch.bool)
    lengths = torch.as_tensor(lengths, device=device, dtype=torch.long).reshape(-1)
    if lengths.shape != (batch_size,):
        raise ValueError(f"Expected lengths shape {(batch_size,)}, got {tuple(lengths.shape)}")
    if ((lengths < 0) | (lengths > sequence_length)).any():
        raise ValueError(f"lengths must be in [0, {sequence_length}]")
    return torch.arange(sequence_length, device=device).unsqueeze(0) < lengths.unsqueeze(1)


def _endpoint_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    false_column = torch.zeros(
        valid_mask.shape[0], 1, device=valid_mask.device, dtype=torch.bool
    )
    next_valid = torch.cat((valid_mask[:, 1:], false_column), dim=1)
    previous_valid = torch.cat((false_column, valid_mask[:, :-1]), dim=1)
    return valid_mask & ~next_valid & previous_valid


@ensure_batched(positions=4, lengths=1, valid_mask=2)
def compute_vel_xyz(
    positions: torch.Tensor,
    fps: float,
    lengths: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the velocities from positions: dx/dt. Works with batches. The last velocity is duplicated to keep the same size.

    Args:
        positions (torch.Tensor): [..., T, J, 3] xyz positions of a human skeleton
        fps (float): frame per seconds
        lengths (Optional[torch.Tensor]): [...] valid prefix lengths.
        valid_mask (Optional[torch.Tensor]): [...] boolean frame-validity mask.

    Returns:
        velocity (torch.Tensor): [..., T, J, 3] velocities computed from the positions
    """
    batch_size, sequence_length = positions.shape[:2]
    valid_mask = _resolve_valid_mask(
        batch_size,
        sequence_length,
        positions.device,
        lengths,
        valid_mask,
    )
    velocity = torch.zeros_like(positions)
    if sequence_length <= 1:
        return velocity
    pair_valid = valid_mask[:, :-1] & valid_mask[:, 1:]
    pair_velocity = fps * (positions[:, 1:] - positions[:, :-1])
    velocity[:, :-1] = pair_velocity * pair_valid[..., None, None]
    endpoint_mask = _endpoint_mask(valid_mask)
    previous_velocity = torch.cat(
        (torch.zeros_like(velocity[:, :1]), velocity[:, :-1]), dim=1
    )
    velocity = torch.where(endpoint_mask[..., None, None], previous_velocity, velocity)
    return velocity


@ensure_batched(root_rot_angles=2, lengths=1, valid_mask=2)
def compute_vel_angle(
    root_rot_angles: torch.Tensor,
    fps: float,
    lengths: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the local root rotation velocity: dtheta/dt.

    Args:
        root_rot_angles (torch.Tensor): [..., T] rotation angle (in radian)
        fps (float): frame per seconds
        lengths (Optional[torch.Tensor]): [...] valid prefix lengths.
        valid_mask (Optional[torch.Tensor]): [...] boolean frame-validity mask.

    Returns:
        local_root_rot_vel (torch.Tensor): [..., T] local root rotation velocity (in radian/s)
    """
    batch_size, sequence_length = root_rot_angles.shape
    valid_mask = _resolve_valid_mask(
        batch_size,
        sequence_length,
        root_rot_angles.device,
        lengths,
        valid_mask,
    )
    local_root_rot_vel = torch.zeros_like(root_rot_angles)
    if sequence_length <= 1:
        return local_root_rot_vel
    pair_valid = valid_mask[:, :-1] & valid_mask[:, 1:]
    local_root_rot_vel[:, :-1] = diff_angles(root_rot_angles, fps) * pair_valid
    endpoint_mask = _endpoint_mask(valid_mask)
    previous_velocity = torch.cat(
        (torch.zeros_like(local_root_rot_vel[:, :1]), local_root_rot_vel[:, :-1]), dim=1
    )
    local_root_rot_vel = torch.where(endpoint_mask, previous_velocity, local_root_rot_vel)
    return local_root_rot_vel


@ensure_batched(posed_joints=4)
def compute_heading_angle(posed_joints: torch.Tensor, skeleton: SkeletonBase) -> torch.Tensor:
    """Compute the heading direction from joint positions using the hip vector.

    Args:
        posed_joints: [B, T, J, 3] global joint positions.
        skeleton: Skeleton instance used to get hip joint indices.

    Returns:
        [B] heading angle in radians.
    """
    # compute root heading for the sequence from hip positions
    r_hip, l_hip = skeleton.hip_joint_idx
    diff = posed_joints[:, :, r_hip] - posed_joints[:, :, l_hip]
    heading_angle = torch.atan2(diff[..., 2], -diff[..., 0])
    return heading_angle


def length_to_mask(
    length: Union[torch.Tensor, List],
    max_len: Optional[int] = None,
    device=None,
) -> torch.Tensor:
    """Convert sequence lengths to a boolean validity mask.

    Args:
        length: Sequence lengths, either a tensor ``[B]`` or a Python list.
        max_len: Optional mask width. If omitted, uses ``max(length)``.
        device: Optional device. When ``length`` is a list, this controls where
            the new tensor is created.

    Returns:
        A boolean tensor of shape ``[B, max_len]`` where ``True`` marks valid
        timesteps.
    """
    if isinstance(length, list):
        if device is None:
            device = "cpu"
        length = torch.tensor(length, device=device)

    # Use requested device for output; move length if needed so mask and length match
    if device is not None:
        target = torch.device(device)
        if length.device != target:
            length = length.to(target)
    device = length.device

    if max_len is None:
        max_len = max(length)

    mask = torch.arange(max_len, device=device).expand(len(length), max_len) < length.unsqueeze(1)
    return mask


class RotateFeatures:
    """Helper that applies a global heading rotation to motion features."""

    def __init__(self, angle: torch.Tensor):
        """Precompute 2D and 3D rotation matrices for a batch of angles.

        Args:
            angle: Rotation angle(s) in radians, shaped ``[B]``.
        """
        self.angle = angle

        ## Create the necessary rotations matrices
        cos, sin = torch.cos(angle), torch.sin(angle)
        one, zero = torch.ones_like(angle), torch.zeros_like(angle)

        # 2D rotation transposed (sin are -sin)
        self.corrective_mat_2d_T = torch.stack((cos, sin, -sin, cos), -1).reshape(angle.shape + (2, 2))
        # 3D rotation on Y axis
        self.corrective_mat_Y = torch.stack((cos, zero, sin, zero, one, zero, -sin, zero, cos), -1).reshape(
            angle.shape + (3, 3)
        )
        self.corrective_mat_Y_T = self.corrective_mat_Y.transpose(1, 2).contiguous()

    def rotate_positions(self, positions: torch.Tensor):
        """Rotate 3D positions around the Y axis."""
        return positions @ self.corrective_mat_Y_T

    def rotate_2d_positions(self, positions_2d: torch.Tensor):
        """Rotate 2D ``(x, z)`` vectors in the ground plane."""
        return positions_2d @ self.corrective_mat_2d_T

    def rotate_rotations(self, rotations: torch.Tensor):
        """Left-multiply global rotation matrices by the heading correction."""
        # "Rotate" the global rotations
        # which means add an extra Y rotation after the transform
        # so at the left R' = R_y R
        # (since we use the convention x' = R x)
        # "bik,btdkj->btdij"

        B, T, J = rotations.shape[:3]
        BTJ = B * T * J
        return (
            self.corrective_mat_Y[:, None, None].expand(B, T, J, 3, 3).reshape(BTJ, 3, 3) @ rotations.reshape(BTJ, 3, 3)
        ).reshape(B, T, J, 3, 3)

    def rotate_6d_rotations(self, rotations_6d: torch.Tensor):
        """Rotate 6D rotation features via matrix conversion."""
        return matrix_to_cont6d(self.rotate_rotations(cont6d_to_matrix(rotations_6d)))
