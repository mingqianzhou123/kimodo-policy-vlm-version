"""V6.1 native conditions; all geometry is canonical Kimodo Y-up, in metres.

This module deliberately takes geometry, not pixels. Camera unprojection,
capture-time transforms and the rolling-window origin belong in the planner.
"""
from __future__ import annotations

import torch

from kimodo_vla.utils.geometry import matrix_to_cont6d


def normalized_future_condition(representation, values, mask, *, batch_size, frames, device, dtype):
    values = torch.as_tensor(values, device=device, dtype=torch.float32)
    mask = torch.as_tensor(mask, device=device)
    if values.ndim == 2 and batch_size == 1:
        values = values.unsqueeze(0)
    if mask.ndim == 2 and batch_size == 1:
        mask = mask.unsqueeze(0)
    shape = (batch_size, frames, representation.motion_rep_dim)
    if tuple(values.shape) != shape or tuple(mask.shape) != shape:
        raise ValueError(f"future condition and mask must both have shape {shape}")
    if mask.dtype != torch.bool:
        raise ValueError("future_condition_mask must be bool")
    if not torch.isfinite(values[mask]).all():
        raise ValueError("non-finite value in constrained future features")
    # Normalize BEFORE masking: a physical zero need not be a normalized zero.
    clean = values.masked_fill(~mask, 0)
    normalized = representation.normalize(clean).to(dtype=dtype).masked_fill(~mask, 0)
    return normalized, mask


def hand_keyframe(representation, *, frames: int, frame: int, smooth_root_xz,
                  hand_positions: dict[str, object], wrist_rotations: dict[str, object]):
    """One paired root/hand keyframe, suitable for predict_future's new arguments.

    For every selected side, supply [wrist, hand-end] absolute positions [2,3]
    and the wrist's global rotation [3,3]. The geometry must already share the
    history's Y-up canonical frame. The smoothed root is a ground-plane path,
    NOT the physical pelvis. Only root XZ and selected hand channels are known.
    Combine non-conflicting keyframes with their masks at the caller.
    """
    if not isinstance(frame, int) or not isinstance(frames, int) or not 0 <= frame < frames:
        raise ValueError("frame must be a future index in [0, frames)")
    if not hand_positions or set(hand_positions) != set(wrist_rotations):
        raise ValueError("each hand needs positions and a wrist rotation")
    if not set(hand_positions) <= {"left", "right"}:
        raise ValueError("hand sides must be left or right")
    root = torch.as_tensor(smooth_root_xz, dtype=torch.float32)
    if root.shape != (2,) or not torch.isfinite(root).all():
        raise ValueError("smooth_root_xz must be two finite coordinates")
    values = torch.zeros(frames, representation.motion_rep_dim)
    mask = torch.zeros_like(values, dtype=torch.bool)
    root_start = representation.slice_dict["smooth_root_pos"].start
    for axis, value in zip((0, 2), root):
        values[frame, root_start + axis] = value
        mask[frame, root_start + axis] = True
    names = representation.skeleton.bone_order_names
    for side, positions in hand_positions.items():
        pos = torch.as_tensor(positions, dtype=torch.float32)
        rot = torch.as_tensor(wrist_rotations[side], dtype=torch.float32)
        if pos.shape != (2, 3) or not torch.isfinite(pos).all():
            raise ValueError("hand positions must be finite [wrist, hand-end] Y-up coordinates")
        if (rot.shape != (3, 3) or not torch.isfinite(rot).all()
                or not torch.allclose(rot.T @ rot, torch.eye(3), atol=1e-4)
                or not torch.isclose(torch.linalg.det(rot), torch.tensor(1.0), atol=1e-4)):
            raise ValueError("wrist rotation must be a proper orthonormal matrix")
        joint_names = (f"{side}_wrist_yaw_skel", f"{side}_hand_roll_skel")
        for i, name in enumerate(joint_names):
            j = names.index(name)
            start = representation.slice_dict["local_joints_positions"].start + 3 * j
            relative = pos[i].clone()
            relative[[0, 2]] -= root  # height stays absolute; no heading rotation here
            values[frame, start:start + 3] = relative
            mask[frame, start:start + 3] = True
        j = names.index(joint_names[0])
        start = representation.slice_dict["global_rot_data"].start + 6 * j
        values[frame, start:start + 6] = matrix_to_cont6d(rot)
        mask[frame, start:start + 6] = True
    return values, mask
