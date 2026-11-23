from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from kimodo_vla.motion.feature_utils import compute_heading_angle, compute_vel_xyz
from kimodo_vla.motion.feet import foot_detect_from_pos_and_vel
from kimodo_vla.motion.smooth_root import get_smooth_root_pos
from kimodo_vla.skeleton.base import SkeletonBase
from kimodo_vla.utils.geometry import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
)


CANONICAL_G1_JOINT_NAMES_29 = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint", "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)

# Unitree's public 29-DoF state/action order. HumanoidEveryday, HIW-500 and
# UnifoLM WBT all expose body joint positions in this order.
UNITREE_G1_JOINT_NAMES_29 = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def rot6d_row_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert HumanoidArena's row-major 6D rotations to matrices."""
    flat = rot6d.reshape(-1, 6)
    column0 = flat[:, (0, 2, 4)]
    column1 = flat[:, (1, 3, 5)]
    column0 = torch.nn.functional.normalize(column0, dim=-1, eps=1e-8)
    column1 = column1 - (column0 * column1).sum(-1, keepdim=True) * column0
    column1 = torch.nn.functional.normalize(column1, dim=-1, eps=1e-8)
    column2 = torch.cross(column0, column1, dim=-1)
    return torch.stack((column0, column1, column2), dim=-1).reshape(rot6d.shape[:-1] + (3, 3))


def _complete_motion_dict(
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    skeleton: SkeletonBase,
    fps: float,
) -> dict[str, torch.Tensor]:
    global_rot_mats, posed_joints, _ = skeleton.fk(local_rot_mats, root_positions)
    smooth_root_pos = get_smooth_root_pos(root_positions.unsqueeze(0)).squeeze(0)
    lengths = torch.tensor([posed_joints.shape[0]], device=posed_joints.device)
    velocities = compute_vel_xyz(posed_joints.unsqueeze(0), fps, lengths=lengths).squeeze(0)
    heading = compute_heading_angle(posed_joints.unsqueeze(0), skeleton).squeeze(0)
    foot_contacts = foot_detect_from_pos_and_vel(
        posed_joints.unsqueeze(0), velocities.unsqueeze(0), skeleton, 0.15, 0.10
    ).squeeze(0)
    return {
        "posed_joints": posed_joints,
        "global_rot_mats": global_rot_mats,
        "local_rot_mats": local_rot_mats,
        "foot_contacts": foot_contacts,
        "smooth_root_pos": smooth_root_pos,
        "root_positions": root_positions,
        "global_root_heading": torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1),
    }


def _interpolate_motion(
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    source_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_frames = local_rot_mats.shape[0]
    lower = source_positions.floor().long()
    upper = (lower + 1).clamp(max=source_frames - 1)
    alpha = (source_positions - lower.float()).reshape(-1, 1, 1)

    root_out = (
        (1.0 - alpha[:, 0]) * root_positions[lower]
        + alpha[:, 0] * root_positions[upper]
    )
    quaternions = matrix_to_quaternion(local_rot_mats)
    quaternion0 = quaternions[lower]
    quaternion1 = quaternions[upper]
    dot = (quaternion0 * quaternion1).sum(dim=-1, keepdim=True)
    quaternion1 = torch.where(dot < 0, -quaternion1, quaternion1)
    dot = dot.abs().clamp(max=1.0)
    angle = torch.acos(dot)
    sin_angle = torch.sin(angle)
    linear = sin_angle.abs() < 1e-6
    weight0 = torch.sin((1.0 - alpha) * angle) / sin_angle.clamp_min(1e-8)
    weight1 = torch.sin(alpha * angle) / sin_angle.clamp_min(1e-8)
    quaternion_out = weight0 * quaternion0 + weight1 * quaternion1
    quaternion_out = torch.where(
        linear,
        (1.0 - alpha) * quaternion0 + alpha * quaternion1,
        quaternion_out,
    )
    quaternion_out = torch.nn.functional.normalize(quaternion_out, dim=-1, eps=1e-8)
    return quaternion_to_matrix(quaternion_out), root_out


def resample_motion(
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    source_fps: float,
    target_fps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if abs(float(source_fps) - float(target_fps)) < 1e-6:
        return local_rot_mats, root_positions
    source_frames = local_rot_mats.shape[0]
    target_frames = int(round((source_frames - 1) * float(target_fps) / float(source_fps))) + 1
    target_times = torch.arange(
        target_frames,
        device=root_positions.device,
        dtype=torch.float32,
    ) / float(target_fps)
    source_positions = (target_times * float(source_fps)).clamp(max=source_frames - 1)
    return _interpolate_motion(local_rot_mats, root_positions, source_positions)


def resample_motion_chunk(
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    source_fps: float,
    target_fps: float,
    previous_local_rot_mat: torch.Tensor | None = None,
    previous_root_position: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if local_rot_mats.shape[0] == 0:
        raise ValueError("Cannot resample an empty motion chunk")
    if previous_local_rot_mat is None:
        previous_local_rot_mat = local_rot_mats[0]
    if previous_root_position is None:
        previous_root_position = root_positions[0]
    extended_local_rot_mats = torch.cat(
        (previous_local_rot_mat.unsqueeze(0), local_rot_mats), dim=0
    )
    extended_root_positions = torch.cat(
        (previous_root_position.unsqueeze(0), root_positions), dim=0
    )
    target_intervals = max(
        1,
        int(round(local_rot_mats.shape[0] * float(target_fps) / float(source_fps))),
    )
    source_positions = torch.linspace(
        0,
        local_rot_mats.shape[0],
        target_intervals + 1,
        device=root_positions.device,
        dtype=torch.float32,
    )
    resampled_local, resampled_root = _interpolate_motion(
        extended_local_rot_mats,
        extended_root_positions,
        source_positions,
    )
    return resampled_local[1:], resampled_root[1:]


def _validate_hand_binary(hand_binary: np.ndarray | torch.Tensor) -> torch.Tensor:
    hand_binary = torch.as_tensor(hand_binary, dtype=torch.float32)
    if hand_binary.ndim != 2 or hand_binary.shape[-1] != 2:
        raise ValueError(
            f"Expected hand_binary shape (T, 2), got {tuple(hand_binary.shape)}"
        )
    if hand_binary.shape[0] == 0:
        raise ValueError("Cannot resample an empty hand sequence")
    if not torch.isfinite(hand_binary).all():
        raise ValueError("Hand state contains NaN or Inf")
    if not torch.logical_or(hand_binary == 0, hand_binary == 1).all():
        raise ValueError("Hand state must contain only binary 0/1 values")
    return hand_binary


def resample_hand_binary(
    hand_binary: np.ndarray | torch.Tensor,
    source_fps: float,
    target_fps: float,
) -> torch.Tensor:
    """Nearest-neighbor resampling for discrete left/right hand states."""
    hand_binary = _validate_hand_binary(hand_binary)
    if abs(float(source_fps) - float(target_fps)) < 1e-6:
        return hand_binary
    source_frames = hand_binary.shape[0]
    target_frames = int(round((source_frames - 1) * float(target_fps) / float(source_fps))) + 1
    target_times = torch.arange(
        target_frames,
        device=hand_binary.device,
        dtype=torch.float32,
    ) / float(target_fps)
    source_indices = (target_times * float(source_fps)).round().long()
    return hand_binary[source_indices.clamp(max=source_frames - 1)]


def resample_hand_binary_chunk(
    hand_binary: np.ndarray | torch.Tensor,
    source_fps: float,
    target_fps: float,
    previous_hand_binary: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """Resample one predicted chunk while preserving the previous executed state."""
    hand_binary = _validate_hand_binary(hand_binary)
    if previous_hand_binary is None:
        previous_hand_binary = hand_binary[0]
    previous_hand_binary = torch.as_tensor(
        previous_hand_binary,
        device=hand_binary.device,
        dtype=hand_binary.dtype,
    ).reshape(1, 2)
    _validate_hand_binary(previous_hand_binary)
    extended = torch.cat((previous_hand_binary, hand_binary), dim=0)
    target_intervals = max(
        1,
        int(round(hand_binary.shape[0] * float(target_fps) / float(source_fps))),
    )
    source_positions = torch.linspace(
        0,
        hand_binary.shape[0],
        target_intervals + 1,
        device=hand_binary.device,
        dtype=torch.float32,
    )
    source_indices = source_positions.round().long().clamp(max=hand_binary.shape[0])
    return extended[source_indices][1:]


def _validate_hand_continuous(
    hand_closure: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    hand_closure = torch.as_tensor(hand_closure, dtype=torch.float32)
    if hand_closure.ndim != 2 or hand_closure.shape[-1] != 2:
        raise ValueError(
            f"Expected hand_closure shape (T, 2), got {tuple(hand_closure.shape)}"
        )
    if hand_closure.shape[0] == 0:
        raise ValueError("Cannot resample an empty hand closure sequence")
    if not torch.isfinite(hand_closure).all():
        raise ValueError("Hand closure contains NaN or Inf")
    if ((hand_closure < -1e-6) | (hand_closure > 1.0 + 1e-6)).any():
        raise ValueError("Hand closure must be within [0, 1]")
    return hand_closure.clamp(0.0, 1.0)


def resample_hand_continuous(
    hand_closure: np.ndarray | torch.Tensor,
    source_fps: float,
    target_fps: float,
) -> torch.Tensor:
    """Linearly resample continuous left/right SIMPLE hand closure."""
    hand_closure = _validate_hand_continuous(hand_closure)
    if abs(float(source_fps) - float(target_fps)) < 1e-6:
        return hand_closure
    source_frames = hand_closure.shape[0]
    target_frames = int(
        round((source_frames - 1) * float(target_fps) / float(source_fps))
    ) + 1
    source_positions = (
        torch.arange(
            target_frames, device=hand_closure.device, dtype=torch.float32
        )
        * float(source_fps)
        / float(target_fps)
    ).clamp(max=source_frames - 1)
    lower = source_positions.floor().long()
    upper = (lower + 1).clamp(max=source_frames - 1)
    alpha = (source_positions - lower.float()).unsqueeze(-1)
    return (1.0 - alpha) * hand_closure[lower] + alpha * hand_closure[upper]


def resample_hand_continuous_chunk(
    hand_closure: np.ndarray | torch.Tensor,
    source_fps: float,
    target_fps: float,
    previous_hand_closure: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """Linearly resample a predicted closure chunk from its executed boundary."""
    hand_closure = _validate_hand_continuous(hand_closure)
    if previous_hand_closure is None:
        previous_hand_closure = hand_closure[0]
    previous = torch.as_tensor(
        previous_hand_closure,
        device=hand_closure.device,
        dtype=hand_closure.dtype,
    ).reshape(1, 2)
    previous = _validate_hand_continuous(previous)
    extended = torch.cat((previous, hand_closure), dim=0)
    target_intervals = max(
        1,
        int(round(hand_closure.shape[0] * float(target_fps) / float(source_fps))),
    )
    source_positions = torch.linspace(
        0,
        hand_closure.shape[0],
        target_intervals + 1,
        device=hand_closure.device,
        dtype=torch.float32,
    )[1:]
    lower = source_positions.floor().long()
    upper = (lower + 1).clamp(max=hand_closure.shape[0])
    alpha = (source_positions - lower.float()).unsqueeze(-1)
    return (1.0 - alpha) * extended[lower] + alpha * extended[upper]


class HumanoidArenaActionDecoder:
    """Decode HumanoidArena V3.1 40D reference actions into Kimodo G1 motion."""

    def __init__(self, skeleton: SkeletonBase, xml_path: str | Path, fps: float):
        self.skeleton = skeleton
        self.xml_path = Path(xml_path)
        self.fps = float(fps)
        self.mujoco_to_kimodo = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float32
        )
        self.kimodo_to_mujoco = self.mujoco_to_kimodo.T
        self._prepare_joint_mapping()

    def _prepare_joint_mapping(self) -> None:
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        class_axes = {}
        for default in tree.findall(".//default"):
            joints = default.findall("joint")
            if default.get("class") and joints and joints[0].get("axis"):
                class_axes[default.get("class")] = joints[0].get("axis")

        joints = root.find("worldbody").findall(".//joint")
        xml_names = [joint.get("name") for joint in joints]
        if set(xml_names) != set(CANONICAL_G1_JOINT_NAMES_29):
            raise ValueError("G1 XML joint names do not match HumanoidArena canonical 29-joint schema")
        self.canonical_to_xml = torch.tensor(
            [CANONICAL_G1_JOINT_NAMES_29.index(name) for name in xml_names], dtype=torch.long
        )
        self.xml_to_skeleton = torch.tensor(
            [self.skeleton.bone_order_names.index(name.replace("_joint", "_skel")) for name in xml_names],
            dtype=torch.long,
        )

        axes = []
        for joint in joints:
            axis_text = joint.get("axis") or class_axes[joint.get("class")]
            axis_mujoco = np.asarray([float(value) for value in axis_text.split()], dtype=np.float32)
            axis_kimodo = self.mujoco_to_kimodo @ torch.from_numpy(axis_mujoco)
            axes.append(axis_kimodo / axis_kimodo.norm().clamp_min(1e-8))
        self.axes_kimodo = torch.stack(axes)

        self.rot_offsets_f2q = torch.eye(3).repeat(self.skeleton.nbjoints, 1, 1)
        parent_map = {child: parent for parent in root.iter() for child in parent}
        coordinate_change = Rotation.from_matrix(self.mujoco_to_kimodo.numpy())
        for joint_index, joint in enumerate(joints):
            body = parent_map[joint]
            if "quat" not in body.attrib:
                continue
            quaternion_wxyz = [float(value) for value in body.get("quat").split()]
            rotation = Rotation.from_quat(
                [
                    quaternion_wxyz[1],
                    quaternion_wxyz[2],
                    quaternion_wxyz[3],
                    quaternion_wxyz[0],
                ]
            )
            rotation = coordinate_change * rotation * coordinate_change.inv()
            skeleton_index = int(self.xml_to_skeleton[joint_index])
            self.rot_offsets_f2q[skeleton_index] = torch.from_numpy(rotation.as_matrix().T).float()

    @staticmethod
    def _integrate_root(actions: torch.Tensor, root_rotations: torch.Tensor) -> torch.Tensor:
        positions = torch.zeros(actions.shape[0], 3, dtype=actions.dtype, device=actions.device)
        positions[:, 2] = actions[:, 2]
        for frame_index in range(1, actions.shape[0]):
            local_delta = torch.zeros(3, dtype=actions.dtype, device=actions.device)
            local_delta[:2] = actions[frame_index, :2]
            rotation = root_rotations[frame_index]
            delta_z = actions[frame_index, 2] - actions[frame_index - 1, 2]
            if rotation[2, 2].abs() > 1e-6:
                local_delta[2] = (
                    delta_z - rotation[2, 0] * local_delta[0] - rotation[2, 1] * local_delta[1]
                ) / rotation[2, 2]
            positions[frame_index, :2] = positions[frame_index - 1, :2] + (rotation @ local_delta)[:2]
        return positions

    def decode_action_pose(
        self, actions: np.ndarray | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Decode Arena actions into the pose inputs required by motion reps."""
        actions = torch.as_tensor(actions, dtype=torch.float32)
        if actions.ndim != 2 or actions.shape[-1] != 40:
            raise ValueError(f"Expected HumanoidArena action shape (T, 40), got {tuple(actions.shape)}")
        if not torch.isfinite(actions).all():
            raise ValueError("HumanoidArena action contains NaN or Inf")

        root_rot_mujoco = rot6d_row_to_matrix(actions[:, 3:9])
        root_positions_mujoco = self._integrate_root(actions, root_rot_mujoco)
        root_positions = (self.mujoco_to_kimodo @ root_positions_mujoco.T).T
        root_rot_f2q = torch.einsum(
            "ij,tjk,kl->til", self.mujoco_to_kimodo, root_rot_mujoco, self.kimodo_to_mujoco
        )
        root_local = torch.einsum("ij,tjk->tik", self.rot_offsets_f2q[0].T, root_rot_f2q)

        num_frames = actions.shape[0]
        local_rot_mats = torch.eye(3).reshape(1, 1, 3, 3).repeat(
            num_frames, self.skeleton.nbjoints, 1, 1
        )
        local_rot_mats[:, 0] = root_local
        joint_angles_xml = actions[:, 9:38][:, self.canonical_to_xml]
        for joint_index, skeleton_index in enumerate(self.xml_to_skeleton.tolist()):
            axis_angle = joint_angles_xml[:, joint_index, None] * self.axes_kimodo[joint_index]
            rotation_f2q = axis_angle_to_matrix(axis_angle)
            local_rot_mats[:, skeleton_index] = torch.einsum(
                "ij,tjk->tik", self.rot_offsets_f2q[skeleton_index].T, rotation_f2q
            )
        return {
            "local_rot_mats": local_rot_mats,
            "root_positions": root_positions,
        }

    def decode(self, actions: np.ndarray | torch.Tensor) -> dict[str, torch.Tensor]:
        pose = self.decode_action_pose(actions)
        return _complete_motion_dict(
            pose["local_rot_mats"],
            pose["root_positions"],
            self.skeleton,
            self.fps,
        )

    def decode_joint_configuration_pose(
        self,
        joint_positions: np.ndarray | torch.Tensor,
        root_positions: np.ndarray | torch.Tensor,
        root_quaternions: np.ndarray | torch.Tensor | None = None,
        root_rotation_matrices: np.ndarray | torch.Tensor | None = None,
        joint_names: tuple[str, ...] = UNITREE_G1_JOINT_NAMES_29,
        planar_origin: np.ndarray | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode an absolute G1 configuration into Kimodo pose coordinates.

        The source coordinate convention is Unitree/MuJoCo: x forward, y left,
        z up, with quaternions in ``(w, x, y, z)`` order. ``planar_origin`` is
        subtracted in the source frame before the basis change and should be
        shared by observed and desired trajectories from the same episode.
        """
        joint_positions = torch.as_tensor(joint_positions, dtype=torch.float32)
        root_positions = torch.as_tensor(root_positions, dtype=torch.float32)
        if joint_positions.ndim != 2 or joint_positions.shape[-1] != 29:
            raise ValueError(
                f"Expected G1 joint positions shape (T, 29), got {tuple(joint_positions.shape)}"
            )
        if root_positions.shape != (joint_positions.shape[0], 3):
            raise ValueError(
                f"Expected root positions shape {(joint_positions.shape[0], 3)}, "
                f"got {tuple(root_positions.shape)}"
            )
        if len(joint_names) != 29 or set(joint_names) != set(CANONICAL_G1_JOINT_NAMES_29):
            raise ValueError("joint_names must contain each canonical G1 joint exactly once")
        if (root_quaternions is None) == (root_rotation_matrices is None):
            raise ValueError(
                "Provide exactly one of root_quaternions or root_rotation_matrices"
            )
        if not torch.isfinite(joint_positions).all() or not torch.isfinite(root_positions).all():
            raise ValueError("G1 configuration contains NaN or Inf")
        if joint_positions.abs().max() > 4.0:
            raise ValueError(
                "G1 joint positions exceed 4 radians; the source joint order or units are likely wrong"
            )

        if root_quaternions is not None:
            root_quaternions = torch.as_tensor(root_quaternions, dtype=torch.float32)
            if root_quaternions.shape != (joint_positions.shape[0], 4):
                raise ValueError(
                    f"Expected root quaternions shape {(joint_positions.shape[0], 4)}, "
                    f"got {tuple(root_quaternions.shape)}"
                )
            if not torch.isfinite(root_quaternions).all():
                raise ValueError("Root quaternion contains NaN or Inf")
            quaternion_norm = root_quaternions.norm(dim=-1, keepdim=True)
            if (quaternion_norm < 1e-6).any():
                raise ValueError("Root quaternion has near-zero norm")
            root_rot_mujoco = quaternion_to_matrix(root_quaternions / quaternion_norm)
        else:
            root_rot_mujoco = torch.as_tensor(
                root_rotation_matrices, dtype=torch.float32
            )
            if root_rot_mujoco.shape != (joint_positions.shape[0], 3, 3):
                raise ValueError(
                    "Expected root rotation matrices shape "
                    f"{(joint_positions.shape[0], 3, 3)}, got {tuple(root_rot_mujoco.shape)}"
                )
            if not torch.isfinite(root_rot_mujoco).all():
                raise ValueError("Root rotation matrix contains NaN or Inf")

        if planar_origin is not None:
            planar_origin = torch.as_tensor(
                planar_origin, dtype=root_positions.dtype
            ).reshape(-1)
            if planar_origin.numel() not in (2, 3):
                raise ValueError("planar_origin must have two or three values")
            root_positions = root_positions.clone()
            root_positions[:, :2] -= planar_origin[:2]

        source_index = torch.tensor(
            [joint_names.index(name) for name in CANONICAL_G1_JOINT_NAMES_29],
            dtype=torch.long,
        )
        canonical_joint_positions = joint_positions[:, source_index]
        joint_angles_xml = canonical_joint_positions[:, self.canonical_to_xml]

        root_positions_kimodo = (
            self.mujoco_to_kimodo @ root_positions.T
        ).T
        root_rot_f2q = torch.einsum(
            "ij,tjk,kl->til",
            self.mujoco_to_kimodo,
            root_rot_mujoco,
            self.kimodo_to_mujoco,
        )
        root_local = torch.einsum(
            "ij,tjk->tik", self.rot_offsets_f2q[0].T, root_rot_f2q
        )

        num_frames = joint_positions.shape[0]
        local_rot_mats = torch.eye(3).reshape(1, 1, 3, 3).repeat(
            num_frames, self.skeleton.nbjoints, 1, 1
        )
        local_rot_mats[:, 0] = root_local
        for joint_index, skeleton_index in enumerate(self.xml_to_skeleton.tolist()):
            axis_angle = (
                joint_angles_xml[:, joint_index, None]
                * self.axes_kimodo[joint_index]
            )
            rotation_f2q = axis_angle_to_matrix(axis_angle)
            local_rot_mats[:, skeleton_index] = torch.einsum(
                "ij,tjk->tik",
                self.rot_offsets_f2q[skeleton_index].T,
                rotation_f2q,
            )
        return {
            "local_rot_mats": local_rot_mats,
            "root_positions": root_positions_kimodo,
        }

    def decode_joint_configuration(
        self,
        joint_positions: np.ndarray | torch.Tensor,
        root_positions: np.ndarray | torch.Tensor,
        root_quaternions: np.ndarray | torch.Tensor | None = None,
        root_rotation_matrices: np.ndarray | torch.Tensor | None = None,
        joint_names: tuple[str, ...] = UNITREE_G1_JOINT_NAMES_29,
        planar_origin: np.ndarray | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode an absolute G1 configuration and derive full motion fields."""
        pose = self.decode_joint_configuration_pose(
            joint_positions,
            root_positions,
            root_quaternions=root_quaternions,
            root_rotation_matrices=root_rotation_matrices,
            joint_names=joint_names,
            planar_origin=planar_origin,
        )
        return _complete_motion_dict(
            pose["local_rot_mats"],
            pose["root_positions"],
            self.skeleton,
            self.fps,
        )

    def encode(
        self,
        local_rot_mats: np.ndarray | torch.Tensor,
        root_positions: np.ndarray | torch.Tensor,
        previous_root_position: np.ndarray | torch.Tensor | None = None,
        hand_binary: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode Kimodo G1 motion into HumanoidArena's 40D semantic action."""
        local_rot_mats = torch.as_tensor(local_rot_mats, dtype=torch.float32)
        root_positions = torch.as_tensor(root_positions, dtype=torch.float32)
        if local_rot_mats.ndim != 4 or local_rot_mats.shape[1:] != (
            self.skeleton.nbjoints,
            3,
            3,
        ):
            raise ValueError(
                "Expected local_rot_mats shape "
                f"(T, {self.skeleton.nbjoints}, 3, 3), got {tuple(local_rot_mats.shape)}"
            )
        if root_positions.shape != (local_rot_mats.shape[0], 3):
            raise ValueError(
                f"Expected root_positions shape {(local_rot_mats.shape[0], 3)}, "
                f"got {tuple(root_positions.shape)}"
            )
        if not torch.isfinite(local_rot_mats).all() or not torch.isfinite(root_positions).all():
            raise ValueError("Kimodo motion contains NaN or Inf")

        device = local_rot_mats.device
        dtype = local_rot_mats.dtype
        mujoco_to_kimodo = self.mujoco_to_kimodo.to(device=device, dtype=dtype)
        kimodo_to_mujoco = self.kimodo_to_mujoco.to(device=device, dtype=dtype)
        rot_offsets = self.rot_offsets_f2q.to(device=device, dtype=dtype)
        axes_kimodo = self.axes_kimodo.to(device=device, dtype=dtype)
        canonical_to_xml = self.canonical_to_xml.to(device=device)

        root_rot_f2q = torch.einsum("ij,tjk->tik", rot_offsets[0], local_rot_mats[:, 0])
        root_rot_mujoco = torch.einsum(
            "ij,tjk,kl->til", kimodo_to_mujoco, root_rot_f2q, mujoco_to_kimodo
        )
        root_positions_mujoco = torch.einsum("ij,tj->ti", kimodo_to_mujoco, root_positions)

        previous_mujoco = None
        if previous_root_position is not None:
            previous_root_position = torch.as_tensor(
                previous_root_position, device=device, dtype=dtype
            ).reshape(3)
            previous_mujoco = kimodo_to_mujoco @ previous_root_position
        position_deltas = torch.zeros_like(root_positions_mujoco)
        if root_positions_mujoco.shape[0] > 0:
            if previous_mujoco is not None:
                position_deltas[0] = root_positions_mujoco[0] - previous_mujoco
            if root_positions_mujoco.shape[0] > 1:
                position_deltas[1:] = root_positions_mujoco[1:] - root_positions_mujoco[:-1]
        local_deltas = torch.einsum(
            "tji,tj->ti", root_rot_mujoco, position_deltas
        )

        num_frames = local_rot_mats.shape[0]
        actions = torch.zeros(num_frames, 40, device=device, dtype=dtype)
        actions[:, :2] = local_deltas[:, :2]
        actions[:, 2] = root_positions_mujoco[:, 2]
        actions[:, 3:9] = root_rot_mujoco[:, :, :2].reshape(num_frames, 6)

        joint_angles_xml = torch.zeros(num_frames, 29, device=device, dtype=dtype)
        for joint_index, skeleton_index in enumerate(self.xml_to_skeleton.tolist()):
            rotation_f2q = torch.einsum(
                "ij,tjk->tik", rot_offsets[skeleton_index], local_rot_mats[:, skeleton_index]
            )
            axis_angle = matrix_to_axis_angle(rotation_f2q)
            joint_angles_xml[:, joint_index] = torch.einsum(
                "ti,i->t", axis_angle, axes_kimodo[joint_index]
            )
        joint_angles_canonical = torch.zeros_like(joint_angles_xml)
        joint_angles_canonical[:, canonical_to_xml] = joint_angles_xml
        actions[:, 9:38] = joint_angles_canonical

        if hand_binary is not None:
            hand_binary = torch.as_tensor(hand_binary, device=device, dtype=dtype)
            if hand_binary.ndim == 1:
                hand_binary = hand_binary.reshape(1, 2).expand(num_frames, -1)
            if hand_binary.shape != (num_frames, 2):
                raise ValueError(
                    f"Expected hand_binary shape {(num_frames, 2)}, got {tuple(hand_binary.shape)}"
                )
            actions[:, 38:40] = hand_binary
        return actions
