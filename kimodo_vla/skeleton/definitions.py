# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unitree G1 skeleton definition used by Kimodo-Policy."""

from .base import SkeletonBase


class G1Skeleton34(SkeletonBase):
    """Unitree G1 skeleton with 32 articulated joints plus 2 toe endpoints."""

    name = "g1skel34"
    right_foot_joint_names = ["right_ankle_roll_skel", "right_toe_base"]
    left_foot_joint_names = ["left_ankle_roll_skel", "left_toe_base"]
    right_hand_joint_names = ["right_wrist_yaw_skel", "right_hand_roll_skel"]
    left_hand_joint_names = ["left_wrist_yaw_skel", "left_hand_roll_skel"]

    hip_joint_names = [
        "right_hip_pitch_skel",
        "left_hip_pitch_skel",
    ]  # used to calculate root orientation, only need 1 pair of hip joints

    bone_order_names_with_parents = [
        ("pelvis_skel", None),
        ("left_hip_pitch_skel", "pelvis_skel"),
        ("left_hip_roll_skel", "left_hip_pitch_skel"),
        ("left_hip_yaw_skel", "left_hip_roll_skel"),
        ("left_knee_skel", "left_hip_yaw_skel"),
        ("left_ankle_pitch_skel", "left_knee_skel"),
        ("left_ankle_roll_skel", "left_ankle_pitch_skel"),
        ("left_toe_base", "left_ankle_roll_skel"),
        ("right_hip_pitch_skel", "pelvis_skel"),
        ("right_hip_roll_skel", "right_hip_pitch_skel"),
        ("right_hip_yaw_skel", "right_hip_roll_skel"),
        ("right_knee_skel", "right_hip_yaw_skel"),
        ("right_ankle_pitch_skel", "right_knee_skel"),
        ("right_ankle_roll_skel", "right_ankle_pitch_skel"),
        ("right_toe_base", "right_ankle_roll_skel"),
        ("waist_yaw_skel", "pelvis_skel"),
        ("waist_roll_skel", "waist_yaw_skel"),
        ("waist_pitch_skel", "waist_roll_skel"),
        ("left_shoulder_pitch_skel", "waist_pitch_skel"),
        ("left_shoulder_roll_skel", "left_shoulder_pitch_skel"),
        ("left_shoulder_yaw_skel", "left_shoulder_roll_skel"),
        ("left_elbow_skel", "left_shoulder_yaw_skel"),
        ("left_wrist_roll_skel", "left_elbow_skel"),
        ("left_wrist_pitch_skel", "left_wrist_roll_skel"),
        ("left_wrist_yaw_skel", "left_wrist_pitch_skel"),
        ("left_hand_roll_skel", "left_wrist_yaw_skel"),
        ("right_shoulder_pitch_skel", "waist_pitch_skel"),
        ("right_shoulder_roll_skel", "right_shoulder_pitch_skel"),
        ("right_shoulder_yaw_skel", "right_shoulder_roll_skel"),
        ("right_elbow_skel", "right_shoulder_yaw_skel"),
        ("right_wrist_roll_skel", "right_elbow_skel"),
        ("right_wrist_pitch_skel", "right_wrist_roll_skel"),
        ("right_wrist_yaw_skel", "right_wrist_pitch_skel"),
        ("right_hand_roll_skel", "right_wrist_yaw_skel"),
    ]
