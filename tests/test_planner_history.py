"""History encoding and masks of KimodoPlanner; needs no checkpoint."""

from types import SimpleNamespace

import numpy as np
import torch

from kimodo_vla.core.types import G1_JOINT_INDEX, Observation, yaw_matrix
from kimodo_vla.model.planner import KimodoPlanner, history_feature_mask


def stub_planner():
    config = SimpleNamespace(hand_control_mode="binary", action_chunk=50, action_history=100)
    model = SimpleNamespace(config=config, fps=30, text_encoder=None)
    return KimodoPlanner(model, {}, control_fps=50.0)


def observation(yaw, joint_mask):
    joints = np.zeros(29, np.float32)
    joints[G1_JOINT_INDEX["left_elbow_joint"]] = 0.5
    return Observation(
        images={"head": np.zeros((4, 4, 3), np.uint8)},
        root_pos=np.zeros(3, np.float32),
        root_rot=yaw_matrix(yaw),
        joint_pos=joints,
        joint_mask=joint_mask,
        hand=np.zeros(2, np.float32),
        instruction="test",
    )


def rot_slot_observed(planner, mask, skeleton_name):
    representation = planner.state_representation
    j = representation.skeleton.bone_order_names.index(skeleton_name)
    start = representation.slice_dict["global_rot_data"].start + 6 * j
    return bool(mask[start : start + 6].all())


def test_full_observation_reproduces_original_mask():
    planner = stub_planner()
    representation = planner.state_representation
    mask = history_feature_mask(representation, np.ones(29, bool))
    expected = torch.zeros(representation.motion_rep_dim, dtype=torch.bool)
    expected[representation.slice_dict["global_root_heading"]] = True
    expected[representation.slice_dict["global_rot_data"]] = True
    assert torch.equal(mask, expected)


def test_unreported_joints_are_masked_and_end_points_follow_parents():
    planner = stub_planner()
    joint_mask = np.zeros(29, bool)
    for name in ("left_elbow_joint", "left_wrist_yaw_joint"):
        joint_mask[G1_JOINT_INDEX[name]] = True
    mask = history_feature_mask(planner.state_representation, joint_mask)
    assert rot_slot_observed(planner, mask, "pelvis_skel")
    assert rot_slot_observed(planner, mask, "left_elbow_skel")
    assert rot_slot_observed(planner, mask, "left_hand_roll_skel")
    assert not rot_slot_observed(planner, mask, "left_knee_skel")
    assert not rot_slot_observed(planner, mask, "left_toe_base")
    assert not rot_slot_observed(planner, mask, "waist_pitch_skel")


def test_history_is_resampled_to_model_rate_and_heading_relative():
    planner = stub_planner()
    planner.reset()
    joint_mask = np.ones(29, bool)
    history = [observation(0.3 + 0.002 * i, joint_mask) for i in range(51)]  # 1 s at 50 Hz
    planner._first_heading = yaw_matrix(0.3)
    features, mask, local_rot, _ = planner.encode_history(history)
    assert features.shape[0] == 31  # 1 s at 30 Hz, inclusive
    assert mask.shape == features.shape
    heading = features[:, planner.state_representation.slice_dict["global_root_heading"]]
    turn = torch.atan2(heading[-1, 1], heading[-1, 0]) - torch.atan2(heading[0, 1], heading[0, 0])
    assert abs(abs(float(turn)) - 0.1) < 1e-2  # 50 steps x 0.002 rad
