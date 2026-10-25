"""Training windows: format, and identical history inputs to what the planner builds at evaluation."""

from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from kimodo_vla.core.types import G1_JOINT_INDEX, Observation, yaw_matrix
from kimodo_vla.data.episodes import KimodoWindowDataset, save_episode
from kimodo_vla.model.planner import KimodoPlanner


def stub_planner():
    config = SimpleNamespace(hand_control_mode="binary", action_chunk=50, action_history=100)
    return KimodoPlanner(SimpleNamespace(config=config, fps=30, text_encoder=None), {}, control_fps=50.0)


def synthetic_episode(steps=400, seed=0):
    """Observations like BiGym's: arms and the root heading move, legs and waist are not reported."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, 29)
    joint_mask = np.zeros(29, bool)
    for name in G1_JOINT_INDEX:
        joint_mask[G1_JOINT_INDEX[name]] = "shoulder" in name or "elbow" in name or "wrist" in name
    observations = []
    for step in range(steps):
        t = step / 50
        joints = np.where(joint_mask, 0.3 * np.sin(0.8 * t + phase), 0.0).astype(np.float32)
        observations.append(Observation(
            images={}, root_pos=np.zeros(3, np.float32), root_rot=yaw_matrix(0.5 + 0.4 * np.sin(0.3 * t)),
            joint_pos=joints, joint_mask=joint_mask, hand=np.zeros(2, np.float32), instruction="pick it", step=step,
        ))
    return observations


def write_dataset(tmp_path, observations, planner):
    first_heading = yaw_matrix(0.5)
    observed, mask, _, _ = planner.kimodo.history(observations, first_heading)
    frames = observed.shape[0]
    target = observed + 1.0  # any distinct values; only slicing is under test here
    hand = torch.zeros(frames, 2)
    hand[frames // 2 :, 0] = 1
    # Pixel value = frame index + 100 * camera index, to check which image a sample gets.
    images = np.arange(frames, dtype=np.uint8)[:, None] + np.array([0, 100, 200], dtype=np.uint8)[None, :]
    images = np.broadcast_to(images[:, :, None, None, None], (frames, 3, 4, 4, 3))
    record = {
        "task": "toy", "episode_id": "ep0", "instruction": "pick it", "fps": 30.0,
        "cameras": ["head", "right_wrist", "left_wrist"],
        "observed_motion": observed, "observed_mask": mask[0].clone(), "target_motion": target, "hand": hand,
    }
    save_episode(tmp_path / "toy" / "ep0", record, images)
    return KimodoWindowDataset([tmp_path], text_embeddings={"pick it": torch.ones(1, 8)}), record


def test_window_format_and_padding(tmp_path):
    planner = stub_planner()
    dataset, record = write_dataset(tmp_path, synthetic_episode(), planner)
    assert len(dataset) == record["target_motion"].shape[0] - 50 + 1

    first = dataset[0]  # cut 0: no history at all
    assert first["gt_motion"].shape == (150, 417) and first["condition_motion_mask"].shape == (150, 417)
    assert not first["gt_mask"][:100].any() and first["gt_mask"][100:].all()
    assert first["egoview"].shape == (3, 4, 4) and first["egoview"].dtype == torch.uint8
    assert first["text_embedding"].shape == (1, 8) and int(first["text_length"]) == 1

    sample = dataset[130]
    assert sample["gt_mask"].all()
    assert int(sample["egoview"][0, 0, 0]) == 130  # the image at the cut
    torch.testing.assert_close(sample["gt_hand"], record["hand"][30:180])
    # Translation is canonicalized: the first target root x/z sits at the origin.
    torch.testing.assert_close(sample["gt_motion"][0, [0, 2]], torch.zeros(2))


def test_training_history_equals_planner_history(tmp_path):
    planner = stub_planner()
    observations = synthetic_episode()
    dataset, _ = write_dataset(tmp_path, observations, planner)
    planner.reset()
    planner._first_heading = yaw_matrix(0.5)
    window = deque(maxlen=planner.history_steps)
    checked = 0
    for obs in observations:
        window.append(obs)
        step = obs.step
        cut = int(round(step * 30 / 50))
        if step % 37 or cut > len(dataset) - 1:  # a spread of steps, before and after the window fills
            continue
        motion, mask, _, _ = planner.condition_history(list(window))
        sample = dataset[cut]
        n = motion.shape[0]
        assert n == min(cut, 100)
        train_mask = sample["condition_motion_mask"][100 - n : 100]
        assert torch.equal(train_mask, mask)
        torch.testing.assert_close(sample["condition_motion"][100 - n : 100][mask], motion[mask], atol=1e-5, rtol=0)
        checked += 1
    assert checked >= 5


def test_camera_selection(tmp_path):
    _, record = write_dataset(tmp_path, synthetic_episode(steps=200), stub_planner())
    both = KimodoWindowDataset([tmp_path], cameras=("head", "left_wrist"))
    images = both[10]["egoview"]
    assert images.shape == (2, 3, 4, 4)
    assert int(images[0, 0, 0, 0]) == 10 and int(images[1, 0, 0, 0]) == 210


def test_image_shift_is_a_crop_of_the_padded_image(tmp_path):
    write_dataset(tmp_path, synthetic_episode(steps=200), stub_planner())
    dataset = KimodoWindowDataset([tmp_path], image_shift=2)
    images = torch.arange(2 * 3 * 6 * 6, dtype=torch.uint8).reshape(2, 3, 6, 6)
    padded = torch.nn.functional.pad(images.float(), (2, 2, 2, 2), mode="replicate")
    for _ in range(20):
        shifted = dataset._shift(images)
        assert shifted.shape == images.shape and shifted.dtype == torch.uint8
        for view in range(2):
            crops = [padded[view, :, dy : dy + 6, dx : dx + 6] for dy in range(5) for dx in range(5)]
            assert any(torch.equal(shifted[view].float(), crop) for crop in crops)


def test_proprio_masking_keeps_only_the_base(tmp_path):
    _, record = write_dataset(tmp_path, synthetic_episode(steps=400), stub_planner())
    plain = KimodoWindowDataset([tmp_path])
    masked = KimodoWindowDataset([tmp_path], proprio_mask_prob=1.0)
    representation = stub_planner().kimodo.representation
    heading = representation.slice_dict["global_root_heading"]
    i = len(plain) - 1  # a late cut: the whole 100-frame history is real
    a, b = plain[i]["condition_motion_mask"], masked[i]["condition_motion_mask"]
    assert bool(a[:100].any(dim=1).all())
    assert torch.equal(a[100:], b[100:])  # the future carries no condition either way
    assert bool(b[:100, heading].all())  # the base stays observed
    assert int(b[:100].sum()) < int(a[:100].sum())  # the arm joints do not
    assert bool((b <= a).all())
