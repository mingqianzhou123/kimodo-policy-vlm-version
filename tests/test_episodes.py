"""Training windows: format, and identical history inputs to what the planner builds at evaluation."""

from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from kimodo_vla.core.types import G1_JOINT_INDEX, Observation, yaw_matrix
from kimodo_vla.data.build import DemoMotion, training_episode
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


class SyntheticDemo:
    """A ``benchmarks.base.Demonstration`` whose commands are exactly what it observed."""

    instruction, control_fps, cameras, hand_mode = "pick it", 50.0, ("head", "right_wrist"), "binary"

    def __init__(self, observations):
        self._observations = observations

    def observations(self):
        return self._observations

    def commanded_motion(self):
        obs = self._observations
        return DemoMotion(
            root_pos=np.stack([o.root_pos for o in obs]), root_rot=np.stack([o.root_rot for o in obs]),
            joint_pos=np.stack([o.joint_pos for o in obs]), hand=np.zeros((len(obs), 2), np.float32),
        )

    def images(self, steps):  # pixel value = control step
        return np.broadcast_to(steps.astype(np.uint8)[:, None, None, None, None], (len(steps), 2, 4, 4, 3))


def test_training_episode_from_a_demonstration(tmp_path):
    planner = stub_planner()
    observations = synthetic_episode(steps=200)
    record, frames = training_episode(SyntheticDemo(observations), planner.kimodo, task="toy", episode_id="ep0")
    frames_count = record["target_motion"].shape[0]
    assert record["observed_motion"].shape == (frames_count, 417) and record["hand"].shape == (frames_count, 2)
    assert record["cameras"] == ["head", "right_wrist"] and record["task"] == "toy"
    # Model frame k carries the image of control step round(k * 50 / 30).
    assert frames.shape[:2] == (frames_count, 2)
    assert [int(frames[k, 0, 0, 0, 0]) for k in (0, 1, 2, 3)] == [0, 2, 3, 5]
    # Commands equal to the observation: the target's rotations are the observed history's.
    rot = planner.kimodo.representation.slice_dict["global_rot_data"]
    observed_mask = record["observed_mask"][rot]
    torch.testing.assert_close(record["target_motion"][:, rot][:, observed_mask],
                               record["observed_motion"][:, rot][:, observed_mask], atol=1e-5, rtol=0)
    save_episode(tmp_path / "toy" / "ep0", record, frames)
    assert len(KimodoWindowDataset([tmp_path])) == frames_count - 50 + 1
