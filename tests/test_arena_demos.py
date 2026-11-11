"""Arena demonstrations -> training episodes: images as the encoder sees them, root path, and the original loader's numbers."""

import os
import sys

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from kimodo_vla.benchmarks.arena import demos
from kimodo_vla.data.build import training_episode
from kimodo_vla.model.features import KimodoFeatures

# checkpoints/dinov3-vitl16-pretrain-lvd1689m/preprocessor_config.json
DINOV3_PREPROCESSOR = dict(
    do_resize=True, size={"height": 224, "width": 224}, resample=2, do_rescale=True, rescale_factor=1 / 255,
    do_normalize=True, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], do_center_crop=None,
)


def synthetic_episode(steps=300, seed=0):
    """Smooth 64-D states and 40-D actions in the dataset's layout, first heading at yaw 0."""
    rng = np.random.default_rng(seed)
    t = np.arange(steps) / 50.0
    yaw = 0.4 * np.sin(0.7 * t) + 0.3 * t
    yaw -= yaw[0]
    pitch = 0.05 * np.sin(1.3 * t)
    rot = Rotation.from_euler("zyx", np.stack([yaw, pitch, 0.03 * np.cos(t)], 1)).as_matrix().astype(np.float32)
    rot6d = rot[:, :, :2].reshape(steps, 6)
    joints = (0.3 * np.sin(t[:, None] * rng.uniform(0.5, 2.0, 29) + rng.uniform(0, 6, 29))).astype(np.float32)
    state = np.concatenate([rot6d, joints, np.gradient(joints, axis=0) * 50], 1).astype(np.float32)
    action = np.zeros((steps, 40), np.float32)
    action[:, 0] = 0.01 + 0.004 * np.sin(t)  # forward, in the current pelvis frame
    action[:, 1] = 0.002 * np.cos(2 * t)
    action[:, 2] = 0.79 + 0.02 * np.sin(0.5 * t)
    action[:, 3:9] = Rotation.from_euler("zyx", np.stack([yaw + 0.02, pitch, 0 * t], 1)).as_matrix()[:, :, :2].reshape(steps, 6)
    action[:, 9:38] = joints + 0.05 * np.sin(3 * t)[:, None]
    action[:, 38] = (t > 2.0).astype(np.float32)
    action[:, 39] = ((t > 1.0) & (t < 4.5)).astype(np.float32)
    return state, action


def test_stored_images_match_the_encoder_input():
    """224x224 uint8 images give DINOv3 the input of the full 480x640 frame, to half a gray level."""
    transformers = pytest.importorskip("transformers")
    processor = transformers.DINOv3ViTImageProcessorFast(**DINOV3_PREPROCESSOR)
    rng = np.random.default_rng(0)
    smooth = torch.nn.functional.interpolate(torch.from_numpy(rng.uniform(0, 255, (4, 3, 30, 40))).float(), size=(480, 640), mode="bilinear")
    full = (smooth + torch.from_numpy(rng.normal(0, 8, (4, 3, 480, 640))).float()).clamp(0, 255).to(torch.uint8)
    direct = processor(images=full, return_tensors="pt")["pixel_values"]
    small = torch.from_numpy(demos.resize_image(full.permute(0, 2, 3, 1).numpy())).permute(0, 3, 1, 2)
    stored = processor(images=small, return_tensors="pt")["pixel_values"]
    std = torch.tensor(DINOV3_PREPROCESSOR["image_std"]).view(1, 3, 1, 1)
    assert direct.shape == stored.shape == (4, 3, 224, 224)
    assert float(((direct - stored) * std * 255).abs().max()) <= 0.5 + 1e-3


def test_root_path_follows_the_reference():
    """Displacements are in the current pelvis frame: walking forward while turning left curves left."""
    steps = 101
    action = np.zeros((steps, 40), np.float32)
    action[:, 0] = 0.01
    action[:, 2] = 0.8
    yaw = np.linspace(0, np.pi / 2, steps)
    action[:, 3:9] = Rotation.from_euler("z", yaw[:, None]).as_matrix()[:, :, :2].reshape(steps, 6)
    motion = demos.ArenaDemo(np.zeros((steps, 64), np.float32), action, "go").commanded_motion()
    assert np.allclose(motion.root_pos[0], [0, 0, 0.8])
    assert np.allclose(np.linalg.norm(np.diff(motion.root_pos[:, :2], axis=0), axis=1), 0.01, atol=1e-6)
    radius = 1.0 / (np.pi / 2)  # a quarter circle of length 1 m
    assert np.allclose(motion.root_pos[-1, :2], [radius, radius], atol=0.01)


def test_history_is_what_the_server_builds():
    """The training history equals the planner's history for the same rows posted to the Arena server."""
    from kimodo_vla.benchmarks.arena.adapter import ArenaAdapter
    from kimodo_vla.core.types import yaw_matrix, yaw_of

    state, action = synthetic_episode(steps=200)
    features = KimodoFeatures(50.0)
    record, _ = training_episode(_NoImages(state, action, "t"), features, task="t", episode_id="e")
    image = {"shape": [2, 2, 3], "dtype": "uint8", "data_b64": "AAAAAAAAAAAAAAAA"}
    posted = ArenaAdapter().to_observations({"task": "t", "observation": {"images": {"front": image}, "state_history": state.tolist()}})
    for step, obs in enumerate(posted):
        obs.step = step
    served, mask, _, _ = features.history(posted, yaw_matrix(float(yaw_of(posted[0].root_rot))))
    assert torch.equal(record["observed_motion"], served.float())
    assert torch.equal(record["observed_mask"], mask[0])


class _NoImages(demos.ArenaDemo):
    def images(self, steps):
        return np.zeros((len(steps), 1, 2, 2, 3), np.uint8)


OLD_REPO = os.environ.get("KIMODO_ORIGINAL_REPO")


@pytest.mark.skipif(not OLD_REPO, reason="set KIMODO_ORIGINAL_REPO to a kimodo-policy checkout")
def test_matches_original_loader():
    """History, target and hands equal the original repository's HumanoidArena loader on the same episode."""
    sys.path.insert(0, OLD_REPO)
    from data import humanoidarena_loader as old
    from data.common import EpisodeRecord
    from motion.representation.kimodo_motionrep import KimodoMotionRep as OldRep
    from skeleton.definitions import G1Skeleton34 as OldSkeleton

    state, action = synthetic_episode()
    adapter = object.__new__(old.HumanoidArenaAdapter)
    adapter.target_fps, adapter.action_chunk, adapter._decoders = 30.0, 50, {}
    adapter._representation = OldRep(skeleton=OldSkeleton(), fps=30.0, stats_path=None)
    adapter.reader = type("Reader", (), {"read": lambda self, episode, columns: {"observation.state": state, "action": action}})()
    episode = EpisodeRecord(source="HumanoidArena", task_id="t", task_name="HOI_double_desk", instruction="t", episode_id="e",
                            data_path=None, source_length=len(state), source_fps=50.0, target_fps=30.0, video_path=None,
                            video_from_timestamp=0.0, first_cut=0, sample_count=1)
    reference = adapter.load_episode(episode)

    record, _ = training_episode(_NoImages(state, action, "t"), KimodoFeatures(50.0), task="t", episode_id="e")
    assert record["target_motion"].shape == reference["target_motion"].shape
    torch.testing.assert_close(record["target_motion"], reference["target_motion"].float(), atol=1e-4, rtol=0)
    mask = reference["observed_motion_valid"][0]
    assert torch.equal(record["observed_mask"], mask)
    torch.testing.assert_close(record["observed_motion"][:, mask], reference["observed_motion"][:, mask].float(), atol=1e-4, rtol=0)
    assert torch.equal(record["hand"], reference["target_hand"].float())
    assert torch.equal(record["hand"], reference["observed_hand"].float())
