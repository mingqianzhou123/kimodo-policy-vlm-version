"""BiGym 2.0 demonstrations as whole-body motion at the control rate.

A demonstration stores, per control step, the full simulator state
(``full_qpos``) and the command that produced it (``action``). Kimodo needs a
whole-body pose trajectory; there are two ways to read one off a demo:

* ``executed_motion``: what the robot actually did (legs from GR00T-WBC,
  arms after PD tracking).
* ``commanded_motion``: what the operator asked for: the arm and torso-pitch
  targets, and the pelvis path integrated from the base velocity commands.

``scripts/check_bigym2_roundtrip.py`` showed the commanded one is the right
training target (executed motion applies arm sag twice and fails almost every
task). ``training_episode`` turns a demo into the files ``kimodo_vla.data``
trains on.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Mapping

import mujoco
import numpy as np
import torch

from kimodo_vla.benchmarks.bigym2.adapter import ARM_JOINT_NAMES, BiGym2Adapter
from kimodo_vla.core.types import G1_JOINT_NAMES, joint_indices, yaw_matrix, yaw_of
from kimodo_vla.model.features import KimodoFeatures
from kimodo_vla.motion.g1_reference import resample_hand_binary


@dataclass
class DemoMotion:
    """Row ``t`` is the pose at control step ``t`` of the episode (50 Hz)."""

    root_pos: np.ndarray   # (T, 3)
    root_rot: np.ndarray   # (T, 3, 3)
    joint_pos: np.ndarray  # (T, 29) G1_JOINT_NAMES order
    hand: np.ndarray       # (T, 2) commanded gripper, 0 open / 1 closed


def raw_actions(episode: Mapping[str, np.ndarray], action_stats: Mapping[str, list]) -> np.ndarray:
    """Recorded actions in physical units; row ``t`` produced state row ``t``."""
    low = np.asarray(action_stats["min"], np.float32)
    high = np.asarray(action_stats["max"], np.float32)
    normalized = np.asarray(episode["action"], np.float32)
    return low + (normalized + 1.0) * 0.5 * (high - low)


def executed_motion(env, episode: Mapping[str, np.ndarray], action_stats: Mapping[str, list]) -> DemoMotion:
    model = env.inner_env.model
    namespace = env.inner_env.robot.namespace
    data = mujoco.MjData(model)
    pelvis = model.body(f"{namespace}pelvis").id
    qpos_index = np.asarray([model.joint(f"{namespace}{name}").qposadr[0] for name in G1_JOINT_NAMES])

    full_qpos = np.asarray(episode["full_qpos"], np.float64)
    frames = len(full_qpos)
    root_pos = np.zeros((frames, 3), np.float32)
    root_rot = np.zeros((frames, 3, 3), np.float32)
    for t, qpos in enumerate(full_qpos):
        data.qpos[: len(qpos)] = qpos
        mujoco.mj_kinematics(model, data)
        root_pos[t] = data.xpos[pelvis]
        root_rot[t] = data.xmat[pelvis].reshape(3, 3)
    grippers = raw_actions(episode, action_stats)[:, -2:]
    grippers[0] = grippers[1]  # row 0 is the reset row, no command produced it
    return DemoMotion(
        root_pos=root_pos,
        root_rot=root_rot,
        joint_pos=full_qpos[:, qpos_index].astype(np.float32),
        hand=(grippers >= 0.5).astype(np.float32),
    )


def commanded_motion(executed: DemoMotion, episode: Mapping[str, np.ndarray], action_stats: Mapping[str, list],
                     adapter: BiGym2Adapter) -> DemoMotion:
    """``executed`` with the arms, torso pitch and pelvis path replaced by the commands."""
    raw = raw_actions(episode, action_stats)
    raw[0] = raw[1]
    dt = 1.0 / adapter.control_fps
    vx, vy, height, wz = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    base_dim = 5 if adapter.pitch_command else 4

    yaw = yaw_of(executed.root_rot[0]) + np.concatenate([[0.0], np.cumsum(wz[1:]) * dt])
    root_pos = np.zeros_like(executed.root_pos)
    root_pos[0] = executed.root_pos[0]
    for t in range(1, len(raw)):
        heading = yaw_matrix(float(yaw[t]))[:2, :2]
        root_pos[t, :2] = root_pos[t - 1, :2] + heading @ np.array([vx[t], vy[t]]) * dt
    root_pos[:, 2] = height

    joint_pos = executed.joint_pos.copy()
    joint_pos[:, joint_indices(ARM_JOINT_NAMES)] = raw[:, base_dim : base_dim + 14]
    if adapter.pitch_command:
        joint_pos[:, joint_indices(["waist_pitch_joint"])[0]] = raw[:, 4]
    return replace(
        executed,
        root_pos=root_pos.astype(np.float32),
        root_rot=np.stack([yaw_matrix(float(y)) for y in yaw]),
        joint_pos=joint_pos,
    )


def training_episode(env, adapter: BiGym2Adapter, kimodo: KimodoFeatures, episode: Mapping[str, np.ndarray],
                     action_stats: Mapping[str, list]) -> tuple[dict, np.ndarray]:
    """One demo as a training record at model rate, plus the control step whose image goes with each frame.

    The history goes through the same ``to_observation`` and ``KimodoFeatures.history``
    calls as evaluation, so training sees exactly what the policy will see.
    """
    no_images = np.zeros((0, 3, 1, 1), np.uint8)
    observations = []
    for step, low_dim_obs in enumerate(episode["low_dim_obs"]):
        obs = adapter.to_observation(SimpleNamespace(low_dim_obs=low_dim_obs, rgb_obs=no_images))
        obs.step = step
        observations.append(obs)
    first_heading = yaw_matrix(float(yaw_of(observations[0].root_rot)))
    observed, mask, _, _ = kimodo.history(observations, first_heading)

    commanded = commanded_motion(executed_motion(env, episode, action_stats), episode, action_stats, adapter)
    target = kimodo.target(commanded.joint_pos, commanded.root_pos, commanded.root_rot, first_heading)
    hand = resample_hand_binary(commanded.hand, adapter.control_fps, kimodo.model_fps)
    if not (observed.shape == target.shape and hand.shape[0] == target.shape[0]):
        raise RuntimeError(f"length mismatch: observed {tuple(observed.shape)}, target {tuple(target.shape)}, hand {tuple(hand.shape)}")

    # Frame k of the model-rate grid is control step round(k * control / model): the step at which
    # evaluation makes its k-th cut (round(step * model / control) == k).
    steps = np.minimum(
        np.round(np.arange(target.shape[0]) * adapter.control_fps / kimodo.model_fps).astype(int),
        len(observations) - 1,
    )
    record = {  # the caller adds "task" and "episode_id"
        "instruction": adapter.instruction,
        "fps": kimodo.model_fps,
        "observed_motion": observed.float(),
        "observed_mask": mask[0].clone(),
        "target_motion": target.float(),
        "hand": torch.as_tensor(hand, dtype=torch.float32),
    }
    return record, steps


def render_frames(render_env, episode: Mapping[str, np.ndarray], steps: np.ndarray, camera: str = "head") -> np.ndarray:
    """``[len(steps), H, W, 3]`` images of ``camera`` at the given control steps, by kinematic replay.

    ``render_env`` is ``make(task, camera_keys=(camera,), camera_shape=...)``; posing the
    stored ``full_qpos`` reproduces the recorded scene without running physics.
    """
    from bigym.loco.demos.kinematic import pose

    render_env.reset(seed=int(np.asarray(episode["seed"]).reshape(-1)[0]))
    inner = render_env.inner_env
    qvel = episode.get("full_qvel")
    frames = []
    for step in steps:
        pose(inner, episode["full_qpos"][step], None if qvel is None else qvel[step])
        frames.append(np.asarray(inner.get_visual_obs()[f"rgb_{camera}"]).transpose(1, 2, 0))
    return np.stack(frames)
