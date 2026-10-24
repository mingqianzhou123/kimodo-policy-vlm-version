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
task). ``BiGym2Demo`` hands a demo to ``kimodo_vla.data.build.training_episode``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from kimodo_vla.benchmarks.bigym2.adapter import ARM_JOINT_NAMES, BiGym2Adapter
from kimodo_vla.controllers.groot_wbc import BaseCommand, pelvis_path
from kimodo_vla.core.types import G1_JOINT_NAMES, Observation, joint_indices, yaw_of
from kimodo_vla.data.build import DemoMotion


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
    base_dim = 5 if adapter.pitch_command else 4
    base = BaseCommand(vx=raw[:, 0], vy=raw[:, 1], yaw_rate=raw[:, 3], height=raw[:, 2])
    root_pos, root_rot = pelvis_path(base, executed.root_pos[0], yaw_of(executed.root_rot[0]), adapter.control_fps)

    joint_pos = executed.joint_pos.copy()
    joint_pos[:, joint_indices(ARM_JOINT_NAMES)] = raw[:, base_dim : base_dim + 14]
    if adapter.pitch_command:
        joint_pos[:, joint_indices(["waist_pitch_joint"])[0]] = raw[:, 4]
    return replace(executed, root_pos=root_pos, root_rot=root_rot, joint_pos=joint_pos)


def demo_frames(episode: Mapping[str, np.ndarray], steps: np.ndarray) -> np.ndarray:
    """``[len(steps), cameras, H, W, 3]`` recorded images of every camera at the given control steps.

    These are the official observations (84x84), the same pixels every baseline
    trains on; DINOv3's processor upsamples them to 224 inside the model.
    """
    return np.stack([episode["rgb_obs"].decode(int(step)).transpose(0, 2, 3, 1) for step in steps])


@dataclass
class BiGym2Demo:
    """One BiGym demo as a ``benchmarks.base.Demonstration``."""

    env: Any
    adapter: BiGym2Adapter
    episode: Mapping[str, np.ndarray]
    action_stats: Mapping[str, list]
    cameras: Sequence[str]  # the dataset's camera_keys, in rgb_obs order

    @property
    def instruction(self) -> str:
        return self.adapter.instruction

    @property
    def control_fps(self) -> float:
        return self.adapter.control_fps

    def observations(self) -> list[Observation]:
        no_images = np.zeros((0, 3, 1, 1), np.uint8)
        observations = []
        for step, low_dim_obs in enumerate(self.episode["low_dim_obs"]):
            obs = self.adapter.to_observation(SimpleNamespace(low_dim_obs=low_dim_obs, rgb_obs=no_images))
            obs.step = step
            observations.append(obs)
        return observations

    def commanded_motion(self) -> DemoMotion:
        executed = executed_motion(self.env, self.episode, self.action_stats)
        return commanded_motion(executed, self.episode, self.action_stats, self.adapter)

    def images(self, steps: np.ndarray) -> np.ndarray:
        return demo_frames(self.episode, steps)
