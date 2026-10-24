"""BiGym 2.0 demonstrations as whole-body motion at the control rate.

A demonstration stores, per control step, the full simulator state
(``full_qpos``) and the command that produced it (``action``). Kimodo needs a
whole-body pose trajectory; there are two ways to read one off a demo:

* ``executed_motion``: what the robot actually did (legs from GR00T-WBC,
  arms after PD tracking).
* ``commanded_motion``: what the operator asked for: the arm and torso-pitch
  targets, and the pelvis path integrated from the base velocity commands.

Which one makes the better training target is exactly what
``scripts/check_bigym2_roundtrip.py`` measures.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import mujoco
import numpy as np

from kimodo_vla.benchmarks.bigym2.adapter import ARM_JOINT_NAMES, BiGym2Adapter
from kimodo_vla.core.types import G1_JOINT_NAMES, joint_indices, yaw_matrix, yaw_of


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
