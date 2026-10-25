"""HumanoidToolBench demonstrations (ToolBook, ``snupilab/humanoidtoolbench-teleop``) as training episodes.

A recording is a LeRobot dataset at 50 Hz. Row ``t`` holds the state the
operator saw, the command issued from it (reaching step ``t + 1``) and the
outcome (``meta/simple_contract.json``). Columns used::

    observation.state (43)    29 body joints (legs, waist yaw/roll/pitch, left arm, right arm),
                              then left and right hand: thumb (3), index (2), middle (2)
    action (43)               [0:15] legs and waist, [15:22] left arm, [22:29] left hand,
                              [29:36] right arm, [36:43] right hand; hands index (2), middle (2), thumb (3)
    teleop.navigate_command   vx, vy, turn flag (the turning rate while turning), target heading (absolute)
    teleop.base_height_command
    observation.images.ego_view   the head camera, the one image the evaluator sends

The training target is the operator's command, as for BiGym (ARCHITECTURE
section 10). The observations mirror what ``ToolBenchAdapter`` reports at
evaluation, where the evaluator sends one state per executed chunk: the measured
state every ``request_interval`` steps, the commanded targets in between, and
the commanded heading throughout (the state has no pelvis pose).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from kimodo_vla.benchmarks.toolbench.adapter import (
    CONTROL_FPS,
    SPAWN_YAW,
    commanded_observation,
    hand_closure,
    measured_observation,
    resize_image,
)
from kimodo_vla.controllers import decoupled_wbc
from kimodo_vla.core.types import Observation, joint_indices
from kimodo_vla.data.build import DemoMotion

BODY_JOINT_NAMES = (  # observation.state[:29] and action[:15] start this way
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    *decoupled_wbc.ARM_JOINT_NAMES,
)
# The evaluator's instruction per scenario, L0 (pick the tool) and L1/L2 (use it): humanoidtoolbench/instructions.py.
INSTRUCTIONS = {
    "G1BallMove": ("Pick the tool for moving the ball to the target.", "Pick the tool and move the ball to the target."),
    "G1BallRetrieve": ("Pick the tool for retrieving the ball to the target.", "Pick the tool and retrieve the ball to the target."),
    "G1IceBreak": ("Pick the tool for breaking the ice blocks.", "Pick the tool and break the ice blocks."),
}
RENAMED = {"G1StickMove": "G1BallMove", "G1HookRetrieve": "G1BallRetrieve"}
# Steps per evaluator request: the planner's 50 executed model frames at the 50 Hz control rate.
REQUEST_INTERVAL = round(50 * CONTROL_FPS / 30.0)


def condition(name: str) -> tuple[str, int, str]:
    """``"G1StickMove-L0-S-v0"`` -> ``("G1BallMove", 0, "S")``."""
    match = re.fullmatch(r"(G1\w+?)-L(\d)-([SR])(?:-v\d+)?", name)
    if match is None:
        raise ValueError(f"not a ToolBench condition: {name!r}")
    scenario, level, mode = match.groups()
    return RENAMED.get(scenario, scenario), int(level), mode


def canonical(name: str) -> str:
    """``"G1HookRetrieve-L0-R-v0"`` -> ``"G1BallRetrieve-L0-R"``: the evaluator's condition name."""
    scenario, level, mode = condition(name)
    return f"{scenario}-L{level}-{mode}"


def instruction(name: str) -> str:
    scenario, level, _ = condition(name)
    return INSTRUCTIONS[scenario][0 if level == 0 else 1]


def action_hands_to_natural(hand: np.ndarray) -> np.ndarray:
    """``(T, 7)`` recorded hand command (index, middle, thumb) -> natural order (thumb, index, middle)."""
    return hand[:, [4, 5, 6, 0, 1, 2, 3]]


def policy_state(state43: np.ndarray, previous_height: np.ndarray) -> np.ndarray:
    """``(T, 32)``: the evaluator's state from recorded ``observation.state`` (``policies/remote_humanoid.py``)."""
    state43 = np.asarray(state43, np.float32)
    hand_q = state43[:, 29:43]  # natural order
    return np.concatenate([
        decoupled_wbc.natural_to_command_order(hand_q),
        state43[:, 15:29],
        state43[:, [13, 14, 12]],  # torso roll, pitch, yaw
        np.asarray(previous_height, np.float32).reshape(-1, 1),
    ], axis=1)


def read_recording(root: Path, episode: int) -> dict[str, np.ndarray]:
    """The columns of one episode, ``(T, ...)`` float arrays."""
    import pyarrow.parquet as pq

    info = json.loads((root / "meta/info.json").read_text())
    path = root / info["data_path"].format(episode_chunk=episode // info["chunks_size"], episode_index=episode)
    table = pq.read_table(path, columns=["observation.state", "action", "teleop.navigate_command",
                                         "teleop.base_height_command", "next.success"])
    return {name: np.stack(table.column(name).to_numpy(zero_copy_only=False)).astype(np.float64) for name in table.column_names}


def episodes(root: Path) -> list[dict[str, Any]]:
    """``meta/episodes.jsonl`` of a recording."""
    return [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]


class ToolBenchDemo:
    """One ToolBook episode as a ``benchmarks.base.Demonstration``."""

    control_fps = CONTROL_FPS
    cameras = ("head",)
    hand_mode = "continuous"  # open/closed commands with short ramps between

    def __init__(self, columns: Mapping[str, np.ndarray], instruction: str, video: Path | None = None,
                 request_interval: int = REQUEST_INTERVAL):
        self.instruction = instruction
        self.video = video
        self.request_interval = int(request_interval)
        self.state = np.asarray(columns["observation.state"], np.float32)
        command = np.asarray(columns["action"], np.float32)
        nav = np.asarray(columns["teleop.navigate_command"], np.float64)
        height = np.asarray(columns["teleop.base_height_command"], np.float64).reshape(-1)
        # Row t of a recording commands step t + 1; step 0 is where the episode starts.
        target = lambda x: np.concatenate([x[:1], x[:-1]])  # noqa: E731
        self.command, self.nav, self.height = target(command), target(nav), target(height)
        self.heading = self.nav[:, 3] - SPAWN_YAW  # episode frame: 0 = the spawn heading
        # The evaluator's "previous base-height command" at step t is the one that produced step t.
        self.previous_height = np.concatenate([[0.74], height[:-1]])

    def __len__(self) -> int:
        return len(self.state)

    def commanded_hands(self) -> np.ndarray:
        """``(T, 14)`` commanded hand joints, natural order."""
        return np.concatenate([action_hands_to_natural(self.command[:, 22:29]),
                               action_hands_to_natural(self.command[:, 36:43])], axis=1)

    def recorded_commands(self) -> np.ndarray:
        """``(T, 36)``: row ``t`` is the operator's decoupled-WBC command for step ``t``, as ToolBench's
        released ACT model converts a recording (its ``action_contract.json``)."""
        command = np.zeros((len(self), decoupled_wbc.DIMENSION), np.float32)
        command[:, 0:14] = decoupled_wbc.natural_to_command_order(self.commanded_hands())
        command[:, 14:21] = self.command[:, 15:22]
        command[:, 21:28] = self.command[:, 29:36]
        command[:, 28:31] = self.command[:, [13, 14, 12]]  # torso roll, pitch, yaw
        command[:, 31] = self.height
        command[:, 32:36] = self.nav
        return command

    def commanded_motion(self) -> DemoMotion:
        joint_pos = np.zeros((len(self), 29), np.float32)
        body = joint_indices(BODY_JOINT_NAMES)
        joint_pos[:, body] = self.state[:, :29]  # the legs belong to the controller: as measured
        joint_pos[:, joint_indices(BODY_JOINT_NAMES[12:15])] = self.command[:, 12:15]
        joint_pos[:, joint_indices(decoupled_wbc.ARM_JOINT_NAMES[:7])] = self.command[:, 15:22]
        joint_pos[:, joint_indices(decoupled_wbc.ARM_JOINT_NAMES[7:])] = self.command[:, 29:36]
        root_pos, root_rot = decoupled_wbc.pelvis_path(
            self.nav[:, 0], self.nav[:, 1], self.nav[:, 3], self.height, self.control_fps, heading_offset=SPAWN_YAW,
        )
        return DemoMotion(root_pos=root_pos, root_rot=root_rot.astype(np.float32), joint_pos=joint_pos,
                          hand=hand_closure(self.commanded_hands()))

    def observations(self) -> list[Observation]:
        commanded = self.commanded_motion()
        states = policy_state(self.state, self.previous_height)
        observations = []
        for t in range(len(self)):
            if t % self.request_interval == 0:
                obs = measured_observation(states[t], float(self.heading[t]), self.instruction)
            else:
                obs = commanded_observation(commanded.joint_pos[t], commanded.root_rot[t], commanded.root_pos[t, 2],
                                            commanded.hand[t], self.instruction)
            obs.step = t
            observations.append(obs)
        return observations

    def images(self, steps: np.ndarray) -> np.ndarray:
        """``[len(steps), 1, 224, 224, 3]``: the head camera at those steps, resized as at evaluation."""
        import av

        wanted = {int(s) for s in steps}
        frames: dict[int, np.ndarray] = {}
        with av.open(str(self.video)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    frames[index] = resize_image(frame.to_ndarray(format="rgb24"))
                if len(frames) == len(wanted):
                    break
        missing = wanted - set(frames)
        if missing:
            raise ValueError(f"{self.video} has no frames {sorted(missing)[:5]}")
        return np.stack([frames[int(s)] for s in steps])[:, None]
