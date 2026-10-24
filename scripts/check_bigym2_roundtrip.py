"""How much success survives our action interface if the model predicted perfectly?

For each demonstration, a ``DemoPlanner`` hands the Agent the demo's own motion
in place of a model prediction, through the exact evaluation path (30 Hz Kimodo
chunks, resampling, 40-D encoding, BiGym2Adapter). Modes:

* ``replay``   the recorded actions, bit for bit (sanity: should succeed every time)
* ``executed`` the motion the robot actually performed
* ``commanded`` the operator's arm / torso / base commands as a pose trajectory
* ``direct``   ``commanded`` handed to the adapter at 50 Hz, skipping Kimodo's 30 Hz
  representation: separates the model-rate loss from the adapter's

    MUJOCO_GL=egl python scripts/check_bigym2_roundtrip.py --data-dir <bigym-g1-native60> \
        --tasks move_plate pick_box --workers 32 --out outputs/roundtrip.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kimodo_vla.benchmarks.base import AgentPolicy
from kimodo_vla.benchmarks.bigym2.adapter import ARM_JOINT_NAMES, BiGym2Adapter
from kimodo_vla.benchmarks.bigym2.demos import DemoMotion, commanded_motion, executed_motion
from kimodo_vla.core.agent import Agent
from kimodo_vla.core.types import ActionChunk, G1_JOINT_NAMES, joint_indices
from kimodo_vla.model.planner import G1_XML_PATH
from kimodo_vla.motion.g1_reference import (
    HumanoidArenaActionDecoder,
    resample_hand_binary,
    resample_hand_binary_chunk,
    resample_motion,
    resample_motion_chunk,
)
from kimodo_vla.skeleton.definitions import G1Skeleton34

MODES = ("replay", "executed", "commanded", "direct")


class DemoPlanner:
    """A Planner that 'predicts' a recorded motion, chunked exactly like KimodoPlanner."""

    history_steps = 2

    def __init__(self, motion: DemoMotion, control_fps=50.0, model_fps=30.0, chunk=50):
        self.control_fps, self.model_fps, self.chunk = control_fps, model_fps, chunk
        self.codec = HumanoidArenaActionDecoder(G1Skeleton34(), G1_XML_PATH, fps=control_fps)
        pose = self.codec.decode_joint_configuration_pose(
            motion.joint_pos, motion.root_pos, root_rotation_matrices=motion.root_rot, joint_names=G1_JOINT_NAMES
        )
        self.rot, self.root = resample_motion(pose["local_rot_mats"], pose["root_positions"], control_fps, model_fps)
        self.hand = resample_hand_binary(motion.hand, control_fps, model_fps)
        self.reset()

    def reset(self, seed=None):
        self._previous_rot = None
        self._previous_hand = None

    def plan(self, history):
        last = len(self.rot) - 1  # past the end of the demo, hold its final pose
        k = int(round(history[-1].step * self.model_fps / self.control_fps))
        before = min(max(k - 1, 0), last)
        idx = np.clip(np.arange(k, k + self.chunk), 0, last)
        rot, root, hand = self.rot[idx], self.root[idx], self.hand[idx]
        previous_rot = self.rot[before] if self._previous_rot is None else self._previous_rot
        previous_hand = self.hand[before] if self._previous_hand is None else self._previous_hand
        control_rot, control_root = resample_motion_chunk(
            rot, root, self.model_fps, self.control_fps,
            previous_local_rot_mat=previous_rot, previous_root_position=self.root[before],
        )
        control_hand = resample_hand_binary_chunk(hand, self.model_fps, self.control_fps, previous_hand_binary=previous_hand)
        actions = self.codec.encode(control_rot, control_root, previous_root_position=self.root[before], hand_binary=control_hand)
        self._previous_rot, self._previous_hand = rot[-1], hand[-1]
        return ActionChunk.from_array40(actions.numpy(), fps=self.control_fps)


class DirectPlanner:
    """The recorded motion at the control rate; frame 0 is the next step's row."""

    history_steps = 2

    def __init__(self, motion: DemoMotion, control_fps=50.0, chunk=83):
        self.motion, self.control_fps, self.chunk = motion, control_fps, chunk

    def reset(self, seed=None):
        pass

    def plan(self, history):
        m, last = self.motion, len(self.motion.joint_pos) - 1
        idx = np.clip(np.arange(history[-1].step + 1, history[-1].step + 1 + self.chunk), 0, last)
        delta = m.root_pos[idx] - m.root_pos[np.clip(idx - 1, 0, last)]
        local = np.einsum("tji,tj->ti", m.root_rot[idx], delta)[:, :2]  # into each frame's pelvis frame
        return ActionChunk(local, m.root_pos[idx, 2], m.root_rot[idx], m.joint_pos[idx], m.hand[idx], self.control_fps)


_CACHE: dict = {}
_YAW_GAIN: float | None = None


def _init_worker(yaw_gain: float) -> None:
    import torch

    global _YAW_GAIN
    torch.set_num_threads(1)  # many workers share the host; the default oversubscribes it
    _YAW_GAIN = yaw_gain


def _task_state(data_dir: Path, task: str):
    if task not in _CACHE:
        from bigym.loco import make
        from bigym.loco.demos.dataset import load_episodes, load_task_metadata

        env = make(task, camera_keys=())
        adapter = BiGym2Adapter.from_env(env, task)
        adapter.yaw_gain = _YAW_GAIN
        metadata = load_task_metadata(data_dir / task)
        episodes = load_episodes(data_dir / task, decode_images=False)
        _CACHE[task] = (env, adapter, metadata["action_stats"], episodes, env.action_stats)
    return _CACHE[task]


def run_one(job):
    data_dir, task, index, mode = job
    from bigym.loco.demos.success_hold import restore_engage_state
    from bigym.loco.eval import is_success

    start = time.time()
    env, adapter, stats, episodes, official_stats = _task_state(Path(data_dir), task)
    _, episode = episodes[index]
    executed = executed_motion(env, episode, stats)
    restore_engage_state(env.bigym, episode)
    arms = joint_indices(ARM_JOINT_NAMES)
    arm_err, root_err, steps = [], [], 0

    if mode == "replay":
        env.set_action_stats(np.asarray(stats["min"], np.float32), np.asarray(stats["max"], np.float32))
        for action in episode["action"][1:]:
            timestep = env.step(action)
            steps += 1
            if timestep.last():
                break
        env.set_action_stats(official_stats["min"], official_stats["max"])
    else:
        motion = executed if mode == "executed" else commanded_motion(executed, episode, stats, adapter)
        planner_type = DirectPlanner if mode == "direct" else DemoPlanner
        policy = AgentPolicy(Agent(planner_type(motion, control_fps=adapter.control_fps)), adapter)
        policy.reset()
        timestep = SimpleNamespace(low_dim_obs=episode["low_dim_obs"][0], rgb_obs=np.zeros((0, 3, 84, 84), np.uint8))
        while True:
            timestep = env.step(policy(timestep))
            steps += 1
            if steps < len(executed.joint_pos):
                obs = adapter.to_observation(timestep)
                arm_err.append(np.abs(obs.joint_pos[arms] - executed.joint_pos[steps, arms]).mean())
                root_err.append(np.linalg.norm(obs.root_pos[:2] - executed.root_pos[steps, :2]))
            if timestep.last():
                break
    return {
        "task": task, "episode": index, "mode": mode, "success": bool(is_success(env)),
        "fell": bool(env.episode_fell()), "steps": steps, "demo_steps": len(executed.joint_pos) - 1,
        "arm_err": float(np.mean(arm_err)) if arm_err else None,
        "root_err_end": float(root_err[-1]) if root_err else None,
        "seconds": round(time.time() - start, 1),
    }


def summarize(results):
    print(f"{'task':22s} {'mode':10s} {'success':>8s} {'fell':>6s} {'arm err (rad)':>14s} {'pelvis drift (m)':>17s}")
    for task in dict.fromkeys(r["task"] for r in results):
        for mode in MODES:
            rows = [r for r in results if r["task"] == task and r["mode"] == mode]
            if not rows:
                continue
            arm = [r["arm_err"] for r in rows if r["arm_err"] is not None]
            drift = [r["root_err_end"] for r in rows if r["root_err_end"] is not None]
            print(
                f"{task:22s} {mode:10s} {np.mean([r['success'] for r in rows]):8.0%} "
                f"{np.mean([r['fell'] for r in rows]):6.0%} "
                f"{(np.mean(arm) if arm else float('nan')):14.3f} {(np.mean(drift) if drift else float('nan')):17.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--yaw-gain", type=float, default=BiGym2Adapter.yaw_gain)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    jobs = [(str(args.data_dir), task, i, mode) for task in args.tasks for i in range(args.episodes) for mode in args.modes]
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=(args.yaw_gain,)) as pool:
        results = []
        # Jobs are task-major; contiguous chunks keep each worker on few tasks, so fewer
        # workers read a task's whole dataset (images included) from disk.
        chunksize = max(1, len(jobs) // (args.workers * 4))
        for i, result in enumerate(pool.imap_unordered(run_one, jobs, chunksize=chunksize)):
            results.append(result)
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(jobs)} episodes done", flush=True)
    results.sort(key=lambda r: (r["task"], r["mode"], r["episode"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    summarize(results)
    print(f"median seconds per episode: {np.median([r['seconds'] for r in results]):.0f}")


if __name__ == "__main__":
    main()
