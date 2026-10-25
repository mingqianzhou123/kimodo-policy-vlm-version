"""Does HumanoidToolBench's controller do what our commands say? Replay ToolBook demos through it.

The public toolkit cannot rebuild a recorded scene (its tools are generated and
its benches laid out differently from the internal tool the demos were recorded
with), so this checks motion, not task success. In a scene of the demo's
condition, with the robot at the recorded spawn, each mode drives the evaluator's
own agent and decoupled WBC for the demo's length:

* ``recorded``  the operator's 36-D commands, as ToolBench's ACT model reads a recording
* ``direct``    the commanded motion through ToolBenchAdapter at 50 Hz (``DirectPlanner``)
* ``kimodo``    the commanded motion through Kimodo's 30 Hz chunks (``DemoPlanner``), as evaluation
* ``always``    ``direct`` with the turn flag always on (heading feedback)

Reported per mode: falls, and how far the executed pelvis path, heading and arms
end up from the ``recorded`` run; for ``recorded`` itself, how far the executed
path ends from the commanded one (which checks the velocity frame).

Runs in HumanoidToolBench's environment, with this repository on the path:

    PYTHONPATH=<kimodo-vla> MUJOCO_GL=egl <HumanoidToolBench>/.venv/bin/python scripts/check_toolbench_roundtrip.py \
        --data-dir <ToolBook>/raw --episodes 2 --workers 16 --out outputs/toolbench_roundtrip.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

MODES = ("recorded", "direct", "kimodo", "always")
FELL_HEIGHT = 0.45  # pelvis below this: on the floor
_ENVS: dict = {}


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)  # many workers share the host; the default oversubscribes it


def _env(condition: str):
    """One environment per condition and worker, as the evaluator builds it."""
    if condition not in _ENVS:
        import gymnasium as gym
        from gymnasium.wrappers import TimeLimit

        import humanoidtoolbench.envs  # noqa: F401  (registers the conditions)
        from humanoidtoolbench.cli.eval_decoupled_wbc import _make_sonic_config
        from humanoidtoolbench.cli.public_eval import canonical_env_id

        sonic_config = _make_sonic_config()
        env = gym.make(canonical_env_id(condition), split="train", sim_mode="mujoco",
                       physics_dt=sonic_config["SIMULATE_DT"], headless=True, sonic_config=sonic_config)
        _ENVS[condition] = (TimeLimit(env, max_episode_steps=100_000), sonic_config)
    return _ENVS[condition]


def _yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _commands(mode: str, demo, motion):
    """``request -> (T, 36)`` for one episode; ``reset()`` before it starts."""
    from kimodo_vla.benchmarks.replay import DemoPlanner, DirectPlanner
    from kimodo_vla.benchmarks.toolbench.policy import make_policy

    if mode == "recorded":
        rows, interval = demo.recorded_commands(), demo.request_interval

        class Recorded:
            step = 0

            def reset(self):
                self.step = 0

            def __call__(self, request):
                chunk = rows[np.clip(np.arange(self.step + 1, self.step + 1 + interval), 0, len(rows) - 1)]
                self.step += interval
                return chunk

        return Recorded()
    planner = DemoPlanner(motion) if mode == "kimodo" else DirectPlanner(motion, chunk=demo.request_interval)
    return make_policy(planner, turn=1.0 if mode == "always" else "operator")


def run_one(job) -> dict:
    recording, episode, mode = job
    from humanoidtoolbench.policies import make_humanoid_policy_agent
    from humanoidtoolbench.policies.remote_humanoid import RemoteChunkPolicy

    from kimodo_vla.benchmarks.toolbench import demos

    start = time.time()
    recording = Path(recording)
    condition = recording.parent.name.removesuffix("-v0")
    demo = demos.ToolBenchDemo(demos.read_recording(recording, episode), demos.instruction(recording.parent.name))
    motion = demo.commanded_motion()
    spawn = json.loads(next(e for e in demos.episodes(recording) if e["episode_index"] == episode)["environment_config"])
    spawn = spawn["dr_state_dict"]["spatial"]["g1_sonic"]

    env, sonic_config = _env(condition)
    sonic_env, task = env.unwrapped, env.unwrapped.task
    original_reset = type(task).reset

    def reset_at_recorded_spawn(seed=None):
        original_reset(task, seed)
        robot = task._layout.actors["robot"]
        robot.pose.position, robot.pose.quaternion = list(spawn["position"]), list(spawn["quaternion"])

    task.reset = reset_at_recorded_spawn
    commands = _commands(mode, demo, motion)
    blank = np.zeros((360, 640, 3), np.uint8)  # replayed motion ignores the image; rendering stays off

    class Local(RemoteChunkPolicy):
        def predict(self, observation, instruction=None, **kwargs):
            request = {"image": {"rgb_head_stereo_left": blank}, "instruction": instruction,
                       "state": self._state(observation), "history": {"reset": self._reset_history}}
            self._reset_history = False
            return self._chunk(commands(request), transport="local")

    agent = make_humanoid_policy_agent(robot=task.robot, policy=Local(policy_name="replay"), controller="decoupled_wbc",
                                       host="", port=0, sonic_config=sonic_config)
    sonic_env.render_obs = False
    observation, info = env.reset(seed=episode)
    agent.reset()
    commands.reset()
    steps = 0
    while not task.robot.stabilized and steps < 1500:  # the evaluator's stabilization
        observation, _, _, _, info = sonic_env.step(agent.get_stabilize_action(observation, info=info))
        steps += 1
    agent.body_controller.finish_stabilization()

    pose, arms, fell = [], [], False
    for _ in range(len(demo) - 1):
        observation, _, _, _, info = env.step(agent.get_action(observation, info=info, instruction=demo.instruction))
        proprio = info["proprio"]
        base = np.asarray(proprio["floating_base_pose"], np.float64)
        pose.append([base[0], base[1], base[2], _yaw(base[3:7])])
        arms.append(np.asarray(proprio["body_q"], np.float64)[15:29])
        if base[2] < FELL_HEIGHT:
            fell = True
            break
    task.reset = original_reset.__get__(task)
    return {"condition": condition, "recording": str(recording), "episode": episode, "mode": mode, "fell": fell,
            "steps": len(pose), "demo_steps": len(demo) - 1, "pose": np.asarray(pose).tolist(),
            "arms": np.asarray(arms).tolist(), "commanded_xy": motion.root_pos[1:, :2].tolist(),
            "seconds": round(time.time() - start, 1)}


def summarize(results: list[dict]) -> list[dict]:
    """Per mode: compare each run with the recorded-command run of the same episode."""
    from kimodo_vla.benchmarks.toolbench.adapter import SPAWN_YAW

    rows = []
    reference = {(r["recording"], r["episode"]): r for r in results if r["mode"] == "recorded"}
    for r in results:
        ref = reference.get((r["recording"], r["episode"]))
        pose, ref_pose = np.asarray(r["pose"]), np.asarray(ref["pose"]) if ref else None
        n = min(len(pose), len(ref_pose)) if ref else 0
        row = {k: r[k] for k in ("condition", "episode", "mode", "fell", "steps", "demo_steps")}
        if n:
            row["drift_end_m"] = float(np.linalg.norm(pose[n - 1, :2] - ref_pose[n - 1, :2]))
            row["drift_max_m"] = float(np.linalg.norm(pose[:n, :2] - ref_pose[:n, :2], axis=1).max())
            row["heading_diff_max"] = float(np.abs(np.angle(np.exp(1j * (pose[:n, 3] - ref_pose[:n, 3])))).max())
            row["arm_err"] = float(np.abs(np.asarray(r["arms"])[:n] - np.asarray(ref["arms"])[:n]).mean())
        if r["mode"] == "recorded" and len(pose):
            # The commanded path, from the executed start: the episode frame's 0 is the spawn heading.
            c, s = np.cos(SPAWN_YAW), np.sin(SPAWN_YAW)
            commanded = np.asarray(r["commanded_xy"])[: len(pose)] @ np.array([[c, s], [-s, c]]) + pose[0, :2]
            row["executed_vs_commanded_end_m"] = float(np.linalg.norm(pose[-1, :2] - commanded[-1]))
            row["commanded_distance_m"] = float(np.linalg.norm(commanded[-1] - commanded[0]))
        rows.append(row)
    print(f"{'condition':18s} {'mode':9s} {'fell':>5s} {'drift end/max (m)':>18s} {'heading max':>12s} {'arm err':>8s} {'exec vs cmd (m)':>16s}")
    for mode in MODES:
        for row in (r for r in rows if r["mode"] == mode):
            print(f"{row['condition']:18s} {mode:9s} {str(row['fell']):>5s} "
                  f"{row.get('drift_end_m', float('nan')):8.3f}/{row.get('drift_max_m', float('nan')):<8.3f} "
                  f"{row.get('heading_diff_max', float('nan')):12.3f} {row.get('arm_err', float('nan')):8.4f} "
                  f"{row.get('executed_vs_commanded_end_m', float('nan')):8.3f} of {row.get('commanded_distance_m', float('nan')):.2f}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2, help="successful episodes per recording")
    parser.add_argument("--recordings", nargs="*", help="substrings selecting recordings (default: all)")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from kimodo_vla.benchmarks.toolbench import demos

    jobs = []
    for info in sorted(args.data_dir.rglob("meta/info.json")):
        recording = info.parent.parent
        if args.recordings and not any(s in str(recording) for s in args.recordings):
            continue
        info = json.loads(info.read_text())
        good = [e["episode_index"] for e in demos.episodes(recording) if e.get("episode_success")]
        good = [i for i in good if (recording / info["data_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i)).exists()]
        jobs += [(str(recording), i, mode) for i in good[: args.episodes] for mode in args.modes]
    print(f"{len(jobs)} runs", flush=True)
    results = []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker, maxtasksperchild=8) as pool:
        for result in pool.imap_unordered(run_one, jobs):
            results.append(result)
            print(f"{len(results)}/{len(jobs)} {result['condition']} ep{result['episode']} {result['mode']}: "
                  f"{result['steps']}/{result['demo_steps']} steps, fell={result['fell']} ({result['seconds']}s)", flush=True)
    rows = summarize(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"runs": results, "summary": rows}))


if __name__ == "__main__":
    main()
