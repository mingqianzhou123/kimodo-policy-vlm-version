"""Can the policy redo its own training demonstrations? Closed loop from each demo's exact starting state.

    MUJOCO_GL=egl python scripts/diag/diag_trainscenes.py <task> <checkpoint> <text cache> <demo dir> <out.json> [max demos]

Each of the task's 60 demos is restored to its engage snapshot (``restore_engage_state``:
the demo's seed and simulator/controller state, as the round-trip check does), the policy
runs closed loop from the demo's first recorded observation until the episode budget,
and success is the official ``is_success``. High here but low on the evaluation seeds
means it memorised the demos; low here too means it cannot even redo them.
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

if os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].isdigit():  # render on the assigned GPU
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])

import numpy as np  # noqa: E402
from bigym.loco import make  # noqa: E402
from bigym.loco.demos.dataset import load_episodes  # noqa: E402
from bigym.loco.demos.success_hold import restore_engage_state  # noqa: E402
from bigym.loco.eval import is_success  # noqa: E402

from kimodo_vla.benchmarks.bigym2.policy import make_kimodo_policy  # noqa: E402

task, checkpoint, cache, demo_dir, out = sys.argv[1:6]
limit = int(sys.argv[6]) if len(sys.argv) > 6 else -1
env = make(task)
policy = make_kimodo_policy(env, task, checkpoint, cache)
records = []
for index, (name, episode) in enumerate(load_episodes(Path(demo_dir) / task, max_episodes=limit, decode_images=False)):
    policy.reset()
    restore_engage_state(env.bigym, episode)
    timestep = SimpleNamespace(low_dim_obs=np.asarray(episode["low_dim_obs"][0]), rgb_obs=episode["rgb_obs"].decode(0))
    steps = 0
    while True:
        timestep = env.step(policy(timestep))
        steps += 1
        if timestep.last():
            break
    records.append({"demo": Path(name).stem, "seed": int(np.asarray(episode["seed"]).reshape(-1)[0]),
                    "success": bool(is_success(env)), "fell": bool(env.episode_fell()), "steps": steps,
                    "demo_steps": len(episode["action"]) - 1})
    print(f"{index + 1}: {records[-1]}", flush=True)
rate = sum(r["success"] for r in records) / len(records)
json.dump({"task": task, "checkpoint": checkpoint, "success_rate": rate, "records": records}, open(out, "w"))
print(f"{task} training scenes: success {rate:.0%} over {len(records)} demos", flush=True)
