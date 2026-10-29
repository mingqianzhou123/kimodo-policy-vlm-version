"""Does the policy use its cameras? Official evaluation with the images changed before the planner sees them.

    MUJOCO_GL=egl python scripts/diag/diag_vision.py <task> <checkpoint> <text cache> <blank|frozen|other> <episodes> <out.json>

blank:  every camera image is black.
frozen: every step shows the episode's own first frame (scene right, no visual feedback after it).
other:  every step shows the first frame of another evaluation scene (seed 620000 + (i + 37) % 100).
The env, seeds, success check and everything else are the official ones; only what the model sees changes.
"""
import json
import os
import sys

if os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].isdigit():  # render on the assigned GPU
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])

import numpy as np  # noqa: E402
from bigym.loco import make  # noqa: E402
from bigym.loco.eval import evaluate  # noqa: E402

from kimodo_vla.benchmarks.bigym2.policy import make_kimodo_policy  # noqa: E402

task, checkpoint, cache, condition, episodes, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
env = make(task)
policy = make_kimodo_policy(env, task, checkpoint, cache)
adapter = policy.adapter
original = adapter.to_observation
state = {"episode": -1, "first": None, "other": None}

if condition == "other":
    other_env = make(task)

    def other_first_frame(index):
        timestep = other_env.reset(seed=620000 + (index + 37) % 100)
        return {k: v.copy() for k, v in original(timestep).images.items()}


def to_observation(timestep):
    obs = original(timestep)
    if condition == "blank":
        obs.images = {k: np.zeros_like(v) for k, v in obs.images.items()}
    elif condition == "frozen":
        if state["first"] is None:
            state["first"] = {k: v.copy() for k, v in obs.images.items()}
        obs.images = state["first"]
    elif condition == "other":
        obs.images = state["other"]
    return obs


adapter.to_observation = to_observation
reset = policy.reset


def new_episode():
    state["episode"] += 1
    state["first"] = None
    if condition == "other":
        state["other"] = other_first_frame(state["episode"])
    reset()


policy.reset = new_episode
result = evaluate(policy, task_name=task, method=f"diag-vision-{condition}", episodes=episodes, env=env)
records = result["episode_records"]
json.dump({"task": task, "condition": condition, "checkpoint": checkpoint, "success_rate": result["success_rate"],
           "fell": sum(r["fell"] for r in records) / len(records), "records": records}, open(out, "w"))
print(f"{task} {condition}: success {result['success_rate']:.0%}", flush=True)
