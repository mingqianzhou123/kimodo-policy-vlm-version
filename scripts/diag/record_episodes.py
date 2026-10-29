"""Record official evaluation episodes of a checkpoint as videos.

    MUJOCO_GL=egl python scripts/diag/record_episodes.py <task> <checkpoint> <text cache> <episodes> <out dir> <label>

Runs BiGym's own ``evaluate`` (seeds 620000 + i, the official reset) on the first
``episodes`` seeds and writes ``<out>/<label>_seed<S>_<success|fail>.mp4`` at 25 fps:
the outside view on top, the three 84x84 cameras the model sees (head, right wrist,
left wrist, enlarged) underneath. ``<out>/<label>.json`` keeps the episode records.
"""
import json
import os
import sys
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].isdigit():  # render on the assigned GPU
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])

import imageio  # noqa: E402
import numpy as np  # noqa: E402
from bigym.loco import make  # noqa: E402
from bigym.loco.eval import evaluate  # noqa: E402

from kimodo_vla.benchmarks.bigym2.policy import make_kimodo_policy  # noqa: E402

task, checkpoint, cache, episodes, out, label = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), Path(sys.argv[5]), sys.argv[6]
out.mkdir(parents=True, exist_ok=True)
env = make(task)
policy = make_kimodo_policy(env, task, checkpoint, cache)


def compose(outside: np.ndarray, rgb_obs: np.ndarray) -> np.ndarray:
    width = outside.shape[1]
    cams = [np.asarray(c[-3:]).transpose(1, 2, 0) for c in rgb_obs]  # (3, H, W) each
    scale = max(1, (width // len(cams)) // cams[0].shape[1])
    strip = np.concatenate([np.kron(c, np.ones((scale, scale, 1), np.uint8)) for c in cams], axis=1)
    pad = width - strip.shape[1]
    strip = np.pad(strip, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)))
    frame = np.concatenate([outside, strip], axis=0)
    return frame[: frame.shape[0] // 16 * 16, : frame.shape[1] // 16 * 16]  # sizes the encoder accepts


class Recorder:
    def __init__(self):
        self.episode, self.writer, self.steps = -1, None, 0

    def reset(self):
        self.flush()
        self.episode += 1
        self.steps = 0
        self.writer = imageio.get_writer(out / f"{label}_seed{620000 + self.episode}.mp4", fps=25, quality=7, macro_block_size=16)
        policy.reset()

    def __call__(self, timestep):
        action = policy(timestep)
        if self.steps % 2 == 0:  # 50 Hz control, 25 fps video
            self.writer.append_data(compose(np.asarray(env.render()), np.asarray(timestep.rgb_obs)))
        self.steps += 1
        return action

    def flush(self):
        if self.writer is not None:
            self.writer.close()
            print("wrote episode", self.episode, self.steps, "steps", flush=True)
            self.writer = None


recorder = Recorder()
result = evaluate(recorder, task_name=task, method=f"record-{label}", episodes=episodes, env=env)
recorder.flush()
for record in result["episode_records"]:  # name each video by its outcome
    seed = record["seed"]
    outcome = "success" if record["success"] else ("fell" if record["fell"] else "fail")
    src = out / f"{label}_seed{seed}.mp4"
    if src.exists():
        src.rename(out / f"{label}_seed{seed}_{outcome}.mp4")
json.dump({"task": task, "checkpoint": checkpoint, "records": result["episode_records"]}, open(out / f"{label}.json", "w"))
print(label, "success", result["success_rate"], flush=True)
