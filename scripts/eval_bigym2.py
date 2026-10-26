"""Run BiGym 2.0's official evaluation block on one Kimodo checkpoint.

    MUJOCO_GL=egl python scripts/eval_bigym2.py --task move_plate \
        --checkpoint log/.../checkpoint_100000 --text-cache data/cache/bigym2 \
        --out outputs/move_plate.json

Fewer than 100 episodes is a smoke run: BiGym then produces no leaderboard record.
``--seed-base`` other than 620000 runs the same loop on a validation block of seeds
(``eval_seeds(episodes, seed_base=...)``) instead: for choosing settings without
looking at the official seeds; it never produces a record.
The env is the official one (three 84x84 cameras); the checkpoint's ``model.cameras`` picks which ones the planner uses.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# MuJoCo's EGL renderer ignores CUDA_VISIBLE_DEVICES and draws on the node's first GPU, so
# every evaluation on a Slurm node rendered on GPU 0: 9 instead of 97 steps/s with three
# cameras. Render on the GPU this process was given (set before mujoco is imported).
if "MUJOCO_EGL_DEVICE_ID" not in os.environ and os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].isdigit():
    os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]

import torch  # noqa: E402
from bigym.loco import make  # noqa: E402
from bigym.loco.eval import evaluate, is_success  # noqa: E402
from bigym.loco.eval.protocol import EVAL_SEED_BASE, eval_seeds  # noqa: E402

from kimodo_vla.benchmarks.bigym2.policy import make_kimodo_policy  # noqa: E402


def validate(policy, env, task: str, episodes: int, seed_base: int) -> dict:
    """``evaluate``'s episode loop on another seed block; the same records, no leaderboard entry."""
    policy.base_seed = seed_base  # diffusion noise follows the episode seeds, as in evaluation
    records = []
    for index, seed in enumerate(eval_seeds(episodes, seed_base=seed_base)):
        policy.reset()
        timestep = env.reset(seed=seed)
        length = 0
        while not timestep.last():
            timestep = env.step(policy(timestep))
            length += 1
        records.append({"episode": index, "seed": seed, "success": int(is_success(env)), "length": length,
                        "termination": env.episode_termination(), "fell": int(env.episode_fell())})
        print(f"[validate] ep {index + 1}/{episodes} seed={seed} success={records[-1]['success']}", flush=True)
    return {"task": task, "episodes": episodes, "seed_base": seed_base,
            "success_rate": sum(r["success"] for r in records) / episodes, "episode_records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--execution-frames", type=int, default=50)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--noise", choices=("per_plan", "fixed"), default="per_plan")
    parser.add_argument("--rtc-overlap-frames", type=int, default=0, help="0 = off; the original runtime used 12")
    parser.add_argument("--rtc-frozen-frames", type=int, default=1)
    parser.add_argument("--blend-steps", type=int, default=0, help="control steps to cross-fade from the previous plan")
    parser.add_argument("--weights", default="model", help="model, or a saved average such as ema0.9999")
    parser.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE, help="another value runs a validation block")
    parser.add_argument("--method", default="kimodo-vla")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    env = make(args.task)
    try:
        policy = make_kimodo_policy(
            env,
            args.task,
            args.checkpoint,
            args.text_cache,
            device=args.device,
            dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype],
            execution_frames=args.execution_frames,
            diffusion_steps=args.diffusion_steps,
            noise=args.noise,
            rtc_overlap_frames=args.rtc_overlap_frames,
            rtc_frozen_frames=args.rtc_frozen_frames,
            blend_steps=args.blend_steps,
            weights=args.weights,
        )
        if args.seed_base != EVAL_SEED_BASE:
            result = validate(policy, env, args.task, args.episodes, args.seed_base)
        else:
            result = evaluate(
                policy,
                task_name=args.task,
                method=args.method,
                checkpoint=str(args.checkpoint),
                episodes=args.episodes,
                env=env,
            )
    finally:
        env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    fell = sum(r["fell"] for r in result["episode_records"]) / result["episodes"]
    print(f"{args.task}: success {result['success_rate']:.1%}, fell {fell:.1%} -> {args.out}")


if __name__ == "__main__":
    main()
