"""Run BiGym 2.0's official evaluation block on one Kimodo checkpoint.

    MUJOCO_GL=egl python scripts/eval_bigym2.py --task move_plate \
        --checkpoint log/.../checkpoint_100000 --text-cache data/cache/bigym2 \
        --out outputs/move_plate.json

Fewer than 100 episodes is a smoke run: BiGym then produces no leaderboard record.
The env is the official one (three 84x84 cameras); the checkpoint's ``model.cameras`` picks which ones the planner uses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from bigym.loco import make
from bigym.loco.eval import evaluate

from kimodo_vla.benchmarks.bigym2.policy import make_kimodo_policy


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
        )
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
