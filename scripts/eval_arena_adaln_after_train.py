"""Run matched Arena evaluation after the long AdaLN training job exits."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path


TASKS = ("football", "doubledesk", "pp_box")
BASELINE = Path(
    "/root/workspace/kimodo-policy/log/HumanoidArena_Multi_Task/"
    "humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse/"
    "2026-10-18_21-07-12/checkpoint_500000"
)


def evaluate(repo: Path, checkpoint: Path, label: str, run_dir: Path, gpu: int) -> dict:
    results = {}
    for task in TASKS:
        out = run_dir / f"{label}_{task}_formal"
        subprocess.run([
            str(repo / ".venv-groot/bin/python"), str(repo / "scripts/eval_arena_doubledesk.py"),
            "--checkpoint", str(checkpoint), "--task", task, "--gpu", str(gpu),
            "--out", str(out), "--reference-manifest",
            str(repo / f"outputs/sonicx7_preflight/sim_{task}/evaluation_manifest.json"),
        ], cwd=repo, check=True)
        measured = json.loads((out / "summary.json").read_text())
        model_result, = measured["per_model"].values()
        rates = list(model_result["per_seed_success_rate"].values())
        n, k = measured["total_episodes"], measured["total_successes"]
        rate, z = k / n, 1.96
        center = (rate + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        results[task] = dict(successes=k, episodes=n, success_rate=rate,
                             per_seed=model_result["per_seed_success_rate"],
                             seed_mean=statistics.mean(rates), seed_sample_sd=statistics.stdev(rates),
                             wilson_95=[center - radius, center + radius])
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-pid", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=7)
    args = parser.parse_args()
    while True:
        try:
            import os
            os.kill(args.train_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(60)
    checkpoint = args.run_dir / "checkpoint_500000"
    if not (checkpoint / "training_state.pt").is_file():
        raise SystemExit(f"training exited without {checkpoint}/training_state.pt")
    groot = evaluate(args.repo, checkpoint, "adaln", args.run_dir, args.gpu)
    original = evaluate(args.repo, BASELINE, "original", args.run_dir, args.gpu)
    summary = dict(results={"adaln": groot, "original": original}, evaluation_complete=True,
                   delta_vs_matched_original={t: groot[t]["success_rate"] - original[t]["success_rate"] for t in TASKS},
                   training_seeds=[42], evaluation_seeds=[0, 1, 2],
                   caveat="Evaluation-seed SD is not training-seed SD; optimization differs from the original baseline.")
    (args.run_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
