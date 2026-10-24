"""Evaluate checkpoints of one training run under the official protocol, several at once.

    MUJOCO_GL=egl python scripts/eval_run.py --run-dir log/bigym2/main9/move_plate_3cam \
        --steps 26000 27000 28000 29000 30000

Each checkpoint gets its own ``eval_bigym2.py`` process (100 episodes, seeds 620000 + i,
official env) on the same GPU; results go to ``<run-dir>/eval/checkpoint_<step>.json``
with a log next to each. A checkpoint that already has a result is skipped, so a rerun
only does what is missing. Task and text cache come from the run's ``config.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", help="default: every checkpoint of the run")
    parser.add_argument("--parallel", type=int, default=5, help="checkpoints evaluated at once")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    config = json.loads((run_dir / "config.json").read_text())
    steps = args.steps or sorted(int(p.name.removeprefix("checkpoint_")) for p in run_dir.glob("checkpoint_*"))
    out_dir = run_dir / "eval"
    out_dir.mkdir(exist_ok=True)
    todo = [s for s in steps if not (out_dir / f"checkpoint_{s}.json").exists()]
    missing = [s for s in todo if not (run_dir / f"checkpoint_{s}").is_dir()]
    if missing:
        raise SystemExit(f"{run_dir.name}: no checkpoint for steps {missing}")
    print(f"{run_dir.name}: evaluating {todo or 'nothing'} ({len(steps) - len(todo)} already done)", flush=True)

    running: dict[int, subprocess.Popen] = {}
    failed = []
    while todo or running:
        while todo and len(running) < args.parallel:
            step = todo.pop(0)
            command = [
                sys.executable, str(SCRIPTS / "eval_bigym2.py"),
                "--task", config["main"]["task"],
                "--checkpoint", str(run_dir / f"checkpoint_{step}"),
                "--text-cache", config["main"]["text_cache"],
                "--episodes", str(args.episodes),
                "--out", str(out_dir / f"checkpoint_{step}.json"),
            ]
            with open(out_dir / f"checkpoint_{step}.log", "w") as log:
                running[step] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        for step, process in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            del running[step]
            if code == 0:
                result = json.loads((out_dir / f"checkpoint_{step}.json").read_text())
                print(f"{run_dir.name} checkpoint_{step}: success {result['success_rate']:.1%}", flush=True)
            else:
                failed.append(step)
                print(f"{run_dir.name} checkpoint_{step}: failed (exit {code}), see {out_dir / f'checkpoint_{step}.log'}", flush=True)
        time.sleep(10)
    if failed:
        raise SystemExit(f"{len(failed)} checkpoint(s) failed: {failed}")


if __name__ == "__main__":
    main()
