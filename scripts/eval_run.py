"""Evaluate checkpoints of one training run under the official protocol, several at once.

    MUJOCO_GL=egl python scripts/eval_run.py --run-dir log/bigym2/main9/move_plate_3cam \
        --steps 26000 27000 28000 29000 30000

Each checkpoint gets its own ``eval_bigym2.py`` process (100 episodes, seeds 620000 + i,
official env) on the same GPU; results go to ``<run-dir>/eval/checkpoint_<step>.json``
with a log next to each. ``--execution-frames`` sets how many of the 50 predicted 30 Hz
frames run before the next prediction (default all 50, 1.67 s; 10 is 0.33 s, close to
π0.5's 0.32 s); other values write to ``<run-dir>/eval_exec<frames>/``. A checkpoint that
already has a result is skipped, so a rerun only does what is missing. Task and text
cache come from the run's ``config.json``. With ``--watch`` it starts while the run is
still training: each checkpoint is evaluated once it is saved (its ``config.json`` is
written after the weights), and it gives up when no new one appears for ``--give-up-hours``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
DEFAULT_EXECUTION_FRAMES = 50  # the whole predicted chunk, as eval_bigym2.py


def eval_dir(run_dir: Path, execution_frames: int) -> Path:
    return run_dir / ("eval" if execution_frames == DEFAULT_EXECUTION_FRAMES else f"eval_exec{execution_frames}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", help="default: every checkpoint of the run")
    parser.add_argument("--parallel", type=int, default=5, help="checkpoints evaluated at once")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--execution-frames", type=int, default=DEFAULT_EXECUTION_FRAMES)
    parser.add_argument("--watch", action="store_true", help="wait for checkpoints that are not saved yet")
    parser.add_argument("--give-up-hours", type=float, default=6.0)
    args = parser.parse_args()
    if args.watch and not args.steps:
        raise SystemExit("--watch needs --steps")

    run_dir = args.run_dir.expanduser().resolve()
    ready = lambda step: (run_dir / f"checkpoint_{step}" / "config.json").exists()  # noqa: E731
    last_progress = time.time()
    while args.watch and not (run_dir / "config.json").exists():  # training has not created the run yet
        if time.time() - last_progress > args.give_up_hours * 3600:
            raise SystemExit(f"{run_dir} never appeared")
        time.sleep(60)
    config = json.loads((run_dir / "config.json").read_text())
    steps = args.steps or sorted(int(p.name.removeprefix("checkpoint_")) for p in run_dir.glob("checkpoint_*"))
    out_dir = eval_dir(run_dir, args.execution_frames)
    out_dir.mkdir(exist_ok=True)
    todo = [s for s in steps if not (out_dir / f"checkpoint_{s}.json").exists()]
    missing = [s for s in todo if not ready(s)]
    if missing and not args.watch:
        raise SystemExit(f"{run_dir.name}: no checkpoint for steps {missing}")
    print(f"{run_dir.name} ({out_dir.name}): evaluating {todo or 'nothing'} ({len(steps) - len(todo)} already done)", flush=True)

    running: dict[int, subprocess.Popen] = {}
    failed = []
    while todo or running:
        while len(running) < args.parallel and any(ready(s) for s in todo):
            step = next(s for s in todo if ready(s))
            todo.remove(step)
            last_progress = time.time()
            command = [
                sys.executable, str(SCRIPTS / "eval_bigym2.py"),
                "--task", config["main"]["task"],
                "--checkpoint", str(run_dir / f"checkpoint_{step}"),
                "--text-cache", config["main"]["text_cache"],
                "--episodes", str(args.episodes),
                "--execution-frames", str(args.execution_frames),
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
        if todo and not running and time.time() - last_progress > args.give_up_hours * 3600:
            raise SystemExit(f"{run_dir.name}: no new checkpoint for {args.give_up_hours} h; still missing {todo}")
        time.sleep(10)
    if failed:
        raise SystemExit(f"{len(failed)} checkpoint(s) failed: {failed}")


if __name__ == "__main__":
    main()
