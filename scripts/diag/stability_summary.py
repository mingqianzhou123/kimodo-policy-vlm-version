"""Compare training-stability arms: per arm and task, how many seeds diverged and how they score on validation seeds.

    python scripts/diag/stability_summary.py log/bigym2/stability/*

Each argument is one arm's ``--out`` folder of ``submit_bigym2.py --seeds``, holding ``<task>_<variant>_s<seed>``
runs. Divergence as in ``loss_spikes.py``; success is the validation block's (``val<seed-base>/``, never the
official seeds), averaged over seeds, per evaluated checkpoint.
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from loss_spikes import scan  # scripts/diag/ is on sys.path when run as a script


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arms", type=Path, nargs="+")
    parser.add_argument("--seed-base", type=int, default=700000)
    args = parser.parse_args()

    print("| arm | task | runs (last step) | diverged | median max loss | mean loss, last 10% | validation success by checkpoint |")
    print("|---|---|---|---|---|---|---|")
    for arm in args.arms:
        runs = defaultdict(list)
        for run in sorted(arm.iterdir()):
            match = re.fullmatch(r"(.+)_s(\d+)", run.name)
            if match and run.is_dir():
                runs[match[1]].append(run)
        for task, task_runs in runs.items():
            scans = [(run, scan(run)) for run in task_runs]
            scans = [(run, s) for run, s in scans if s]
            if not scans:
                continue
            diverged = [f"s{run.name.rsplit('_s', 1)[1]}@{s['diverged'] // 1000}k" for run, s in scans if s["diverged"]]
            success = defaultdict(list)
            for run, _ in scans:
                for path in (run / f"val{args.seed_base}").glob("checkpoint_*.json"):
                    success[int(path.stem.removeprefix("checkpoint_"))].append(json.loads(path.read_text())["success_rate"])
            scores = ", ".join(f"{step // 1000}k {100 * statistics.mean(v):.0f}% (n={len(v)})" for step, v in sorted(success.items()))
            tail = [s["mean_last_10pct"] for _, s in scans if s["mean_last_10pct"] is not None]
            print(f"| {arm.name} | {task} | {len(scans)} ({min(s['last_step'] for _, s in scans) // 1000}k) | "
                  f"{' '.join([str(len(diverged)), *diverged])} | {statistics.median(s['max_loss'] for _, s in scans):.3f} | "
                  f"{statistics.mean(tail):.4f} | {scores or '-'} |")


if __name__ == "__main__":
    main()
