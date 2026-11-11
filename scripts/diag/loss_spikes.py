"""Find training divergence in run directories: where the motion loss blows up, and how high it sits late.

    python scripts/diag/loss_spikes.py log/bigym2/main9_recipe/*_3cam

Normal motion loss is about 0.001; a logged value above 0.5 after the first 2000 steps is a spike. A run is
diverged from the first point where the median of 10 logged values in a row (500 steps) is above 0.5: the
loss stays up instead of recovering (on BiGym this showed up as success dropping to zero). A median, so a
single update that SpikeGuard dropped (logged with its high loss) does not count.
"""
import json
import statistics
import sys
from pathlib import Path

SPIKE, AFTER, RUN = 0.5, 2000, 10


def scan(run: Path) -> dict | None:
    """Divergence numbers of one run directory, from its ``metrics.jsonl``; None if it has none yet."""
    path = run / "metrics.jsonl"
    if not path.exists():
        return None
    rows = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        rows[record["step"]] = record  # a resumed run repeats steps; keep the last
    rows = [rows[k] for k in sorted(rows)]
    if not rows:
        return None
    late = [r for r in rows if r["step"] > AFTER]
    losses = [r["motion_loss"] for r in late]
    start = next((i for i in range(len(late) - RUN + 1) if statistics.median(losses[i:i + RUN]) > SPIKE), None)
    diverged = None if start is None else next(r["step"] for r in late[start:] if r["motion_loss"] > SPIKE)
    peak = max(rows, key=lambda r: r["motion_loss"])
    last = rows[-1]["step"]
    mean = lambda v: sum(v) / len(v) if v else None  # noqa: E731
    return {
        "last_step": last,
        "first_spike": next((r["step"] for r in late if r["motion_loss"] > SPIKE), None),
        "spikes": sum(loss > SPIKE for loss in losses),
        "diverged": diverged,
        "max_loss": peak["motion_loss"],
        "max_loss_step": peak["step"],
        "max_grad": max(r.get("grad_control", 0.0) for r in late) if late else None,
        "mean_40_60k": mean([r["motion_loss"] for r in rows if 40000 <= r["step"] < 60000]),
        "mean_last_10pct": mean([r["motion_loss"] for r in rows if r["step"] >= 0.9 * last]),
        "skipped_steps": rows[-1].get("skipped_steps"),
    }


def main() -> None:
    show = lambda v, f: "-" if v is None else format(v, f)  # noqa: E731
    print(f"{'run':44s} {'last':>7s} {'first >0.5':>10s} {'spikes':>6s} {'diverged':>8s} {'max (step)':>18s} "
          f"{'max grad':>8s} {'mean 40-60k':>11s} {'mean last 10%':>13s} {'skipped':>7s}")
    for run in map(Path, sys.argv[1:]):
        s = scan(run)
        if s is None:
            continue
        print(f"{run.name:44s} {s['last_step']:7d} {show(s['first_spike'], 'd'):>10s} {s['spikes']:6d} "
              f"{show(s['diverged'], 'd'):>8s} {s['max_loss']:9.3f} ({s['max_loss_step']:6d}) {show(s['max_grad'], '.2f'):>8s} "
              f"{show(s['mean_40_60k'], '.4f'):>11s} {show(s['mean_last_10pct'], '.4f'):>13s} {show(s['skipped_steps'], 'd'):>7s}")


if __name__ == "__main__":
    main()
