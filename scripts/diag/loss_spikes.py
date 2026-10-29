"""Find training divergence in run directories: where the motion loss blows up, and how high it sits late.

    python scripts/diag/loss_spikes.py log/bigym2/main9_recipe/*_3cam

Normal motion loss is about 0.001; a value above 0.5 after the first 2000 steps is a spike. A run whose
loss stays far above that is diverged (on BiGym this showed up as success dropping to zero).
"""
import json
import sys
from pathlib import Path

print(f"{'run':44s} {'first >0.5':>10s} {'spikes':>6s} {'max (step)':>18s} {'mean 40-60k':>11s} {'mean last 10%':>13s}")
for run in map(Path, sys.argv[1:]):
    path = run / "metrics.jsonl"
    if not path.exists():
        continue
    rows = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        rows[record["step"]] = record  # a resumed run repeats steps; keep the last
    rows = [rows[k] for k in sorted(rows)]
    spikes = [r["step"] for r in rows if r["step"] > 2000 and r["motion_loss"] > 0.5]
    peak = max(rows, key=lambda r: r["motion_loss"])
    last = rows[-1]["step"]
    mid = [r["motion_loss"] for r in rows if 40000 <= r["step"] < 60000]
    tail = [r["motion_loss"] for r in rows if r["step"] >= 0.9 * last]
    mean = lambda v: f"{sum(v) / len(v):.4f}" if v else "-"  # noqa: E731
    print(f"{run.name:44s} {str(spikes[0]) if spikes else 'never':>10s} {len(spikes):6d} "
          f"{peak['motion_loss']:9.3f} ({peak['step']:6d}) {mean(mid):>11s} {mean(tail):>13s}")
