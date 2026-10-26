"""Success-rate tables of BiGym 2.0 runs, in the paper's format.

    python scripts/summarize_bigym2.py log/bigym2/main9 --out log/bigym2/main9/results.md

A run is a directory with ``config.json`` and ``eval/checkpoint_<step>.json``
(``scripts/eval_run.py``); its name is ``<task>_<variant>``. Each evaluation setting
gets its own column: ``eval/`` replans after the whole 1.67 s chunk, ``eval_exec<n>/``
after n frames at 30 Hz (10 = 0.33 s). The headline is the
paper's: success pooled over the checkpoints the run's config scores
(``main.score_checkpoints``, else the last five: 26k-30k as for π0.5, 80k-100k for the
baseline-recipe runs). Next to it: the final checkpoint, the best checkpoint (it
consults the evaluation seeds, so appendix only) and the fall rate. A second table
lists every evaluated checkpoint (learning curve).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_config import score_steps  # scripts/ is on sys.path when run as a script

# BiGym 2.0 paper, Table II: π0.5, one run per task, last five checkpoints pooled.
PI05 = {
    "pick_box": 19, "reach_target_multi_modal": 90, "reach_target_dual": 58,
    "drawer_top_close": 100, "drawer_top_open": 99, "move_plate": 65,
    "move_two_plates": 68, "dishwasher_close": 79, "dishwasher_load_cups": 92,
}
PI05_MEAN = 75  # as reported; the mean of the rounded entries above is 74.4
VARIANT_ORDER = ("3cam", "head", "head_pt419h")


def execution_frames(eval_dir: Path) -> int:
    return 50 if eval_dir.name == "eval" else int(eval_dir.name.removeprefix("eval_exec"))


def load_run(run_dir: Path) -> list[dict]:
    """One entry per evaluation setting of the run (the default one even before it has results)."""
    config = json.loads((run_dir / "config.json").read_text())
    task = config["main"]["task"]
    variant = run_dir.name.removeprefix(f"{task}_")
    saved = sorted(int(p.name.removeprefix("checkpoint_")) for p in run_dir.glob("checkpoint_*") if p.is_dir())
    entries = []
    for eval_dir in [run_dir / "eval", *sorted(run_dir.glob("eval_exec*"))]:
        evals = {}
        for path in eval_dir.glob("checkpoint_*.json"):
            records = json.loads(path.read_text())["episode_records"]
            evals[int(path.stem.removeprefix("checkpoint_"))] = {
                "success": sum(r["success"] for r in records),
                "fell": sum(r["fell"] for r in records),
                "episodes": len(records),
            }
        frames = execution_frames(eval_dir)
        entries.append({
            "task": task, "variant": f"{variant}, replan {frames / 30:.2f}s", "base": variant, "frames": frames,
            "saved": saved, "evals": evals, "score": score_steps(config["main"]),
        })
    return entries


def headline(run: dict) -> dict | None:
    """Pooled score over the scored checkpoints, or None until all are trained and evaluated."""
    last = run["score"]
    if any(step not in run["evals"] for step in last):
        return None
    pooled = [run["evals"][s] for s in last]
    episodes = sum(e["episodes"] for e in pooled)
    rates = {s: 100 * e["success"] / e["episodes"] for s, e in run["evals"].items()}
    return {
        "last5": 100 * sum(e["success"] for e in pooled) / episodes,
        "fell": 100 * sum(e["fell"] for e in pooled) / episodes,
        "final": rates[last[-1]],
        "peak": max(rates.values()),
    }


def status(run: dict) -> str:
    if any(step not in run["saved"] for step in run["score"]):
        trained = run["saved"][-1] if run["saved"] else 0
        return f"training ({trained // 1000}k)"
    return f"evaluating ({sum(s in run['evals'] for s in run['score'])}/{len(run['score'])})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="directory holding the run directories")
    parser.add_argument("--out", type=Path, help="also write the tables here (markdown) and the numbers next to it (.json)")
    args = parser.parse_args()

    runs = [entry for p in sorted(args.root.glob("*/config.json")) for entry in load_run(p.parent)]
    tasks = [t for t in PI05 if any(r["task"] == t for r in runs)] + sorted({r["task"] for r in runs} - set(PI05))
    order = {v: i for i, v in enumerate(VARIANT_ORDER)}
    columns = sorted({(order.get(r["base"], len(order)), r["base"], -r["frames"], r["variant"]) for r in runs})
    variants = [column[-1] for column in columns]
    by_key = {(r["task"], r["variant"]): r for r in runs}
    scores = {key: headline(run) for key, run in by_key.items()}

    lines = ["## Success rate (%), scored checkpoints pooled", "",
             "| Task | π0.5 (paper) | " + " | ".join(variants) + " |",
             "|---|---|" + "---|" * len(variants)]
    for task in tasks:
        cells = []
        for variant in variants:
            run, score = by_key.get((task, variant)), scores.get((task, variant))
            if run is None:
                cells.append("–")
            elif score is None:
                cells.append(status(run))
            else:
                cells.append(f"**{score['last5']:.0f}** (final {score['final']:.0f}, peak {score['peak']:.0f}, fell {score['fell']:.0f})")
        lines.append(f"| {task} | {PI05.get(task, '–')} | " + " | ".join(cells) + " |")
    means = []
    for variant in variants:
        values = [scores.get((t, variant)) for t in PI05]
        means.append(f"**{sum(v['last5'] for v in values) / len(values):.0f}**" if all(values) else "–")
    lines.append(f"| nine-task mean | {PI05_MEAN} | " + " | ".join(means) + " |")

    lines += ["", "## Every evaluated checkpoint (success %)", ""]
    for variant in variants:
        steps = sorted({s for (t, v), r in by_key.items() if v == variant for s in r["evals"]})
        if not steps:
            continue
        lines += [f"**{variant}**", "", "| Task | " + " | ".join(f"{s / 1000:g}k" for s in steps) + " |",
                  "|---|" + "---|" * len(steps)]
        for task in tasks:
            run = by_key.get((task, variant))
            if run is None:
                continue
            row = [f"{100 * run['evals'][s]['success'] / run['evals'][s]['episodes']:.0f}" if s in run["evals"] else "" for s in steps]
            lines.append(f"| {task} | " + " | ".join(row) + " |")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
        numbers = {f"{t}/{v}": {"headline": scores[(t, v)], "checkpoints": by_key[(t, v)]["evals"]} for t, v in by_key}
        args.out.with_suffix(".json").write_text(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
