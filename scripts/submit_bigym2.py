"""Submit the BiGym 2.0 main-table runs to Slurm: training, then evaluation of its checkpoints.

    source <env script>                     # exports KIMODO_ENV_SCRIPT, which the jobs source
    python scripts/submit_bigym2.py --out log/bigym2/main9 --pt419h <pt_419h checkpoint dir>
    python scripts/submit_bigym2.py --out log/bigym2/main9 --tasks move_plate --variants 3cam --dry-run

Per (task, variant) the run directory is ``<out>/<task>_<variant>``: one training job
(1 GPU), then two evaluation jobs that start once it succeeds, each scoring five
checkpoints at a time on 1 GPU: the last five (the headline score) and the earlier
ones (learning curve). Variants: ``3cam`` (main result), ``head`` (ablation),
``head_pt419h`` (head only, ControlNet and hand head from pt_419h).

Rerunning fills in what is missing: a finished run is not trained again, an
unfinished one with a ``resume.pt`` continues with ``--resume``, and evaluation skips
checkpoints that already have a result. Slurm output goes to ``<out>/slurm/``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from summarize_bigym2 import PI05  # noqa: E402  the paper's nine main tasks
from train import DEFAULT_CONFIG, checkpoint_steps  # noqa: E402

VARIANTS = {
    "3cam": [],
    "head": ["--cameras", "head"],
    "head_pt419h": ["--cameras", "head", "--init-checkpoint", "{pt419h}"],
}


def sbatch(args, name: str, command: str, *, cpus: int, mem: str, time: str, after: str | None) -> str:
    options = [
        "--parsable", f"--partition={args.partition}", "--gres=gpu:1", f"--cpus-per-task={cpus}",
        f"--mem={mem}", f"--time={time}", f"--job-name={name}",
        f"--output={args.out / 'slurm' / f'{name}-%j.log'}",
    ]
    if after:
        options.append(f"--dependency=afterok:{after}")
    wrapped = f"source {shlex.quote(args.env_script)} && cd {shlex.quote(str(REPO))} && {command}"
    if args.dry_run:
        print(f"sbatch {' '.join(options)} --wrap {shlex.quote(wrapped)}")
        return f"<{name}>"
    return subprocess.run(["sbatch", *options, "--wrap", wrapped], capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(PI05))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--pt419h", help="pt_419h checkpoint directory (variant head_pt419h)")
    parser.add_argument("--env-script", default=os.environ.get("KIMODO_ENV_SCRIPT"), help="sourced by every job; default $KIMODO_ENV_SCRIPT")
    parser.add_argument("--partition", default="train")
    parser.add_argument("--dry-run", action="store_true", help="print the sbatch commands only")
    args = parser.parse_args()
    if not args.env_script:
        raise SystemExit("no --env-script (or $KIMODO_ENV_SCRIPT)")
    if "head_pt419h" in args.variants and not args.pt419h:
        raise SystemExit("variant head_pt419h needs --pt419h")
    args.out = args.out.expanduser().resolve()
    (args.out / "slurm").mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(DEFAULT_CONFIG)
    steps = sorted(checkpoint_steps(config))
    final, curve = steps[-config.main.final_checkpoints:], steps[:-config.main.final_checkpoints]
    for task in args.tasks:
        for variant in args.variants:
            name = f"{task}_{variant}"
            run_dir = args.out / name
            train = None
            if not (run_dir / f"checkpoint_{config.main.max_steps}").is_dir():
                if (run_dir / "resume.pt").exists():
                    command = f"python scripts/train.py --resume {shlex.quote(str(run_dir))}"
                elif run_dir.exists():
                    print(f"{name}: {run_dir} exists without resume.pt; remove it to start over")
                    continue
                else:
                    extra = [a.format(pt419h=args.pt419h) for a in VARIANTS[variant]]
                    command = shlex.join(["python", "scripts/train.py", "--task", task, *extra, "--run-dir", str(run_dir)])
                train = sbatch(args, f"kv-train-{name}", command, cpus=12, mem="96G", time="16:00:00", after=None)
            jobs = []
            for label, eval_steps in (("final", final), ("curve", curve)):
                command = shlex.join(["python", "scripts/eval_run.py", "--run-dir", str(run_dir), "--steps", *map(str, eval_steps)])
                jobs.append(sbatch(args, f"kv-eval-{label}-{name}", command, cpus=24, mem="128G", time="1-00:00:00", after=train))
            print(f"{name}: train {train or 'done'}, eval {' '.join(jobs)}", flush=True)


if __name__ == "__main__":
    main()
