"""Submit the BiGym 2.0 main-table runs to Slurm: training, then evaluation of its checkpoints.

    source <env script>                     # exports KIMODO_ENV_SCRIPT, which the jobs source
    python scripts/submit_bigym2.py --out log/bigym2/main9 --pt419h <pt_419h checkpoint dir>
    python scripts/submit_bigym2.py --out log/bigym2/main9 --tasks move_plate --variants 3cam --dry-run
    python scripts/submit_bigym2.py --out log/bigym2/main9_recipe --config configs/bigym2_recipe.yaml \
        --variants 3cam --gpus-per-train 8 --execution-frames 50 --eval-watch

Per (task, variant) the run directory is ``<out>/<task>_<variant>``: one training job
(``--gpus-per-train`` GPUs; ``torch.distributed.run`` above one), then per evaluation setting two
evaluation jobs that start once it succeeds, each scoring five checkpoints at a time on
1 GPU: the scored ones (the headline) and the rest (learning curve). With
``--eval-watch`` there is instead one evaluation job per setting that starts with the
training job and scores every checkpoint as soon as it is saved. Evaluation settings are
``--execution-frames`` (default 50 = replan every 1.67 s, and 10 = every 0.33 s, close
to π0.5). Variants: ``3cam`` (main result), ``head`` (ablation), ``head_pt419h`` (head
only, ControlNet and hand head from pt_419h). ``--config`` is the training config.

Rerunning fills in what is missing: a job of the same name still in the Slurm queue
(job names carry the output folder's name) is reused (evaluation then waits for that training job), a finished run is not trained
again, an unfinished one with a ``resume.pt`` continues with ``--resume``, and
evaluation skips checkpoints that already have a result. Slurm output goes to ``<out>/slurm/``.
"""

from __future__ import annotations

import argparse
import getpass
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from eval_run import DEFAULT_EXECUTION_FRAMES, eval_dir  # noqa: E402
from summarize_bigym2 import PI05  # noqa: E402  the paper's nine main tasks
from run_config import DEFAULT_CONFIG, checkpoint_steps, load_yaml, score_steps  # noqa: E402

VARIANTS = {
    "3cam": [],
    "head": ["--cameras", "head"],
    "head_pt419h": ["--cameras", "head", "--init-checkpoint", "{pt419h}"],
}


def queued_jobs() -> dict[str, str]:
    """Job name -> id of this user's pending and running jobs."""
    user = os.environ.get("USER") or getpass.getuser()
    out = subprocess.run(["squeue", "-u", user, "--noheader", "--format=%j %i"], capture_output=True, text=True, check=True).stdout
    return dict(line.split() for line in out.splitlines() if line.strip())


def sbatch(args, name: str, command: str, *, cpus: int, mem: str, time: str, after: str | None,
           gpus: int = 1, dependency: str = "afterok") -> str:
    options = [
        "--parsable", f"--partition={args.partition}", f"--gres=gpu:{gpus}", f"--cpus-per-task={cpus}",
        f"--mem={mem}", f"--time={time}", f"--job-name={name}",
        f"--output={args.out / 'slurm' / f'{name}-%j.log'}",
    ]
    if after:
        options.append(f"--dependency={dependency}:{after}")
    if args.exclude:
        options.append(f"--exclude={args.exclude}")
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
    parser.add_argument("--execution-frames", type=int, nargs="+", default=[DEFAULT_EXECUTION_FRAMES, 10],
                        help="one evaluation per value: 30 Hz frames run before replanning")
    parser.add_argument("--pt419h", help="pt_419h checkpoint directory (variant head_pt419h)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="training config")
    parser.add_argument("--gpus-per-train", type=int, default=1)
    parser.add_argument("--eval-watch", action="store_true", help="evaluate checkpoints while the run trains")
    parser.add_argument("--eval-parallel", type=int, default=5, help="checkpoints one evaluation job scores at once")
    parser.add_argument("--env-script", default=os.environ.get("KIMODO_ENV_SCRIPT"), help="sourced by every job; default $KIMODO_ENV_SCRIPT")
    parser.add_argument("--partition", default="train")
    parser.add_argument("--exclude", help="nodes to avoid, e.g. gpu27 (sbatch --exclude)")
    parser.add_argument("--dry-run", action="store_true", help="print the sbatch commands only")
    args = parser.parse_args()
    if not args.env_script:
        raise SystemExit("no --env-script (or $KIMODO_ENV_SCRIPT)")
    if "head_pt419h" in args.variants and not args.pt419h:
        raise SystemExit("variant head_pt419h needs --pt419h")
    args.out = args.out.expanduser().resolve()
    (args.out / "slurm").mkdir(parents=True, exist_ok=True)

    config = load_yaml(args.config.expanduser().resolve())
    steps = sorted(checkpoint_steps(config.main))
    final = score_steps(config.main)
    curve = [s for s in steps if s not in final]
    gpus = args.gpus_per_train
    # torchrun's launcher script points at the login node's Python; the module works on every node
    trainer = ["python"] if gpus == 1 else ["python", "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}"]
    config_args = [] if args.config.resolve() == DEFAULT_CONFIG else ["--config", str(args.config.resolve())]
    queued = queued_jobs()
    for task in args.tasks:
        for variant in args.variants:
            name = f"{task}_{variant}"
            run_dir = args.out / name
            tag = f"{args.out.name}-{name}"  # job names carry the output folder: run sets share task names
            train = queued.get(f"kv-train-{tag}")
            if train is None and not (run_dir / f"checkpoint_{steps[-1]}").is_dir():
                if (run_dir / "resume.pt").exists():
                    command = shlex.join([*trainer, "scripts/train.py", "--resume", str(run_dir)])
                elif run_dir.exists():
                    print(f"{name}: {run_dir} exists without resume.pt; remove it to start over")
                    continue
                else:
                    extra = [a.format(pt419h=args.pt419h) for a in VARIANTS[variant]]
                    command = shlex.join([*trainer, "scripts/train.py", "--task", task, *extra, *config_args, "--run-dir", str(run_dir)])
                train = sbatch(args, f"kv-train-{tag}", command, cpus=12 if gpus == 1 else 8 * gpus,
                               mem="96G" if gpus == 1 else f"{64 * gpus}G", time="16:00:00" if gpus == 1 else "3-00:00:00",
                               after=None, gpus=gpus)
            jobs = []
            for frames in args.execution_frames:
                suffix = "" if frames == DEFAULT_EXECUTION_FRAMES else f"-x{frames}"
                groups = (("all", final + curve),) if args.eval_watch else (("final", final), ("curve", curve))
                for label, eval_steps in groups:
                    job_name = f"kv-eval-{label}{suffix}-{tag}"
                    if job_name in queued:
                        jobs.append(f"{queued[job_name]}(queued)")
                        continue
                    if all((eval_dir(run_dir, frames) / f"checkpoint_{s}.json").exists() for s in eval_steps):
                        continue
                    command = ["python", "scripts/eval_run.py", "--run-dir", str(run_dir), "--execution-frames", str(frames),
                               "--parallel", str(args.eval_parallel), "--steps", *map(str, sorted(eval_steps))]
                    if args.eval_watch:  # starts when training starts and waits for each checkpoint
                        jobs.append(sbatch(args, job_name, shlex.join([*command, "--watch"]), cpus=5 * args.eval_parallel,
                                           mem=f"{16 * args.eval_parallel}G", time="3-00:00:00", after=train, dependency="after"))
                    else:
                        jobs.append(sbatch(args, job_name, shlex.join(command), cpus=24, mem="128G", time="1-00:00:00", after=train))
            print(f"{name}: train {train or 'done'}, eval {' '.join(jobs)}", flush=True)


if __name__ == "__main__":
    main()
