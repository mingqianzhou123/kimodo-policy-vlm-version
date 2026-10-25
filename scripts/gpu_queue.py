"""Run a list of jobs one per GPU, only on GPUs nobody else is using.

    python scripts/gpu_queue.py jobs.txt --log-dir outputs/queue

``jobs.txt`` has one job per line, ``<name> <command...>`` (``#`` starts a comment),
run in order. A job runs from the current directory with ``CUDA_VISIBLE_DEVICES`` set
to one GPU. Per job, ``<log-dir>`` gets ``<name>.log`` (output), ``<name>.gpu``,
``<name>.pid`` and, when it ends, ``<name>.exit`` (its exit code).

A GPU is free when no job of this queue runs on it and less than ``--max-used-mib``
of its memory is in use: someone else's process holds memory even while idle.
Jobs run in their own session, so they survive the queue; a restarted queue skips
jobs that have an exit file or are still running, and reports jobs that died
without one. Failed jobs are not retried.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path


def say(message: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}", flush=True)


def gpu_memory_used() -> dict[int, int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return {int(i): int(m) for i, m in (line.split(",") for line in out.strip().splitlines())}


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_jobs(path: Path) -> list[tuple[str, str]]:
    jobs = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, command = line.split(maxsplit=1)
            jobs.append((name, command))
    if len({name for name, _ in jobs}) != len(jobs):
        raise SystemExit("job names must be unique")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", help="candidate GPUs; default all")
    parser.add_argument("--max-used-mib", type=int, default=500)
    parser.add_argument("--poll", type=float, default=60.0, help="seconds between checks")
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    jobs = read_jobs(args.jobs)
    files = lambda name, suffix: args.log_dir / f"{name}.{suffix}"  # noqa: E731
    children: list[subprocess.Popen] = []

    while True:
        for child in children:
            child.poll()  # reap finished jobs, or they stay alive as zombies
        pending, running = [], {}  # running: name -> gpu
        for name, command in jobs:
            if files(name, "exit").exists():
                continue
            if files(name, "pid").exists():
                if alive(int(files(name, "pid").read_text())):
                    running[name] = int(files(name, "gpu").read_text())
                elif not files(name, "exit").exists():  # it may have finished since the first check
                    files(name, "exit").write_text("died without an exit code\n")
                    say(f"{name}: died without an exit code, see {files(name, 'log')}")
                continue
            pending.append((name, command))

        if not pending and not running:
            break
        used = gpu_memory_used()
        candidates = args.gpus if args.gpus is not None else sorted(used)
        free = [g for g in candidates if g not in running.values() and used.get(g, 1 << 30) < args.max_used_mib]
        for gpu, (name, command) in zip(free, pending):
            files(name, "gpu").write_text(str(gpu))
            wrapped = f"{command}; echo $? > {shlex.quote(str(files(name, 'exit')))}"
            with open(files(name, "log"), "a") as log:
                process = subprocess.Popen(
                    ["bash", "-c", wrapped],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            children.append(process)
            files(name, "pid").write_text(str(process.pid))
            say(f"{name}: started on GPU {gpu}")
            time.sleep(30)  # let it allocate before the next memory reading
        time.sleep(args.poll)

    results = {name: files(name, "exit").read_text().strip() for name, _ in jobs}
    failed = [name for name, code in results.items() if code != "0"]
    say(f"all {len(jobs)} jobs ended; {len(failed)} failed{': ' + ', '.join(failed) if failed else ''}")


if __name__ == "__main__":
    main()
