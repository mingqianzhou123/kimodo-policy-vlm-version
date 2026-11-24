"""Run the blocking box grasp over one seed group, one episode per run folder, several GPUs at once.

    python agentic/scripts/run_grasp_suite.py --group heldout --tag 20261124-r1 --gpus 2 3 4

Run folders are named <prefix>-<tag>-s<seed>: exp1-static for the held-out seeds from the task
start (the acceptance run), exp1-stance for the same seeds with ``--start-at-stance`` (the
diagnostic without walking), dbg-exp1 for the debug seeds. A run that ends without an episode
verdict (the simulator crashed or hung) is moved to <runs>/infra-failed/ and tried again, so
that only episodes that were actually played count.
"""
import argparse
import json
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

from paths import AGENTIC as ROOT, RUNS


def verdict(run):
    try:
        return json.loads((run/'grasp-result.json').read_text()).get('termination')
    except (OSError, ValueError):
        return None


def work(gpu, seeds, args, lock):
    while True:
        try:
            seed = seeds.get_nowait()
        except queue.Empty:
            return
        run = RUNS/f'{args.prefix}-{args.tag}-s{seed}'
        for attempt in range(args.tries):
            if verdict(run):
                break
            if run.exists():
                failed = RUNS/'infra-failed'/f'{run.name}-{int(time.time())}'
                failed.parent.mkdir(exist_ok=True)
                shutil.move(run, failed)
            command = [sys.executable, 'scripts/run_oracle.py', '--run-id', run.name, '--grasp-loop', '--seed', str(seed),
                       '--sim-gpu', str(gpu), '--policy-gpu', str(gpu), '--share-policy-sim-gpu', *args.extra]
            with (RUNS/f'{run.name}.launch.log').open('a') as log:
                done = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            with lock:
                print(f'seed {seed} gpu {gpu} try {attempt+1}: exit {done.returncode}, {verdict(run) or "no verdict"}', flush=True)
            time.sleep(20)  # the launcher refuses a GPU that has not released its memory yet


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--group', choices=['debug', 'heldout'], required=True)
    p.add_argument('--tag', required=True, help='Names this evaluation, e.g. 20261124-r1')
    p.add_argument('--gpus', type=int, nargs='+', required=True)
    p.add_argument('--prefix', help='Only for debug seeds; the default is dbg-exp1')
    p.add_argument('--seeds', type=int, nargs='+', help='Subset of the group')
    p.add_argument('--tries', type=int, default=3)
    args, args.extra = p.parse_known_args()
    group = json.loads((ROOT/'configs/grasp-loop.json').read_text())['seeds'][args.group]
    chosen = args.seeds or group
    if not set(chosen) <= set(group):
        raise SystemExit(f'Seeds outside the {args.group} group')
    if args.group == 'heldout':
        if args.prefix:
            raise SystemExit('Held-out seeds run under their fixed names only')
        args.prefix = 'exp1-stance' if '--start-at-stance' in args.extra else 'exp1-static'
    elif (args.prefix or '').startswith('exp1-'):
        raise SystemExit('exp1-* names are reserved for the held-out seeds')
    args.prefix = args.prefix or 'dbg-exp1'
    seeds, lock = queue.Queue(), threading.Lock()
    for seed in chosen:
        seeds.put(seed)
    workers = [threading.Thread(target=work, args=(gpu, seeds, args, lock)) for gpu in args.gpus]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
