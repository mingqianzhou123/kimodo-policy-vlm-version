"""Run the blocking hammer grasp (DoubleDesk, right hand, robot standing still) over one group.

    python agentic/pilot_experiments/experiment1b_hammer_grasp/run_hammer_suite.py --group map --tag 20261125-m1 --gpus 5 6 --per-gpu 2
    python agentic/pilot_experiments/experiment1b_hammer_grasp/run_hammer_suite.py --group region --tag 20261126-r1 --gpus 5 6 --per-gpu 2
    python agentic/pilot_experiments/experiment1b_hammer_grasp/run_hammer_suite.py --group inject --tag 20261126-i1 --gpus 5 6 --per-gpu 2
    python agentic/pilot_experiments/experiment1b_hammer_grasp/run_hammer_suite.py --group motion --tag 20261126-v1 --gpus 5 6 --per-gpu 2

``map`` runs one episode per reach-map cell (configs/hammer-grasp.json, stand.map) as
exp1b-map-<tag>-<cell>. ``region`` runs the acceptance seeds with the hammer placed inside the
fixed region as exp1b-region-<tag>-s<seed>. ``inject`` runs the acceptance seeds once per
injected displacement in stand.inject (exp1b-inj-<kind><value>-<tag>-s<seed>); ``motion`` runs
them once per speed and delay in stand.motion (exp1b-move<cm/s>-d<delay>-<tag>-s<seed>).

A run that ends without an episode verdict (the simulator crashed or hung at start-up) is moved
to <runs>/infra-failed/ and tried again, so that only episodes that were actually played count.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))  # agentic/scripts: paths.py and the launcher
from paths import AGENTIC as ROOT, RUNS

START_TIMEOUT_S = 420


def verdict(run):
    try:
        return json.loads((run/'grasp-result.json').read_text()).get('termination')
    except (OSError, ValueError):
        return None


def work(gpu, jobs, args, lock):
    while True:
        try:
            name, seed, extra = jobs.get_nowait()
        except queue.Empty:
            return
        run = RUNS/name
        for attempt in range(args.tries):
            if verdict(run):
                break
            if run.exists():
                failed = RUNS/'infra-failed'/f'{run.name}-{int(time.time())}'
                failed.parent.mkdir(exist_ok=True)
                shutil.move(run, failed)
            command = [sys.executable, 'scripts/run_oracle.py', '--run-id', run.name, '--task', 'doubledesk', '--grasp-loop',
                       '--stand-still', '--seed', str(seed), '--sim-gpu', str(gpu), '--policy-gpu', str(gpu),
                       '--share-policy-sim-gpu', '--start-timeout-s', str(START_TIMEOUT_S), *extra, *args.extra]
            with (RUNS/f'{run.name}.launch.log').open('a') as log:
                done = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            with lock:
                print(f'{name} gpu {gpu} try {attempt+1}: exit {done.returncode}, {verdict(run) or "no verdict"}', flush=True)
            time.sleep(20)  # the launcher refuses a GPU that has not released its memory yet


def jobs_for(args, config):
    """(run name, seed, launcher arguments) of every episode in the group, in a fixed order."""
    stand, out = config['stand'], []
    shared = ['--gpu-min-free-mib', '16384'] if args.per_gpu > 1 else []
    if args.group == 'map':
        from arena.hammer_layout import cell_name, map_cells, placeable
        for i, cell in enumerate(map_cells(config)):
            if not placeable(cell, config):
                continue   # part of the hammer would hang over the table edge: not a placement
            values = [cell[k] for k in ('depth', 'lateral', 'yaw', 'extra')]
            out.append((f'exp1b-map-{args.tag}-{cell_name(cell)}', stand['map']['first_seed'] + i,
                        ['--hammer-at', *map(str, values), *shared]))
        return out
    seeds = args.seeds or config['seeds']['region']
    if not set(seeds) <= set(config['seeds']['region']):
        raise SystemExit('Seeds outside the region group')
    if args.group == 'region':
        return [(f'exp1b-region-{args.tag}-s{s}', s, ['--hammer-in-region', *shared]) for s in seeds]
    if args.group == 'inject':
        for kind, values in stand['inject'].items():
            if kind.startswith('_'):
                continue
            for v in values:
                vector = {'along': [v, 0, 0], 'across': [0, v, 0], 'turn': [0, 0, v]}[kind]
                label = f'{kind}{int(round(v * (1 if kind == "turn" else 100))):+03d}'
                out += [(f'exp1b-inj-{label}-{args.tag}-s{s}', s,
                         ['--hammer-in-region', '--inject', *map(str, vector), *shared]) for s in seeds]
        return out
    for speed in stand['motion']['speeds_mps']:
        for delay in stand['motion']['delay_steps']:
            out += [(f'exp1b-move{int(round(speed * 100)):03d}-d{delay}-{args.tag}-s{s}', s,
                     ['--hammer-in-region', '--hammer-speed', str(speed), '--latch-delay-steps', str(delay), *shared])
                    for s in seeds]
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--group', choices=['map', 'region', 'inject', 'motion'], required=True)
    p.add_argument('--tag', required=True, help='Names this evaluation, e.g. 20261125-m1')
    p.add_argument('--gpus', type=int, nargs='+', required=True)
    p.add_argument('--seeds', type=int, nargs='+', help='Subset of the region seeds')
    p.add_argument('--only', nargs='+', help='Run only the jobs whose names contain one of these strings')
    p.add_argument('--tries', type=int, default=3)
    p.add_argument('--per-gpu', type=int, default=1, help='Episodes run side by side on each GPU')
    p.add_argument('--dry-run', action='store_true', help='List the runs and stop')
    args, args.extra = p.parse_known_args()
    config = json.loads((ROOT/'configs/hammer-grasp.json').read_text())
    planned = [j for j in jobs_for(args, config) if not args.only or any(s in j[0] for s in args.only)]
    if args.dry_run:
        print('\n'.join(f'{n} seed {s}: {" ".join(e)}' for n, s, e in planned))
        raise SystemExit(0)
    jobs, lock = queue.Queue(), threading.Lock()
    for job in planned:
        jobs.put(job)
    workers = [threading.Thread(target=work, args=(gpu, jobs, args, lock)) for _ in range(args.per_gpu) for gpu in args.gpus]
    for i, worker in enumerate(workers):
        if i == len(args.gpus):
            time.sleep(90)   # two simulators starting on one GPU at once can lose the device
        worker.start()
    for worker in workers:
        worker.join()
