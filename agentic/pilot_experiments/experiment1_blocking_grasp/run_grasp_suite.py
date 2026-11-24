"""Run the blocking box grasp over one seed group, one episode per run folder, several GPUs at once.

    python agentic/pilot_experiments/experiment1_blocking_grasp/run_grasp_suite.py --group heldout --tag 20261124-r1 --gpus 2 3 4
    python agentic/pilot_experiments/experiment1_blocking_grasp/run_grasp_suite.py --group map --tag 20261124-m1 --gpus 2 3 4 5
    python agentic/pilot_experiments/experiment1_blocking_grasp/run_grasp_suite.py --group region --tag 20261125-r1 --gpus 2 3 4 5 --speed 0.10

Run folders are named <prefix>-<tag>-s<seed>: exp1-static for the held-out seeds from the task
start (the acceptance run), exp1-stance for the same seeds with ``--start-at-stance`` (the
diagnostic without walking), dbg-exp1 for the debug seeds.

The robot never walks in the last two groups. ``map`` runs one episode per workspace map cell
(configs/grasp-loop.json, stand.map) as exp1-map-<tag>-<cell>. ``region`` runs the region seeds
with the box placed inside the fixed region: exp1-region (static, the acceptance run),
exp1-move003 / -move010 / -move030 (moving), and with the long delay exp1-long-static and
exp1-long-move010.

A run that ends without an episode verdict (the simulator crashed or hung) is moved to
<runs>/infra-failed/ and tried again, so that only episodes that were actually played count.
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
            command = [sys.executable, 'scripts/run_oracle.py', '--run-id', run.name, '--grasp-loop', '--seed', str(seed),
                       '--sim-gpu', str(gpu), '--policy-gpu', str(gpu), '--share-policy-sim-gpu', *extra, *args.extra]
            with (RUNS/f'{run.name}.launch.log').open('a') as log:
                done = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            with lock:
                print(f'seed {seed} gpu {gpu} try {attempt+1}: exit {done.returncode}, {verdict(run) or "no verdict"}', flush=True)
            time.sleep(20)  # the launcher refuses a GPU that has not released its memory yet


def region_prefix(speed, delay, config):
    """Fixed run names of the region conditions; anything else is not part of the plan."""
    stand = config['stand']
    names = {(0., 75): 'exp1-region', (0., stand['long_delay_steps']): 'exp1-long-static'}
    for v in stand['speeds_mps']:
        names[(v, 75)] = f'exp1-move{int(round(v * 100)):03d}'
    names[(.10, stand['long_delay_steps'])] = 'exp1-long-move010'
    if (speed, delay) not in names:
        raise SystemExit(f'No planned condition with speed {speed} m/s and delay {delay} steps')
    return names[(speed, delay)]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--group', choices=['debug', 'heldout', 'map', 'region'], required=True)
    p.add_argument('--tag', required=True, help='Names this evaluation, e.g. 20261124-r1')
    p.add_argument('--gpus', type=int, nargs='+', required=True)
    p.add_argument('--prefix', help='Only for debug seeds; the default is dbg-exp1')
    p.add_argument('--seeds', type=int, nargs='+', help='Subset of the group')
    p.add_argument('--tries', type=int, default=3)
    p.add_argument('--speed', type=float, default=0., help='Region group: box speed in m/s')
    p.add_argument('--latch-delay-steps', type=int, default=75, help='Region group: steps each round waits')
    p.add_argument('--per-gpu', type=int, default=1, help='Episodes run side by side on each GPU')
    args, args.extra = p.parse_known_args()
    config = json.loads((ROOT/'configs/grasp-loop.json').read_text())
    jobs, lock = queue.Queue(), threading.Lock()
    shared = ['--gpu-min-free-mib', '16384'] if args.per_gpu > 1 else []
    delay = ['--latch-delay-steps', str(args.latch_delay_steps)]
    if args.group in ('map', 'region'):
        if args.prefix or args.seeds and args.group == 'map' or '--start-at-stance' in args.extra:
            raise SystemExit('Map and region runs have fixed names and start where they are placed')
        if args.group == 'map':
            from arena.box_layout import cell_name, map_cells
            for i, cell in enumerate(map_cells(config)):
                values = [cell[k] for k in ('depth', 'lateral', 'yaw', 'extra')]
                jobs.put((f'exp1-map-{args.tag}-{cell_name(cell)}', config['stand']['map']['first_seed'] + i,
                          ['--stand-still', '--box-at', *map(str, values), *delay, *shared]))
        else:
            prefix = region_prefix(args.speed, args.latch_delay_steps, config)
            group = config['seeds']['region']
            if not set(args.seeds or group) <= set(group):
                raise SystemExit('Seeds outside the region group')
            for seed in args.seeds or group:
                jobs.put((f'{prefix}-{args.tag}-s{seed}', seed, ['--stand-still', '--box-in-region',
                          '--box-speed', str(args.speed), *delay, *shared]))
        workers = [threading.Thread(target=work, args=(gpu, jobs, args, lock))
                   for _ in range(args.per_gpu) for gpu in args.gpus]
        for i, worker in enumerate(workers):
            if i == len(args.gpus):
                time.sleep(90)   # two simulators starting on one GPU at once can lose the device
            worker.start()
        for worker in workers:
            worker.join()
        raise SystemExit(0)
    group = config['seeds'][args.group]
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
    for seed in chosen:
        jobs.put((f'{args.prefix}-{args.tag}-s{seed}', seed, delay))
    workers = [threading.Thread(target=work, args=(gpu, jobs, args, lock)) for gpu in args.gpus]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
