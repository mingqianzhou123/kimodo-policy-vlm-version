"""Queue experiment four's cells on the region seeds, several episodes per GPU.

    python agentic/pilot_experiments/experiment4_latency_robustness/run_exp4_suite.py --tag 20261125-p1 \
        --gpus 4 5 6 7 --per-gpu 2 --cells anchored:194:0.1 fixed:194:0.1 blocking:25:0.03 anchored:jitter:0.1

A cell is UPDATE:DELAY:SPEED. DELAY is control steps (25, 75, 194, 383, 620) or ``jitter``
(each call's delay drawn from experiment three's Gemini latencies). Runs are named
exp4-<update>-d<steps>|jgem-v<cm/s>-<tag>-s<seed>; the robot stands still and the box is placed
in the fixed region, as in experiment one's stance runs. Episodes that end without a verdict
are moved aside and retried by experiment one's runner.
"""
import argparse
import json
from pathlib import Path
import queue
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'scripts'))
sys.path.insert(0, str(HERE.parent/'experiment1_blocking_grasp'))
from paths import AGENTIC as ROOT   # noqa: E402
from run_grasp_suite import work    # noqa: E402

DELAYS = (25, 75, 194, 383, 620)
SPEEDS = (0., .03, .1)
LATENCIES = HERE/'gemini-latencies.json'


def cell(text):
    """(update, delay steps or 'jitter', speed) from UPDATE:DELAY:SPEED; only planned cells pass."""
    update, delay, speed = text.split(':')
    speed = float(speed)
    if update not in ('blocking', 'fixed', 'anchored') or speed not in SPEEDS:
        raise SystemExit(f'Not a planned cell: {text}')
    if delay == 'jitter':
        if speed != .1:
            raise SystemExit('The jitter condition runs at 0.10 m/s only')
        return update, delay, speed
    if int(delay) not in DELAYS:
        raise SystemExit(f'Delay {delay} is not one of {DELAYS}')
    return update, int(delay), speed


def name(update, delay, speed, tag, seed):
    d = 'jgem' if delay == 'jitter' else f'd{delay:03d}'
    return f'exp4-{update}-{d}-v{int(round(speed * 100)):03d}-{tag}-s{seed}'


def flags(update, delay, speed):
    out = ['--stand-still', '--box-in-region', '--box-speed', str(speed), '--target-update', update]
    if delay == 'jitter':
        return out + ['--latency-samples', str(LATENCIES)]
    return out + ['--latch-delay-steps', str(delay)]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', required=True)
    p.add_argument('--gpus', type=int, nargs='+', required=True)
    p.add_argument('--per-gpu', type=int, default=2)
    p.add_argument('--cells', nargs='+', required=True, metavar='UPDATE:DELAY:SPEED')
    p.add_argument('--seeds', type=int, nargs='+', help='Subset of the region seeds')
    p.add_argument('--tries', type=int, default=3)
    args, args.extra = p.parse_known_args()
    config = json.loads((ROOT/'configs/grasp-loop.json').read_text())
    seeds = args.seeds or config['seeds']['region']
    if not set(seeds) <= set(config['seeds']['region']):
        raise SystemExit('Seeds outside the region group')
    shared = ['--gpu-min-free-mib', '16384'] if args.per_gpu > 1 else []
    jobs = queue.Queue()
    for text in args.cells:
        c = cell(text)
        for seed in seeds:
            jobs.put((name(*c, args.tag, seed), seed, [*flags(*c), *shared]))
    lock = threading.Lock()
    workers = [threading.Thread(target=work, args=(gpu, jobs, args, lock)) for _ in range(args.per_gpu) for gpu in args.gpus]
    for i, worker in enumerate(workers):
        if i == len(args.gpus):
            time.sleep(90)   # two simulators starting on one GPU at once can lose the device
        worker.start()
    for worker in workers:
        worker.join()
