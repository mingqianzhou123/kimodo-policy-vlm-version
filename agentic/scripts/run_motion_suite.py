"""Run one group of experiment two's moving goals: every goal of the group, once per seed.

Groups, goals, seeds and the start position are in agentic/configs/exp2-motions.json. Each run is
named <group prefix>-<date>-<suffix>-<goal>-s<seed>; its goal file is written beside the runs,
under <runs>/<group prefix>-<date>-<suffix>/goals/. Runs are spread over the given GPUs, one at a
time per GPU. Finished runs are skipped, so an interrupted suite can simply be started again.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from paths import AGENTIC as ROOT, RUNS
from kimodo_vla.oracle.locomotion import episode_seconds, load_goal

CONFIG = ROOT/'configs/exp2-motions.json'


def reach_goals(grid):
    """Right-hand targets on a distance x height grid, each with every listed approach pitch.

    The robot walks straight ahead until the target is ``standoff_m`` in front of where it stands,
    stops, then reaches. The standing point is this rule's choice: Kimodo's hand condition is
    written relative to the root's ground point, so it cannot be given a hand target alone.
    """
    goals = []
    for distance in grid['distances_m']:
        for height in grid['heights_m']:
            for label, pitch in grid['approach_pitch_deg'][str(height)].items():
                goal = {'name': f"d{round(100*distance):03d}-h{round(100*height):03d}-{label}"}
                advance = distance-grid['standoff_m'][str(height)]
                if advance > grid['walk_threshold_m']:
                    goal['root'] = {'xy': [advance, 0.], 'speed_m_s': grid['speed_m_s']}
                goal['hand'] = {'side': 'right', 'position': [distance, grid['lateral_m'], height],
                                'approach': [float(np.cos(np.radians(pitch))), 0., float(np.sin(np.radians(pitch)))],
                                'duration_s': grid['reach_s']}
                goals.append(goal)
    return goals


def expand(group):
    """The (goal, seeds) pairs of one configured group."""
    if 'reach_grid' in group:
        return [(goal, group['seeds']) for goal in reach_goals(group['reach_grid'])]
    pairs = []
    for goal in group['goals']:
        goal = dict(goal)
        pairs.append((goal, goal.pop('seeds', group.get('seeds'))))
    return pairs


def steps_for(goal):
    return int(round(episode_seconds(load_goal(goal))*50))


def wait_idle(gpu, seconds=120):
    """The launcher refuses a GPU still releasing the previous run's memory."""
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        used, util = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.used,utilization.gpu',
                                              '--format=csv,noheader,nounits'], text=True).split(',')
        if int(used) <= 100 and int(util) == 0:
            return
        time.sleep(3)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--group', required=True)
    p.add_argument('--gpus', required=True, type=int, nargs='+')
    p.add_argument('--date', default=time.strftime('%Y%m%d'))
    p.add_argument('--suffix', default='r1')
    p.add_argument('--only', nargs='+', help='goal names to run; default all of the group')
    p.add_argument('--list', action='store_true', help='print the runs and stop')
    args, extra = p.parse_known_args()  # anything else goes to run_oracle.py unchanged
    config = json.loads(CONFIG.read_text())
    group = config['groups'][args.group]
    suite = f"{group['runs']}-{args.date}-{args.suffix}"
    jobs = []
    for goal, seeds in expand(group):
        if args.only and goal['name'] not in args.only:
            continue
        for seed in seeds:
            jobs.append((f"{suite}-{goal['name']}-s{seed}", goal, seed))
    if args.list:
        for run_id, goal, seed in jobs:
            print(run_id, steps_for(goal), json.dumps(goal))
        return
    goals = RUNS/suite/'goals'
    goals.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def worker(gpu):
        while True:
            with lock:
                if not jobs:
                    return
                run_id, goal, seed = jobs.pop(0)
            manifest = RUNS/run_id/'run.json'
            if manifest.exists() and json.loads(manifest.read_text()).get('status') == 'completed':
                print('SKIP', run_id, flush=True)
                continue
            if (RUNS/run_id).exists():
                print('EXISTS-UNFINISHED', run_id, '(move it away to rerun)', flush=True)
                continue
            path = goals/f"{goal['name']}.json"
            path.write_text(json.dumps(goal, indent=1))
            wait_idle(gpu)
            print('START', run_id, 'gpu', gpu, flush=True)
            done = subprocess.run([sys.executable, 'scripts/run_oracle.py', '--run-id', run_id, '--motion-goal', str(path),
                '--start-xy', *map(str, config['start_xy']), '--noise-seed', str(seed), '--steps', str(steps_for(goal)),
                '--sim-gpu', str(gpu), '--policy-gpu', str(gpu), '--share-policy-sim-gpu', *group.get('flags', []), *extra],
                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print('COMPLETE' if done.returncode == 0 else 'FAILED', run_id, flush=True)

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    main()
