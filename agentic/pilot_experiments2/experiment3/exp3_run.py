"""Run the command loop on a task's development seeds, one episode per run folder, several GPUs at once.

    python agentic/pilot_experiments2/experiment3/exp3_run.py --task pp_box --commander scripted --tag 20261126-d1 --gpus 4 5
    python agentic/pilot_experiments2/experiment3/exp3_run.py --task sit_sofa --commander qwen --tag 20261126-q1 --gpus 1 \\
        --seeds 91000 91001 --vlm-model <qwen weights folder>

Without --episodes only the development seeds of exp3_harness.json are accepted (training-
distribution layouts; debugging never touches the evaluation episodes). The formal run uses the 60
official evaluation episodes of the task (plan v3, section 2 item 2): --episodes <list> from
exp3_eval_episodes.py together with --formal, and only once the configuration is frozen. Run folders
are named exp3dev-<task>-<commander>-<tag>-s<seed>, or exp3eval-<task>-<commander>-<tag>-g<group>r<repeat>. A run that ends without Arena's verdict (the simulator
crashed or hung) is moved to <runs>/infra-failed/ and tried again, so only played episodes count.
Each run is judged afterwards with exp3_judge.py.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'scripts'))  # agentic/scripts: paths.py and the launcher
from paths import AGENTIC as ROOT, RUNS

CONFIG = HERE/'exp3_harness.json'


def played(run):
    try:
        # A simulator or policy-server crash ('sim_error', e.g. a GPU another job filled) is not a played episode.
        return json.loads((run/'episode.json').read_text()).get('failure_reason') not in (None, 'sim_error')
    except (OSError, ValueError):
        return False


def work(gpu, jobs, args, lock):
    while True:
        try:
            name, seed, episode = jobs.get_nowait()
        except queue.Empty:
            return
        run = RUNS/name
        attempt = refused = 0
        while attempt < args.tries and not played(run):
            if run.exists():
                failed = RUNS/'infra-failed'/f'{run.name}-{int(time.time())}'
                failed.parent.mkdir(exist_ok=True)
                shutil.move(run, failed)
            command = [sys.executable, 'scripts/run_oracle.py', '--run-id', name, '--command-loop', str(CONFIG),
                       '--commander', args.commander, '--task', args.task, '--seed', str(seed), '--condition', args.condition,
                       *(['--episode-seed', str(episode['episode_seed']), '--repeat-idx', str(episode['repeat_idx'])] if episode else []),
                       '--sim-gpu', str(gpu), '--policy-gpu', str(gpu), '--share-policy-sim-gpu', '--steps', str(args.steps),
                       *(['--vlm-model', str(args.vlm_model)] if args.vlm_model else []), *args.extra]
            with (RUNS/f'{name}.launch.log').open('a') as log:
                done = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            with lock:
                print(f'seed {seed} gpu {gpu} try {attempt+1}: exit {done.returncode}, played {played(run)}', flush=True)
            if done.returncode and not run.exists() and refused < args.refusals:
                refused += 1      # the launcher refused the GPU before anything started; ask again later
                time.sleep(60)
                continue
            attempt += 1
            time.sleep(20)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True, choices=['pp_box', 'sit_sofa'])
    p.add_argument('--commander', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--gpus', type=int, nargs='+', required=True)
    p.add_argument('--seeds', type=int, nargs='*', help='Default: every development seed of the task')
    p.add_argument('--episodes', type=Path, help='Evaluation-episode list (exp3_eval_episodes.py); needs --formal')
    p.add_argument('--formal', action='store_true', help='Allow the evaluation episodes: only for the formal run')
    p.add_argument('--condition', default='dev')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--vlm-model', type=Path)
    p.add_argument('--tries', type=int, default=2)
    p.add_argument('--refusals', type=int, default=30)
    p.add_argument('extra', nargs='*', help='Passed to run_oracle.py after --')
    a = p.parse_args()
    config = json.loads(CONFIG.read_text())
    if a.commander not in config['commanders']:
        raise SystemExit(f'Unknown commander {a.commander!r}')
    jobs = queue.Queue()
    if a.episodes:
        if not a.formal or a.seeds:
            raise SystemExit('Evaluation episodes need --formal and no --seeds: they are for the formal run only')
        episodes = json.loads(a.episodes.read_text())
        if any(e['task'] != a.task for e in episodes):
            raise SystemExit(f'The list holds episodes of another task than {a.task}')
        for e in episodes:
            jobs.put((f"exp3eval-{a.task}-{a.commander}-{a.tag}-g{e['group_seed']}r{e['repeat_idx']:02d}", e['group_seed'], e))
    else:
        allowed = config['tasks'][a.task]['dev_seeds']
        seeds = a.seeds or allowed
        if set(seeds)-set(allowed):
            raise SystemExit(f'Only development seeds of {a.task}: {allowed}')
        for seed in seeds:
            jobs.put((f'exp3dev-{a.task}-{a.commander}-{a.tag}-s{seed}', seed, None))
    lock = threading.Lock()
    threads = [threading.Thread(target=work, args=(gpu, jobs, a, lock)) for gpu in a.gpus]
    for t in threads:
        t.start()
        time.sleep(30)   # stagger simulator start-ups
    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
