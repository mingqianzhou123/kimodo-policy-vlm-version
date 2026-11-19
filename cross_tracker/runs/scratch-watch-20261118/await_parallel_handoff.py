"""Activate prepared evaluation pairs after their finetunes finish, independently of the browser."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import time

from parallel_state import gpu_pairs

ROOT = Path('/root/workspace/humanoidVLA/cross_tracker')
WATCH = ROOT / 'runs/scratch-watch-20261118'
RUN = ROOT / 'runs/scratch-eval-20261119-r5'
EXTRA = {(0, 1): ('sonic', 'twist2'), (2, 3): ('mix',)}


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def training_complete(methods):
    for method in methods:
        run = ROOT / f'runs/box-aligned-{method}-pt419h-s42-dev-20261118-r1'
        receipt = json.loads((run / 'launch.json').read_text())
        if receipt.get('status') != 'completed' or receipt.get('stage') != 'completed':
            return False
        if not (run / 'training/checkpoint_200000/training_state.pt').is_file():
            return False
    return True


def gpu_idle(pair):
    rows = subprocess.check_output(['nvidia-smi', '-i', ','.join(map(str, pair)),
        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).splitlines()
    assert len(rows) == 2
    return all(int(row.split(',')[0]) < 100 and int(row.split(',')[1]) == 0 for row in rows)


def yield_idle_utility(pair, state):
    """Only use the inspected utility's temporary pause; never signal GPU owners."""
    proof = WATCH / 'idle-utility.json'
    if not proof.exists() or time.time() - state.get('last_idle_utility_pause', 0) < 300:
        return False
    expected = json.loads(proof.read_text())
    entry = Path('/root/bin/xiancun')
    assert hashlib.sha256(entry.read_bytes()).hexdigest() == expected['sha256']
    rows = subprocess.check_output(['nvidia-smi', '-i', ','.join(map(str, pair)),
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
    pids = {int(row.strip()) for row in rows if row.strip()}
    if not pids:
        return False
    for pid in pids:
        try:
            command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            return False
        if str(entry).encode() not in command:
            return False
    subprocess.run([str(entry), 'pause', '5'], check=True, timeout=30)
    state['last_idle_utility_pause'] = time.time()
    state.setdefault('temporary_yields', []).append(dict(pair=pair, owner_pids=sorted(pids), time_unix=time.time()))
    return True


def scheduler_alive():
    state = json.loads((RUN / 'batch.json').read_text())
    command = Path('/proc') / str(state['pid']) / 'cmdline'
    return command.exists() and str(RUN / 'run_batch.py').encode() in command.read_bytes() and state['status'] != 'failed'


def main():
    lock = (WATCH / 'parallel-activation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(pid=os.getpid(), started_unix=time.time(), status='waiting_for_finetunes')
    receipt = WATCH / 'parallel-activation.json'
    try:
        while True:
            state['checked_unix'] = time.time()
            pairs = gpu_pairs(json.loads((RUN / 'enabled-pairs.json').read_text()))
            started = (RUN / 'batch.json').exists()
            if started:
                assert scheduler_alive(), 'R5 is not alive; reconcile before any restart'
            ready = []
            state['finetunes_complete'] = {}
            for pair, methods in EXTRA.items():
                complete = training_complete(methods)
                state['finetunes_complete'][str(pair)] = complete
                if (pair in pairs and started) or not complete:
                    continue
                if not gpu_idle(pair):
                    if not yield_idle_utility(pair, state):
                        continue
                    time.sleep(3)
                if gpu_idle(pair) and pair not in pairs:
                    ready.append(pair)
            if ready:
                pairs.extend(ready)
                save(RUN / 'enabled-pairs.json', pairs)
                state.update(enabled_pairs=pairs, status='handoff_at_idle_boundary' if not started else 'parallel_running')
                save(receipt, state)
                state.setdefault('activations', []).append(dict(pairs=ready, time_unix=time.time()))
            if not started and len(pairs) > 1:
                if all(training_complete(EXTRA[pair]) and gpu_idle(pair) for pair in pairs if pair in EXTRA):
                    state['status'] = 'handoff_at_idle_boundary'
                    save(receipt, state)
                    subprocess.run(['python3', str(WATCH / 'handoff_r5.py')], check=True, timeout=3660)
                    assert scheduler_alive(), 'Handoff did not produce a live R5 scheduler'
                    started = True
            state.update(enabled_pairs=pairs, status='parallel_running' if (RUN / 'batch.json').exists() else 'waiting_for_finetunes')
            if started and all(pair in pairs for pair in EXTRA):
                state.update(status='all_three_pairs_activated', finished_unix=time.time())
                save(receipt, state)
                print('ALL_THREE_PAIRS_ACTIVATED', flush=True)
                return
            save(receipt, state)
            time.sleep(30)
    except BaseException as error:
        state.update(status='needs_attention', error=repr(error), checked_unix=time.time())
        save(receipt, state)
        raise


if __name__ == '__main__':
    main()
