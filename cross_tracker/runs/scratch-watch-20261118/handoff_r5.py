"""Audit and transfer R4 at an idle boundary after additional GPUs become available."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import time

from parallel_state import gpu_pairs
from summarize_final import summarize

ROOT = Path('/root/workspace/humanoidVLA/cross_tracker')
OLD = ROOT / 'runs/scratch-eval-20261118-r4'
NEW = ROOT / 'runs/scratch-eval-20261119-r5'
WATCH = ROOT / 'runs/scratch-watch-20261118'


def prerequisites():
    assert json.loads((NEW / 'prepare.json').read_text())['status'] == 'complete'
    assert not (NEW / 'batch.json').exists()
    assert not os.environ.get('ISAAC_MULTI_IMAGE_SHM_NAME')
    pairs = gpu_pairs(json.loads((NEW / 'enabled-pairs.json').read_text()))
    assert (6, 7) in pairs and len(pairs) >= 2
    for pair in pairs:
        if pair == (6, 7):
            continue
        methods = ('sonic', 'twist2') if pair == (0, 1) else ('mix',)
        for method in methods:
            run = ROOT / f'runs/box-aligned-{method}-pt419h-s42-dev-20261118-r1'
            receipt = json.loads((run / 'launch.json').read_text())
            assert receipt['status'] == receipt['stage'] == 'completed', method
            assert (run / 'training/checkpoint_200000/training_state.pt').is_file()
        rows = subprocess.check_output(['nvidia-smi', '-i', ','.join(map(str, pair)),
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).splitlines()
        assert len(rows) == 2
        assert all(int(row.split(',')[0]) < 100 and int(row.split(',')[1]) == 0 for row in rows), pair
    for metadata, root in [('source-files.json', NEW / 'code'),
                           ('shared-eval-assets.json', ROOT / 'data/cache'), ('runner-sources.json', NEW)]:
        for name, digest in json.loads((NEW / metadata).read_text()).items():
            with (root / name).open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, name
    return pairs


def main():
    pairs = prerequisites()
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        state = json.loads((OLD / 'batch.json').read_text())
        pid = state['pid']
        proc = Path(f'/proc/{pid}')
        assert str(OLD / 'run_batch.py').encode() in (proc / 'cmdline').read_bytes()
        children = proc / f'task/{pid}/children'
        if state['current'] is not None or children.read_text().strip():
            time.sleep(0.5)
            continue
        os.kill(pid, signal.SIGSTOP)
        terminated = False
        try:
            state = json.loads((OLD / 'batch.json').read_text())
            if state['current'] is not None or children.read_text().strip():
                continue
            reports = {}
            for item in state['completed']:
                code = Path(item['command'][2]).parents[1]
                assert code.is_relative_to(ROOT / 'runs')
                reports[item['run_id']] = json.loads((code / 'runs' / item['run_id'] / 'run.json').read_text())
            manifests = {method: json.loads(path.read_text()) for method in ('sonic', 'twist2', 'mix', 'cct')
                         if (path := NEW / f'{method}-manifest.json').exists()}
            audit = summarize(state, reports, manifests, require_complete=False)
            with (NEW / 'previous-batch.json').open('x') as output:
                json.dump(state, output, indent=2)
            os.kill(pid, signal.SIGTERM)
            os.kill(pid, signal.SIGCONT)
            terminated = True
            for _ in range(50):
                if not proc.exists() or ') Z ' in (proc / 'stat').read_text():
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError('Old scheduler did not exit; new queue not started')
            with (NEW / 'batch.log').open('x') as log:
                process = subprocess.Popen([str(ROOT / 'data/cache/envs/policy/bin/python'),
                    '-u', str(NEW / 'run_batch.py')], stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(100):
                assert process.poll() is None, 'New scheduler exited; inspect batch.log'
                if (NEW / 'batch.json').exists():
                    new_state = json.loads((NEW / 'batch.json').read_text())
                    if new_state['pid'] == process.pid and new_state['status'] != 'failed':
                        break
                time.sleep(0.2)
            else:
                raise RuntimeError('New scheduler heartbeat not observed')
            receipt = dict(old_pid=pid, new_pid=process.pid, carried=audit['valid_episodes'],
                failed_attempts=len(state['failed']), gpu_pairs=pairs, time_unix=time.time(),
                boundary='No active child; old parent suspended, receipts audited, old exit verified')
            with (NEW / 'handoff.json').open('x') as output:
                json.dump(receipt, output, indent=2)
            temporary = WATCH / 'targets.tmp'
            temporary.write_text(json.dumps([str(NEW)]))
            temporary.replace(WATCH / 'targets.json')
            print('HANDOFF_COMPLETE', receipt, flush=True)
            return
        finally:
            if not terminated:
                os.kill(pid, signal.SIGCONT)
    raise TimeoutError('No idle boundary within 1 hour; old queue left running')


if __name__ == '__main__':
    main()
