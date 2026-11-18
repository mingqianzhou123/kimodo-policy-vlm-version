"""Transfer the queue only at an idle episode boundary; never terminate an active rollout."""
from pathlib import Path
import json
import os
import signal
import subprocess
import time

ROOT = Path('/root/workspace/humanoidVLA')
OLD = ROOT / 'cross_tracker/runs/scratch-eval-20261118-r3'
NEW = ROOT / 'cross_tracker/runs/scratch-eval-20261118-r4'
WATCH = ROOT / 'cross_tracker/runs/scratch-watch-20261118'


def main():
    assert json.loads((NEW / 'prepare.json').read_text())['status'] == 'complete'
    assert not (NEW / 'batch.json').exists()
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
            assert len({(x['method'], x['controller'], x['group'], x['repeat']) for x in state['completed']}) == len(state['completed'])
            for item in state['completed']:
                code = Path(item['command'][2]).parents[1]
                assert code.is_relative_to(ROOT / 'cross_tracker/runs')
                report = json.loads((code / 'runs' / item['run_id'] / 'run.json').read_text())
                assert report['status'] == 'completed' and report['valid_for_success_rate']
                assert len(report['episodes']) == 1
                for name, value in report['episodes'][0].items():
                    assert item[name] == value, (item['run_id'], name)
            (NEW / 'previous-batch.json').write_text(json.dumps(state, indent=2))
            (NEW / 'carried-completed.json').write_text(json.dumps(state['completed'], indent=2))
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
                process = subprocess.Popen([str(ROOT / 'cross_tracker/data/cache/envs/policy/bin/python'),
                    '-u', str(NEW / 'run_batch.py')], stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            receipt = dict(old_pid=pid, new_pid=process.pid, carried=len(state['completed']),
                           time_unix=time.time(), boundary='No active child; parent suspended and rechecked')
            (NEW / 'handoff.json').write_text(json.dumps(receipt, indent=2))
            targets = WATCH / 'targets.tmp'
            targets.write_text(json.dumps([str(NEW)]))
            targets.replace(WATCH / 'targets.json')
            print('HANDOFF_COMPLETE', receipt, flush=True)
            return
        finally:
            if not terminated:
                os.kill(pid, signal.SIGCONT)
        time.sleep(0.5)
    raise TimeoutError('No idle episode boundary within 1 hour; old queue left running')


if __name__ == '__main__':
    main()
