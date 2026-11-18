"""Read-only sidecar: persist queue liveness, log progress, and abnormal-state alerts."""
from pathlib import Path
import json
import os
import re
import shutil
import time

ROOT = Path('/root/workspace/humanoidVLA/cross_tracker')
HERE = Path(__file__).resolve().parent


def write(value):
    path = HERE / 'heartbeat.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def inspect(run):
    state = json.loads((run / 'batch.json').read_text())
    process = Path('/proc') / str(state['pid']) / 'cmdline'
    alive = process.exists() and str(run / 'run_batch.py').encode() in process.read_bytes()
    item = state.get('current')
    result = dict(run=str(run), status=state['status'], alive=alive,
                  completed=len(state['completed']), errors=len(state['failed']),
                  total=state['total_episodes'], current=item, issues=[])
    if state['failed']:
        result['issues'].append('execution_errors_require_review')
    if not alive and len(state['completed']) != state['total_episodes']:
        result['issues'].append('queue_exited_before_all_scenes_completed')
    if item and alive:
        episode = run / 'code/cross_tracker/runs' / item['run_id']
        logs = [run / (item['run_id'] + '.log'), episode / 'server.log', episode / 'simulation.log']
        logs = [p for p in logs if p.exists()]
        result['log_age_seconds'] = round(time.time() - max(
            [p.stat().st_mtime for p in logs] + [item['started_unix']]), 1)
        launcher_log = run / (item['run_id'] + '.log')
        tail = launcher_log.read_text()[-3000:] if launcher_log.exists() else ''
        result['waiting_for_gpu'] = 'WAITING_FOR_IDLE_GPUS' in tail and not episode.exists()
        simulator = episode / 'simulation.log'
        if simulator.exists():
            text = simulator.read_text()[-15000:]
            steps = re.findall(r'reward step=(\d+)', text)
            result['latest_reported_step'] = int(steps[-1]) if steps else None
        if result['log_age_seconds'] > 600 and not result['waiting_for_gpu']:
            result['issues'].append('no_log_progress_for_10_minutes')
    return result


def main():
    previous = None
    while True:
        try:
            targets = json.loads((HERE / 'targets.json').read_text())
            queues = [inspect(ROOT / 'runs' / name) for name in targets]
            free = shutil.disk_usage(ROOT).free
            issues = [dict(run=q['run'], issue=i) for q in queues for i in q['issues']]
            if free < 20 * 1024**3:
                issues.append(dict(issue='less_than_20GiB_free'))
            state = dict(checked_unix=time.time(), monitor_pid=os.getpid(),
                         queues=queues, issues=issues, free_bytes=free)
        except Exception as error:
            state = dict(checked_unix=time.time(), monitor_pid=os.getpid(),
                         issues=[dict(issue='monitor_read_error', error=repr(error))])
        write(state)
        key = json.dumps(dict(issues=state['issues'], queues=[
            (q['run'], q['completed'], q['errors'], q['status']) for q in state.get('queues', [])]))
        if key != previous:
            with (HERE / 'events.jsonl').open('a') as output:
                output.write(json.dumps(state) + '\n')
            print(key, flush=True)
            previous = key
        time.sleep(15)


if __name__ == '__main__':
    main()
