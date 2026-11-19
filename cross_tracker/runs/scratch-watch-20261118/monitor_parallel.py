"""Read-only monitoring of every worker; compatible with the current serial queue."""
from pathlib import Path
import json
import os
import re
import shutil
import time

ROOT = Path('/root/workspace/humanoidVLA/cross_tracker')
HERE = Path(__file__).resolve().parent


def worker(run, item):
    episode = run / 'code/cross_tracker/runs' / item['run_id']
    logs = [run / (item['run_id'] + '.log'), episode / 'server.log', episode / 'simulation.log']
    age = time.time() - max([p.stat().st_mtime for p in logs if p.exists()] + [item['started_unix']])
    tail = logs[0].read_text()[-3000:] if logs[0].exists() else ''
    waiting = 'WAITING_FOR_IDLE_GPUS' in tail and not episode.exists()
    simulator = episode / 'simulation.log'
    steps = re.findall(r'reward step=(\d+)', simulator.read_text()[-15000:]) if simulator.exists() else []
    process = Path('/proc') / str(item.get('pid', -1)) / 'cmdline'
    alive = process.exists() and item['run_id'].encode() in process.read_bytes()
    issues = []
    if age > 600 and not waiting:
        issues.append('no_log_progress_for_10_minutes')
    # A child can exit up to five seconds before the scheduler records its receipt.
    if not alive and age > 60:
        issues.append('worker_exited_without_queue_update')
    return dict(run_id=item['run_id'], gpu_pair=item.get('gpu_pair', [6, 7]),
                alive=alive, log_age_seconds=round(age, 1), waiting_for_gpu=waiting,
                latest_reported_step=int(steps[-1]) if steps else None, issues=issues)


def inspect(run):
    state = json.loads((run / 'batch.json').read_text())
    process = Path('/proc') / str(state['pid']) / 'cmdline'
    alive = process.exists() and str(run / 'run_batch.py').encode() in process.read_bytes()
    items = state.get('active', [state['current']] if state.get('current') else [])
    workers = [worker(run, item) for item in items]
    issues = []
    if state['failed']:
        issues.append('execution_errors_require_review')
    if not alive and len(state['completed']) != state['total_episodes']:
        issues.append('queue_exited_before_all_scenes_completed')
    if alive and time.time() - state.get('heartbeat_unix', 0) > 120:
        issues.append('scheduler_heartbeat_stale')
    if state.get('attention'):
        issues.append('retry_limit_requires_review')
    if state.get('resource_errors'):
        issues.append('gpu_resource_query_failed')
    issues += [f"{entry['run_id']}: {issue}" for entry in workers for issue in entry['issues']]
    return dict(run=str(run), status=state['status'], alive=alive, completed=len(state['completed']),
        errors=len(state['failed']), total=state['total_episodes'], current=state.get('current'),
        workers=workers, issues=issues)


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
        temporary = HERE / 'heartbeat-parallel.tmp'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(HERE / 'heartbeat-parallel.json')
        key = json.dumps(dict(issues=state['issues'], queues=[
            (q['run'], q['completed'], q['errors'], q['status']) for q in state.get('queues', [])]))
        if key != previous:
            with (HERE / 'events-parallel.jsonl').open('a') as output:
                output.write(json.dumps(state) + '\n')
            print(key, flush=True)
            previous = key
        time.sleep(15)


if __name__ == '__main__':
    main()
