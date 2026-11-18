"""Refresh the existing read-only monitor on the Jupyter terminal every 15 seconds."""
from pathlib import Path
import json
import time

HERE = Path(__file__).resolve().parent
while True:
    heartbeat = json.loads((HERE / 'heartbeat.json').read_text())
    lines = [time.strftime('%F %T %Z'),
             f"Heartbeat age: {time.time() - heartbeat['checked_unix']:.1f}s",
             f"Alerts: {heartbeat['issues']}"]
    for queue in heartbeat.get('queues', []):
        lines += [f"Queue: {queue['run']}",
                  f"{queue['status']} | alive={queue['alive']} | valid={queue['completed']}/{queue['total']} | infrastructure errors={queue['errors']}",
                  f"Current: {(queue['current'] or {}).get('run_id')}",
                  f"Step: {queue.get('latest_reported_step')} | log age: {queue.get('log_age_seconds')}s | GPU wait: {queue.get('waiting_for_gpu')}"]
        batch = json.loads((Path(queue['run']) / 'batch.json').read_text())
        lines += ['', 'INTERIM ONLY: successes / validated episodes (target: 60 per cell)']
        for method in ['sonic', 'twist2', 'mix', 'cct']:
            cells = []
            for controller in ['sonic', 'twist2']:
                values = [x for x in batch['completed'] if x['method'] == method and x['controller'] == controller]
                cells.append(f"{controller}: {sum(x['success'] for x in values)}/{len(values)}")
            lines.append(f"{method:8s} {' | '.join(cells)}")
    print('\033[2J\033[H' + '\n'.join(lines), flush=True)
    time.sleep(15)
