"""Show every active GPU pair and all eight partial evaluation cells."""
from pathlib import Path
import json
import time

HERE = Path(__file__).resolve().parent


def main():
    while True:
        try:
            heartbeat = json.loads((HERE / 'heartbeat-parallel.json').read_text())
            lines = [time.strftime('%F %T %Z'),
                f"Heartbeat age: {time.time() - heartbeat['checked_unix']:.1f}s",
                f"Alerts: {heartbeat['issues']}"]
            for queue in heartbeat.get('queues', []):
                lines += [f"Queue: {queue['run']}",
                    f"{queue['status']} | alive={queue['alive']} | valid={queue['completed']}/{queue['total']} | infrastructure errors={queue['errors']}"]
                for worker in queue['workers']:
                    lines += [f"GPUs {worker['gpu_pair']}: {worker['run_id']}",
                        f"  step={worker['latest_reported_step']} | log age={worker['log_age_seconds']}s | GPU wait={worker['waiting_for_gpu']}"]
                batch = json.loads((Path(queue['run']) / 'batch.json').read_text())
                lines += ['', 'INTERIM ONLY: successes / validated episodes (target: 60 per cell)']
                for method in ['sonic', 'twist2', 'mix', 'cct']:
                    cells = []
                    for controller in ['sonic', 'twist2']:
                        values = [x for x in batch['completed'] if x['method'] == method and x['controller'] == controller]
                        cells.append(f"{controller}: {sum(x['success'] for x in values)}/{len(values)}")
                    lines.append(f"{method:8s} {' | '.join(cells)}")
        except Exception as error:
            lines = [time.strftime('%F %T %Z'), f'MONITOR READ ERROR: {error!r}']
        print('\033[2J\033[H' + '\n'.join(lines), flush=True)
        time.sleep(15)


if __name__ == '__main__':
    main()
