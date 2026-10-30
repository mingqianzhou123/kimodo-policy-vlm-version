"""Read-only evaluation supervision; all intermediate rates are provisional."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervision', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=50)
    args = parser.parse_args()
    root = args.supervision.resolve()
    while True:
        state = json.loads((root/'status.json').read_text())
        rows = []
        for condition in sorted((root/'evaluation_18x100').glob('G1*')):
            if not condition.is_dir():
                continue
            runs = sorted(condition.glob('run-*'), key=lambda p: p.stat().st_mtime)
            if not runs:
                continue
            episodes = []
            for path in runs[-1].glob('summaries/episode_*.json'):
                try:
                    episodes.append(json.loads(path.read_text()))
                except (json.JSONDecodeError, FileNotFoundError):
                    continue  # the official scorer may still be writing this file
            rows.append(dict(condition=condition.name, completed=len(episodes),
                successes=sum(bool(e['success']) for e in episodes),
                latest_seed=max((int(e['episode'].split('_')[-1]) for e in episodes), default=None)))
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        completed, successes = sum(r['completed'] for r in rows), sum(r['successes'] for r in rows)
        progress = dict(updated=datetime.now(timezone.utc).isoformat(), phase=state['phase'],
            provisional=state['phase']!='complete', completed_episodes=completed, total_episodes=1800,
            successes=successes, conditions=rows, gpu_returncode=gpu.returncode,
            gpu_telemetry=gpu.stdout.strip().splitlines())
        temporary = root/'evaluation_progress.tmp'
        temporary.write_text(json.dumps(progress, indent=2)+'\n')
        temporary.replace(root/'evaluation_progress.json')
        (root/'evaluation_progress.html').write_text('<!doctype html><meta charset="utf-8">'
            '<meta http-equiv="refresh" content="30"><title>ToolBench evaluation</title>'
            '<style>body{font:16px system-ui;max-width:900px;margin:2rem auto}pre{white-space:pre-wrap}</style>'
            '<h1>HumanoidToolBench evaluation</h1><p>Intermediate counts are provisional. '
            'Use the official validated summary for final results.</p><pre>'
            +html.escape(json.dumps(progress, indent=2))+'</pre>')
        print(f"{state['phase']}: episodes={completed}/1800 successes={successes} "
              f"validated_conditions={state.get('evaluated_conditions', 0)}/18", flush=True)
        if state['phase'] in ['complete', 'failed']:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
