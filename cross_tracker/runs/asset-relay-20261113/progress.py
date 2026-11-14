"""Compact live evidence from this readiness run, without touching running jobs."""
import json
from pathlib import Path
import time

root = Path('cross_tracker')
run = root / 'runs/readiness-20261113-r2'
bundles = json.loads((run / 'weight-upload-bundles.json').read_text())
transferred = total = 0
for name, item in bundles.items():
    if not name.startswith('text-weights-'):
        continue
    total += item['bytes']
    path = root / name
    if not path.exists():
        path = root / 'data/cache/setup' / name
    transferred += min(path.stat().st_size, item['bytes']) if path.exists() else 0
audit = run / 'full-data-check.json'
load = root / 'runs/gpu-load-20261113/result.json'
status = {'utc': time.strftime('%H:%M:%S', time.gmtime()),
          'prepared': len(list((root / 'data/cache/prepared-v1').rglob('*.json'))),
          'audited': len(json.loads(audit.read_text())['prepared_checks']) if audit.exists() else 0,
          'coverage_verified': (run / 'dataset-coverage.json').exists(),
          'gpu_loaded': load.exists(),
          'text_upload_GB': round(transferred/1e9, 2),
          'text_upload_percent': round(100*transferred/total, 1),
          'weights_verified': (run / 'weights-complete.json').exists()}
(run / 'latest-status.json').write_text(json.dumps(status))
with (run / 'progress.jsonl').open('a') as log:
    log.write(json.dumps(status)+'\n')
print(json.dumps(status))
