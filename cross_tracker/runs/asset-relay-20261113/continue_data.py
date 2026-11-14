"""Wait for this upload batch, convert its six datasets, then audit all 744 episodes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('cross_tracker')
run = root / 'runs/readiness-20261113-r2'
sizes = json.loads((run / 'remaining-upload-sizes.json').read_text())
for attempt in range(180):
    ready = [name for name, size in sizes.items()
             if (root / name).exists() and (root / name).stat().st_size == size]
    if len(ready) == len(sizes):
        break
    if attempt % 4 == 0:
        print('WAITING_FOR_UPLOAD', len(ready), len(sizes), flush=True)
    time.sleep(20)
else:
    raise TimeoutError('Video upload did not finish within one hour')
subprocess.run([sys.executable, '-u', str(run / 'import_videos.py'), *sorted(sizes)], check=True)
for attempt in range(180):
    count = len(list((root / 'data/cache/prepared-v1').rglob('*.json')))
    if count == 744:
        break
    print('WAITING_FOR_BOX_CONVERSION', count, flush=True)
    time.sleep(20)
else:
    raise TimeoutError('Not all frozen episodes were prepared')
command = [sys.executable, '-u', 'cross_tracker/scripts/check_data.py',
           '--paths', 'cross_tracker/configs/paths.local.json',
           '--prepared', 'cross_tracker/data/cache/prepared-v1',
           '--videos', 'cross_tracker/data/cache/videos',
           '--report', str(run / 'full-data-check.json')]
with (run / 'full-data-check.log').open('x') as log:
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                   env={**os.environ, 'PYTHONPATH': 'cross_tracker/data/cache/python'})
report = json.loads((run / 'full-data-check.json').read_text())
assert len(report['prepared_checks']) == 744
print('FULL_DATA_CHECK_PASSED', len(report['prepared_checks']), flush=True)
