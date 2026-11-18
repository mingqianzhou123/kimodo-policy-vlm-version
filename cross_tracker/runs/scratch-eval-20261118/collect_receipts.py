"""Export a read-only startup snapshot for downloading through JupyterLab."""
from pathlib import Path
import hashlib
import json
import time

run = Path('/root/workspace/humanoidVLA/cross_tracker/runs/scratch-eval-20261118-r2')
names = ['prepare.json', 'batch.json', 'source-files.json', 'sonic-manifest.json',
         'twist2-manifest.json', 'mix-manifest.json']
receipt = {'collected_unix': time.time(), 'remote_run': str(run), 'files': {}}
for name in names:
    raw = (run / name).read_bytes()
    receipt['files'][name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'data': json.loads(raw)}
receipt['runs'] = {}
for item in receipt['files']['batch.json']['data']['completed']:
    path = run / 'code/cross_tracker/runs' / item['run_id'] / 'run.json'
    receipt['runs'][item['run_id']] = json.loads(path.read_text())
destination = Path('/root/workspace/000-scratch-eval-evidence-20261118.json')
assert not destination.exists(), destination
destination.write_text(json.dumps(receipt, indent=2))
print('EVIDENCE', destination, destination.stat().st_size,
      hashlib.sha256(destination.read_bytes()).hexdigest())
