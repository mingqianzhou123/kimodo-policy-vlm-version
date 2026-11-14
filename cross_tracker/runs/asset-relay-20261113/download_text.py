"""Bounded parallel public HF ranges for this readiness recovery run."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path('cross_tracker')
sys.path.insert(0, str(root / 'scripts'))
from fetch_assets import matches

records = json.loads((root / 'data/manifests/weights-v1.json').read_text())['files']
chunk = 32 * 1024**2
jobs = []
pending = {}
for name, record in records.items():
    dest = root / 'data/cache/weights' / name
    if dest.exists():
        assert matches(dest, record), name
        continue
    parts = root / 'data/cache/weight-ranges' / name
    parts.mkdir(parents=True, exist_ok=True)
    pending[name] = []
    for offset in range(0, record['bytes'], chunk):
        end = min(offset + chunk, record['bytes']) - 1
        part = parts / f'{offset:012d}'
        pending[name].append(part)
        if part.exists():
            assert part.stat().st_size == end-offset+1
        else:
            jobs.append((record['url'], offset, end, part))

def fetch(job):
    url, start, end, part = job
    temp = part.with_suffix('.partial')
    command = ['curl', '-L', '-f', '-sS', '--retry', '8', '--retry-all-errors',
               '--retry-delay', '2', '--connect-timeout', '20', '--max-time', '180',
               '--max-filesize', str(end-start+1), '--range', f'{start}-{end}',
               '-o', str(temp), url]
    subprocess.run(command, check=True)
    assert temp.stat().st_size == end-start+1, part
    temp.rename(part)
    return part

with ThreadPoolExecutor(max_workers=12) as pool:
    futures = [pool.submit(fetch, job) for job in jobs]
    for i, future in enumerate(as_completed(futures), 1):
        print('RANGE', i, len(jobs), future.result(), flush=True)

for name, parts in pending.items():
    dest = root / 'data/cache/weights' / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    joined = dest.with_suffix(dest.suffix + '.joined')
    with joined.open('xb') as target:
        for part in parts:
            with part.open('rb') as source:
                shutil.copyfileobj(source, target)
    assert matches(joined, records[name]), name
    joined.rename(dest)
    print('VERIFIED_TEXT_ASSET', name, flush=True)
print('ALL_TEXT_WEIGHTS_VERIFIED', flush=True)
