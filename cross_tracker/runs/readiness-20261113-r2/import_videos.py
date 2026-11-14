"""This relay's validated upload import; start only after the browser upload finishes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

root = Path('cross_tracker')
sys.path.insert(0, str(root / 'scripts'))
from fetch_assets import matches

run = root / 'runs/readiness-20261113-r2'
bundles = json.loads((run / 'video-bundles.json').read_text())
assets = json.loads((root / 'data/manifests/assets-v1.json').read_text())['videos']
names = sys.argv[1:]
imported = []
for name in names:
    archive = root / name
    assert hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest() == bundles[name], name
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            relative = Path(item.filename).relative_to('videos').as_posix()
            expected = assets[relative]
            destination = root / 'data/cache/videos' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                assert matches(destination, expected), relative
            else:
                incoming = destination.with_name(destination.name + '.incoming')
                with z.open(item) as source, incoming.open('xb') as target:
                    shutil.copyfileobj(source, target)
                assert matches(incoming, expected), relative
                incoming.rename(destination)
            imported.append(relative)
    archive.rename(root / 'data/cache/setup' / name)
receipt = run / (names[0] + '.import.json')
receipt.write_text(json.dumps(imported, indent=2))
print('IMPORTED', len(imported), flush=True)

datasets = sorted({'/'.join(name.split('/')[:2]) for name in imported})
def prepare(dataset):
    log = run / ('prepare-' + dataset.replace('/', '-') + '.log')
    command = [sys.executable, '-u', 'cross_tracker/scripts/prepare_kimodo.py',
               '--paths', 'cross_tracker/configs/paths.local.json',
               '--video-root', 'cross_tracker/data/cache/videos', '--datasets', dataset]
    with log.open('x') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, 'PYTHONPATH': 'cross_tracker/data/cache/python'})
    log.with_suffix('.exit.json').write_text(json.dumps({'returncode': result.returncode}))
    print('PREPARE_EXIT', dataset, result.returncode, flush=True)
    return result.returncode
with ThreadPoolExecutor(max_workers=3) as pool:
    codes = list(pool.map(prepare, datasets))
raise SystemExit(any(codes))
