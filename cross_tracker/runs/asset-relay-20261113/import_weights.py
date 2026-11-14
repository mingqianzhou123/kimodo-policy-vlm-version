"""Validate this relay's five archives and every pinned weight before use."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

root = Path('cross_tracker')
run = root / 'runs/readiness-20261113-r2'
sys.path.insert(0, str(root / 'scripts'))
from fetch_assets import matches

bundles = json.loads((run / 'weight-upload-bundles.json').read_text())
assets = json.loads((root / 'data/manifests/weights-v1.json').read_text())['files']
imported = []
for name, expected in bundles.items():
    archive = root / name
    for attempt in range(720):
        if archive.exists() and archive.stat().st_size == expected['bytes']:
            break
        if attempt % 20 == 0:
            print('WAITING_FOR_UPLOAD', name, flush=True)
        time.sleep(15)
    else:
        raise TimeoutError(name)
    assert hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest() == expected['sha256'], name
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            relative = Path(item.filename).relative_to('weights').as_posix()
            destination = root / 'data/cache/weights' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                assert matches(destination, assets[relative]), relative
            else:
                incoming = destination.with_name(destination.name + '.incoming')
                with z.open(item) as source, incoming.open('xb') as target:
                    shutil.copyfileobj(source, target)
                assert matches(incoming, assets[relative]), relative
                incoming.rename(destination)
            imported.append(relative)
    archive.rename(root / 'data/cache/setup' / name)
    print('IMPORTED_WEIGHT_ARCHIVE', name, flush=True)
verified = {name: matches(root / 'data/cache/weights' / name, record) for name, record in assets.items()}
assert all(verified.values())
(run / 'weights-complete.json').write_text(json.dumps(verified, indent=2)+'\n')
print('ALL_WEIGHTS_VERIFIED', len(verified), flush=True)
