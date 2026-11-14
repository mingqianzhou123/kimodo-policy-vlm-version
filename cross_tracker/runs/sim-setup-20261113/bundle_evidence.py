"""Export bounded diagnostic records and a content manifest for local verification."""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parents[2]
folders = [root / 'runs/sim-setup-20261113', root / 'runs/interface-gpu-20261113-r3']
folders += sorted((root / 'runs').glob('sim-closed-loop-20261113*'))
files = set()
for folder in folders:
    for p in folder.rglob('*'):
        if p.is_file() and p.stat().st_size < 20_000_000 and '__pycache__' not in p.parts:
            files.add(p)
files.add(root / 'data/cache/text-eval/manifest.json')
files.add(root / 'data/cache/sim-assets/table-materials/manifest.json')
archive = root / 'readiness-simulation-evidence.zip'
records = []
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files):
        data = p.read_bytes()
        name = str(p.relative_to(root))
        records.append({'path':name, 'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()})
        z.writestr(name, data)
    z.writestr('evidence-manifest.json', json.dumps(records, indent=2))
print(json.dumps({'files':len(records), 'bytes':archive.stat().st_size, 'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
