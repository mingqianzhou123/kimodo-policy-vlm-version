"""Replace broken HTTP MDL references with unchanged, verified local dependencies."""
from pathlib import Path
import hashlib
import json
import os
import zipfile

root = Path(__file__).resolve().parents[2]
assets = root / 'data/cache/sim-assets/arena/assets'
dest = root / 'data/cache/sim-assets/table-materials'
archive = root / 'table-materials.zip'
assert not dest.exists()
with zipfile.ZipFile(archive) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(dest)
for entry in json.loads((dest / 'manifest.json').read_text()):
    p = dest / entry['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == entry['sha256']
backup = root / 'data/cache/setup/table-usd-originals'
backup.mkdir(exist_ok=False)
base = b'http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/materials/'
records = []
for p in assets.rglob('*.usd'):
    original = p.read_bytes()
    if base not in original:
        continue
    assert original.startswith(b'#usda '), p
    fixed = original
    replacements = 0
    for mdl in dest.glob('*.mdl'):
        old = base + mdl.name.encode()
        replacements += fixed.count(old)
        fixed = fixed.replace(old, os.path.relpath(mdl, p.parent).encode())
    assert base not in fixed
    relative = p.relative_to(assets)
    saved = backup / relative
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_bytes(original)
    p.write_bytes(fixed)
    records.append({'path':str(relative),'replacements':replacements,
        'before_sha256':hashlib.sha256(original).hexdigest(),
        'after_sha256':hashlib.sha256(fixed).hexdigest()})
(root / 'runs/sim-setup-20261113/table-material-localization.json').write_text(json.dumps(records,indent=2))
archive.rename(root / 'data/cache/setup/table-materials.zip')
print(json.dumps({'localized_usd_files':len(records),'references':sum(x['replacements'] for x in records)}))
