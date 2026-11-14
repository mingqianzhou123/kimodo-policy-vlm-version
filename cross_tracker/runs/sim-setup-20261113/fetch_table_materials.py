"""Cache the official external MDL dependencies without changing their content."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import re
import urllib.request
import zipfile

root = Path(__file__).resolve().parents[2]
dest = root / 'data/cache/relay/table-materials'
dest.mkdir(parents=True, exist_ok=True)
base = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/materials/'
materials = ['MetalPainted_White_Glossy_A', 'Metal_Glossy_A', 'Wood_Laminate_Gray_A', 'Wood_Pressed_A']
records = []

def fetch(relative):
    assert '..' not in Path(relative).parts
    p = dest / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with urllib.request.urlopen(base + relative, timeout=60) as response:
            data = response.read()
        p.write_bytes(data)
    data = p.read_bytes()
    return {'path': relative, 'url': base + relative, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    records.extend(pool.map(fetch, [name + '.mdl' for name in materials]))
    textures = set()
    for name in materials:
        textures.update(re.findall(r'texture_2d\("\./([^"]+)"', (dest / (name + '.mdl')).read_text()))
    records.extend(pool.map(fetch, sorted(textures)))
(dest / 'manifest.json').write_text(json.dumps(records, indent=2))
archive = dest.with_suffix('.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as z:
    for p in sorted(dest.rglob('*')):
        if p.is_file(): z.write(p, p.relative_to(dest))
print(json.dumps({'files':len(records), 'archive':str(archive), 'bytes':archive.stat().st_size, 'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
