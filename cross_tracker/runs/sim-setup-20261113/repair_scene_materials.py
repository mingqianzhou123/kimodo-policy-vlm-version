"""Correct the exported material name in the isolated evaluation assets."""
from pathlib import Path
import hashlib
import json

setup = Path(__file__).resolve().parent
root = setup.parents[1] / 'data/cache/sim-assets/arena/assets'
backup = setup / 'material-originals'
backup.mkdir(exist_ok=False)
records = []
for variant in ('', 'small_warehouse_box', 'small_warehouse_boxing', 'small_warehouse_vision_navigation'):
    path = root / 'objects/small_warehouse' / variant / 'SubUSDs/materials/M_TrafficCone.mdl'
    original = path.read_bytes()
    exported = path.with_name('OmniUe4Base_1.mdl').read_text()
    assert 'export material OmniUe4Base(' in exported
    assert original.count(b'::OmniUe4Base_1(') == 1
    fixed = original.replace(b'::OmniUe4Base_1(', b'::OmniUe4Base(')
    (backup / f'{variant or "root"}.mdl').write_bytes(original)
    path.write_bytes(fixed)
    records.append({'path': str(path.relative_to(root)),
        'before_sha256': hashlib.sha256(original).hexdigest(),
        'after_sha256': hashlib.sha256(fixed).hexdigest()})
(setup / 'material-repair.json').write_text(json.dumps(records, indent=2))
print('Repaired isolated assets:', len(records))
