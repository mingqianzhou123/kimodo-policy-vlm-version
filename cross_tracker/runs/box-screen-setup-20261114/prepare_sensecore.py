"""One-time CPU deployment audit and repaired box preparation on SenseCore."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SETUP = Path(__file__).resolve().parent
HOME_DIR = ROOT.parent.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT.parent)]
from repaired_data import checksum, read_segment
from fetch_assets import matches


def load(path):
    return json.loads(path.read_text())


def link(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        assert path.resolve() == target.resolve(), path
    else:
        path.symlink_to(target)


os.chdir(ROOT.parent)
for name, expected in load(SETUP / 'dependency-files.json').items():
    assert checksum(ROOT.parent / name) == expected, name
inputs = ROOT / 'data/cache/box-inputs'
subprocess.run([sys.executable, str(SETUP / 'reproduce_box_inputs.py')], check=True)
roots = {key: str(inputs / key) for key in ('audit', 'repair')}
paths = SETUP / 'paths.json'
paths.write_text(json.dumps(roots, indent=2) + '\n')
manifest = load(ROOT / 'data/manifests/balanced-v1.json')
assets = load(ROOT / 'data/manifests/assets-v1.json')
datasets = [d for d in manifest['datasets'] if d['dataset'] in
            ['HOI_pp_box/sonic_refpose_v3_1', 'HOI_pp_box/twist2_refpose_v3_1']]
assert len(datasets) == 2
for d in datasets:
    assert len(d['train']) == 43 and len(d['validation']) == 11
    for e in d['episodes']:
        read_segment(d, e, roots, assets)
print('NUMERIC_OK 108', flush=True)
verified = {}
for name, meta in assets['videos'].items():
    if any(name.startswith(d['dataset'] + '/') for d in datasets):
        path = HOME_DIR / 'data/HumanoidArena_dataset_v3_1' / name
        assert path.stat().st_size == meta['bytes'] and checksum(path) == meta['sha256'], path
        link(ROOT / 'data/cache/videos' / name, path)
        verified[str((ROOT / 'data/cache/videos' / name).relative_to(ROOT))] = meta['sha256']
link(ROOT / 'data/cache/weights', HOME_DIR / 'weights/kimodo-policy')
relative = load(ROOT / 'configs/box-screen.json')['initial_checkpoint']['path']
weights = load(ROOT / 'data/manifests/weights-v1.json')['files']
for name in [relative + '/config.json', relative + '/training_state.pt']:
    path = ROOT / 'data/cache/weights' / name
    assert matches(path, weights[name]), path
    verified[str(path.relative_to(ROOT))] = checksum(path)
for name, meta in load(ROOT / 'data/manifests/shared-weights-v1.json')['files'].items():
    path = ROOT / 'data/cache/weights' / name
    assert matches(path, meta), path
    verified[str(path.relative_to(ROOT))] = checksum(path)
link(ROOT / 'data/cache/text/HOI_pp_box.pt', HOME_DIR / 'cache/arena/HOI_pp_box.pt')
verified['data/cache/text/HOI_pp_box.pt'] = checksum(ROOT / 'data/cache/text/HOI_pp_box.pt')
print('VIDEOS_WEIGHTS_OK', flush=True)
subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_kimodo.py'), '--paths', str(paths),
                '--video-root', str(ROOT / 'data/cache/videos'), '--datasets',
                *[d['dataset'] for d in datasets]], check=True)
from check_data import check_prepared
checks = check_prepared(ROOT / 'data/cache/prepared-v1', ROOT / 'data/cache/videos')
assert len(checks) == 108
for d in datasets:
    task, source = d['dataset'].replace('_refpose_v3_1', '').split('/')
    for split in ('train', 'validation'):
        folder = ROOT / 'data/cache/prepared-v1' / split / task / source
        assert {int(p.stem.split('_')[-1]) for p in folder.glob('*.json')} == set(d[split])
        for f in folder.iterdir():
            if f.is_file():
                verified[str(f.relative_to(ROOT))] = checksum(f)
report = {'passed': True, 'prepared_episodes': len(checks), 'manifest_sha256': checksum(ROOT / 'data/manifests/balanced-v1.json'),
          'files': verified, 'prepared_checks': checks, 'success_certified': False}
(SETUP / 'assets-verified.json').write_text(json.dumps(report, indent=2) + '\n')
print('SENSECORE_ASSETS_VERIFIED', len(checks), flush=True)
