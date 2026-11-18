"""Publish the CCT final to the existing queue only after transfer verification."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time
import zipfile

ROOT = Path('/root/workspace/humanoidVLA/cross_tracker')
RUN = ROOT / 'runs/scratch-eval-20261118-r4'
MODEL = 'box-aligned-cct-scratch-s42-20261118-r1'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('receipt', type=Path)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    relay = json.loads(args.receipt.read_text())
    verified = relay['checkpoint']
    assert verified['run_id'] == MODEL
    assert verified['status'] == 'verified_final' and verified['global_step'] == 200000
    assert verified['all_tensors_finite'] and verified['cct_encoder_keys'] > 0
    assert verified['backup'], 'Independent SenseCore backup receipt missing'
    assert set(verified['files']) == {'config.json', 'training_state.pt'}
    assert args.archive.stat().st_size == relay['bytes'] and sha(args.archive) == relay['sha256']
    target = ROOT / 'data/cache/eval-checkpoints' / MODEL / 'checkpoint_200000'
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        assert len(archive.namelist()) == 2 and set(archive.namelist()) == set(verified['files'])
        for name, info in verified['files'].items():
            destination = target / name
            if not destination.exists():
                partial = target / (name + '.partial')
                with archive.open(name) as source, partial.open('wb') as output:
                    shutil.copyfileobj(source, output)
                assert partial.stat().st_size == info['bytes'] and sha(partial) == info['sha256']
                partial.replace(destination)
            assert destination.stat().st_size == info['bytes'] and sha(destination) == info['sha256']
    manifest = dict(method='cct', run_id=MODEL, step=200000, global_step=200000,
                    checkpoint=str(target), files=verified['files'])
    published = RUN / 'cct-manifest.json'
    if published.exists():
        assert json.loads(published.read_text()) == manifest, 'Existing CCT manifest differs'
    else:
        audit = dict(checked_unix=time.time(), relay=relay, archive=str(args.archive),
                     checkpoint=str(target), status='verified_import')
        (RUN / 'cct-import-verification.json').write_text(json.dumps(audit, indent=2))
        temporary = published.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(published)
    print('CCT_FINAL_PUBLISHED', target, 'GPU rollout validation pending', flush=True)


if __name__ == '__main__':
    main()
