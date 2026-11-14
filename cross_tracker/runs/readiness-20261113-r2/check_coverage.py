"""Independently match all converted identities and splits to the frozen protocol."""
from collections import Counter
import hashlib
import json
from pathlib import Path

root = Path('cross_tracker')
run = root / 'runs/readiness-20261113-r2'
manifest_path = root / 'data/manifests/balanced-v1.json'
manifest = json.loads(manifest_path.read_text())
digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
prepared = root / 'data/cache/prepared-v1'
expected = {}
for dataset in manifest['datasets']:
    name = dataset['dataset'].removesuffix('_refpose_v3_1')
    for episode in dataset['episodes']:
        relative = f"{episode['split']}/{name}/episode_{episode['episode']:06d}.json"
        expected[relative] = (episode['split'], f"{name}/{episode['episode']:06d}")
actual = {str(p.relative_to(prepared)): p for p in prepared.rglob('*.json')}
assert set(actual) == set(expected), {'missing': sorted(set(expected)-set(actual)),
                                      'extra': sorted(set(actual)-set(expected))}
counts = Counter()
for relative, path in actual.items():
    receipt = json.loads(path.read_text())
    split, episode_id = expected[relative]
    assert receipt['split'] == split and receipt['manifest_sha256'] == digest, relative
    counts[episode_id.rsplit('/', 1)[0]+'/'+split] += 1
audit = json.loads((run / 'full-data-check.json').read_text())
ids = [item['episode'] for item in audit['prepared_checks']]
assert len(ids) == len(set(ids)) == len(expected) == 744
assert set(ids) == {episode_id for _, episode_id in expected.values()}
assert audit['manifest_sha256'] == digest
result = {'episodes': len(ids), 'train': sum(v[0] == 'train' for v in expected.values()),
          'validation': sum(v[0] == 'validation' for v in expected.values()),
          'counts': dict(sorted(counts.items())), 'manifest_sha256': digest,
          'exact_frozen_coverage': True, 'demonstration_success': 'unverified'}
(run / 'dataset-coverage.json').write_text(json.dumps(result, indent=2)+'\n')
print('FROZEN_COVERAGE_VERIFIED', result['episodes'], result['train'], result['validation'])
