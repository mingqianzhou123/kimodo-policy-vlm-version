"""Reproduce the existing non-Mimic overlays; accept only byte-identical frozen hashes."""
import json
from pathlib import Path
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from repaired_data import checksum

inputs = ROOT / 'data/cache/box-inputs'
manifest = json.loads((ROOT / 'data/manifests/balanced-v1.json').read_text())
raw_root = ROOT.parent.parent / 'data/HumanoidArena_dataset_v3_1'
link = inputs / 'audit/dataset'
link.parent.mkdir(parents=True, exist_ok=True)
if not link.exists():
    link.symlink_to(raw_root)
assert link.resolve() == raw_root.resolve()
for d in manifest['datasets']:
    if d['dataset'] not in ['HOI_pp_box/sonic_refpose_v3_1', 'HOI_pp_box/twist2_refpose_v3_1']:
        continue
    source = inputs / 'audit' / d['source']['path']
    assert checksum(source) == d['source']['sha256'], source
    table = pq.read_table(source).combine_chunks()
    action = table['action'].chunk(0).values.to_numpy().reshape(-1, 40)
    episode = table['episode_index'].to_numpy()
    eligible = np.zeros(len(table), dtype=bool)
    for start, count in zip(*np.unique(episode, return_index=True, return_counts=True)[1:]):
        eligible[start:start+count] = not np.any(np.linalg.norm(action[start:start+count, :2], axis=1) > .5)
    overlay = pa.table({'index': table['index'], 'episode_index': table['episode_index'],
        'frame_index': table['frame_index'], 'timestamp': table['timestamp'],
        'candidate.action': pa.FixedSizeListArray.from_arrays(pa.array(action.copy().reshape(-1)), 40),
        'numeric_eligible': pa.array(eligible)})
    dest = inputs / 'repair' / d['overlay']['path']
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        temporary = dest.with_suffix('.candidate.parquet')
        pq.write_table(overlay, temporary, compression='zstd')
        assert checksum(temporary) == d['overlay']['sha256'], f'Not byte-identical; transfer frozen overlay instead: {temporary}'
        temporary.rename(dest)
    assert checksum(dest) == d['overlay']['sha256'], dest
    print('FROZEN_OVERLAY_REPRODUCED', d['dataset'], pa.__version__, flush=True)
