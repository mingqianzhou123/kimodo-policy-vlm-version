"""Audit every selected raw episode and its independent official state/action contract.

Run in the training environment with HumanoidToolBench/src on PYTHONPATH.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from humanoidtoolbench.actions import DecoupledActionCodec
from humanoidtoolbench.policies.remote_humanoid import build_decoupled_policy_state
from kimodo_vla.benchmarks.replay import DirectPlanner
from kimodo_vla.benchmarks.toolbench import demos
from kimodo_vla.benchmarks.toolbench.adapter import ToolBenchAdapter, hand_targets
from kimodo_vla.controllers import decoupled_wbc


def audit(entry, base, files):
    data, video = base / entry['data'], base / entry['video']
    root = data.parents[2]
    episode = int(data.stem.split('_')[-1])
    meta = next(e for e in demos.episodes(root) if e['episode_index'] == episode)
    assert meta['episode_success'] and meta['simulator'] == 'mujoco', (data, meta)
    info = json.loads((root / 'meta/info.json').read_text())
    contract = json.loads((root / 'meta/simple_contract.json').read_text())
    assert info['fps'] == 50
    assert tuple(contract['joint_order']['observation.state'][:29]) == demos.BODY_JOINT_NAMES
    assert demos.canonical(root.parent.name) == entry['condition']
    assert demos.instruction(root.parent.name) == entry['instruction']
    for path in (data, video):
        expected = files[path.relative_to(base).as_posix()]
        assert path.stat().st_size == expected['size'], path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], path
    columns = demos.read_recording(root, episode)
    assert len(columns['action']) == entry['length']
    assert bool(np.any(columns['next.success'])), data
    for key, width in [('observation.state', 43), ('action', 43), ('teleop.navigate_command', 4)]:
        assert columns[key].shape == (entry['length'], width)
        assert np.isfinite(columns[key]).all()
    timing = pq.read_table(data, columns=['timestamp', 'frame_index'])
    times = np.array(timing['timestamp'])
    assert np.max(np.abs(np.diff(times) - .02)) < 1e-4, data
    demo = demos.ToolBenchDemo(columns, entry['instruction'])
    state = demos.policy_state(demo.state, demo.previous_height)
    for t in {0, len(demo) // 2, len(demo) - 1}:
        official = build_decoupled_policy_state({'joint_qpos': demo.state[t]}, demo.previous_height[t])
        np.testing.assert_allclose(state[t], official, atol=1e-7)
    motion = demo.commanded_motion()
    obs = demo.observations()[0]
    obs.step = 0
    chunk = DirectPlanner(motion, chunk=len(demo)-1).plan([obs])
    rebuilt = ToolBenchAdapter().to_response(chunk, {})
    recorded = demo.recorded_commands()[1:]
    error = np.abs(rebuilt - recorded)
    # Both thumb poses exist in the recordings. The architecture's two closure
    # scalars cannot preserve that distinction: report hand residual separately.
    np.testing.assert_allclose(rebuilt[:, 14:34], recorded[:, 14:34], atol=1e-4)
    angle_error = np.abs(np.angle(np.exp(1j * (rebuilt[:, 35] - recorded[:, 35]))))
    assert angle_error.max() < 1e-5
    decoded = list(DecoupledActionCodec.decode_chunk(rebuilt[:3]))
    assert len(decoded) == min(3, len(rebuilt))
    observed = demo.observations()
    for t in [1, min(81, len(demo)-1), len(demo)-1]:
        if t in demos.request_steps(len(demo)):
            np.testing.assert_allclose(observed[t].hand,
                demos.hand_closure(decoupled_wbc.command_to_natural_order(state[t:t+1, :14]))[0], atol=1e-6)
    return {'condition': entry['condition'], 'frames': len(demo),
            'motion_max_error': float(error[:, 14:34].max()),
            'heading_max_error': float(angle_error.max()),
            'hand_max_error': float(error[:, :14].max()),
            'hand_mean_error': float(error[:, :14].mean()),
            'turn_flag_disagreements': int(np.sum((np.abs(rebuilt[:,34]) >= .1) != (np.abs(recorded[:,34]) >= .1)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda e: audit(e, args.manifest.parent, manifest['files']), manifest['episodes']))
    result = {'passed': True, 'episodes': len(rows), 'frames': sum(r['frames'] for r in rows),
              'conditions': dict(Counter(r['condition'] for r in rows)),
              'motion_max_error': max(r['motion_max_error'] for r in rows),
              'heading_max_error': max(r['heading_max_error'] for r in rows),
              'hand_max_error': max(r['hand_max_error'] for r in rows),
              'turn_flag_disagreements': sum(r['turn_flag_disagreements'] for r in rows),
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(), 'rows': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
