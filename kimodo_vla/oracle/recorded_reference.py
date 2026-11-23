"""Immutable original HTTP action chunks, without VLM, Kimodo, IK or feedback replanning."""
import hashlib
import json
from pathlib import Path
import numpy as np
from kimodo_vla.core.types import yaw_matrix, yaw_of
from .geometry import rotation_wxyz


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def export_reference(source, destination):
    source, destination = Path(source), Path(destination)
    executions = read_rows(source/'execution.jsonl')
    plans = read_rows(source/'plans.jsonl')
    initial = executions[0]['before']
    rotation = rotation_wxyz(initial['root_quat'])
    heading = yaw_matrix(float(yaw_of(rotation)))
    warmup = np.r_[0., 0., initial['root_pos'][2], (heading.T@rotation)[:,:2].ravel(), initial['joints'], 0., 0.]
    chunks = [np.tile(warmup, (5, 1))]
    plan_paths = sorted(source.glob('plan-*.npz'))
    if len(plan_paths) != len(plans):
        raise ValueError('Plan archive/count mismatch')
    chunks += [np.load(path)['actions'] for path in plan_paths]
    lengths = np.array([len(chunk) for chunk in chunks], dtype=int)
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    times = np.array([initial['time']]+[plan['time'] for plan in plans])
    if not np.allclose(times, initial['time']+offsets/50, atol=1e-5, rtol=0):
        raise ValueError('Original request times do not match recorded chunk schedule')
    actions = np.concatenate(chunks)
    if actions.shape[1] != 40 or not np.isfinite(actions).all() or np.any(actions[:,-2:] != 0):
        raise ValueError('Invalid action data or unexpected gripper command')
    if not offsets[-1] < len(executions) <= len(actions):
        raise ValueError('Consumed frame count inconsistent with chunks')
    destination.mkdir(parents=True, exist_ok=False)
    arrays = dict(actions=actions, chunk_lengths=lengths, request_times=times)
    for field in ('reference_joints','reference_root_world','reference_root_quat','reference_wrist_world'):
        arrays[field] = np.array([row[field] for row in executions])
    arrays['after_times'] = np.array([row['after']['time'] for row in executions])
    np.savez_compressed(destination/'reference.npz', **arrays)
    target = json.loads((source/'target.json').read_text())
    (destination/'target.json').write_text(json.dumps(target, indent=2))
    metadata = json.loads((source/'run.json').read_text())
    paths = plan_paths+[source/name for name in ('execution.jsonl','plans.jsonl','target.json','run.json')]
    manifest = dict(source_run=source.name, source_sha256={p.name:sha(p) for p in paths},
        fixture_sha256={name:sha(destination/name) for name in ('reference.npz','target.json')},
        initial_sample=initial, consumed_frames=len(executions), produced_frames=len(actions),
        warmup='5 commands reconstructed exactly from original policy warmup formula',
        tracker_sha256=metadata['tracker_sha256'], arena_sha256=metadata['arena_sha256'],
        marker_reach_rad=metadata['marker_reach_rad'], camera_lateral_m=metadata['camera_lateral_m'])
    (destination/'replay.json').write_text(json.dumps(manifest,indent=2))
    return manifest


class RecordedReference:
    def __init__(self, run, fixture):
        self.run, self.fixture = Path(run), Path(fixture)
        self.metadata = json.loads((self.fixture/'replay.json').read_text())
        for name, expected in self.metadata['fixture_sha256'].items():
            if sha(self.fixture/name) != expected:
                raise ValueError(f'Fixture hash mismatch: {name}')
        with np.load(self.fixture/'reference.npz') as data:
            self.actions = data['actions'].copy()
            self.lengths = data['chunk_lengths'].copy()
            self.times = data['request_times'].copy()
        self.offsets = np.r_[0, np.cumsum(self.lengths)]
        self.target = json.loads((self.fixture/'target.json').read_text())
        self.reset()

    def reset(self, seed=None):
        self.chunk = 0
        return {'status':'ok'}

    def __call__(self, request):
        if self.chunk >= len(self.lengths):
            raise ValueError('Recorded trajectory exhausted; refusing to invent a hold command')
        samples = request['observation']['oracle_history']
        if not samples:
            raise ValueError('Timestamped observation required')
        if self.chunk == 0:
            initial = self.metadata['initial_sample']
            for name in ('time','root_pos','root_quat','joints','reference_root_pos','reference_root_quat','reference_joints'):
                if not np.allclose(samples[0][name], initial[name], atol=1e-6, rtol=0):
                    raise ValueError(f'Initial state mismatch: {name}')
            (self.run/'target.json').write_text(json.dumps(self.target,indent=2))
        if abs(samples[-1]['time']-self.times[self.chunk]) > 1e-5:
            raise ValueError(f'Request timing mismatch at chunk {self.chunk}')
        begin, end = self.offsets[self.chunk:self.chunk+2]
        actions = self.actions[begin:end].tolist()
        record = dict(plan=self.chunk, time=samples[-1]['time'], source='replay',
            model_loaded=False, action_offset=int(begin), commands=len(actions),
            original_source=self.metadata['source_run'])
        with (self.run/'plans.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        with (self.run/'commands.jsonl').open('a') as f:
            for i, action in enumerate(actions):
                f.write(json.dumps(dict(frame=int(begin)+i,action=action))+'\n')
        self.chunk += 1
        return {'action_chunk':actions, 'oracle_target':self.target}
