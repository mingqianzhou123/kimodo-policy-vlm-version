"""Rebuild only observation histories for a changed execution length; share RGB arrays."""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import torch


def worker(job):
    recording, episode, source, output, execution = job
    torch.set_num_threads(1)
    from kimodo_vla.benchmarks.toolbench.demos import ToolBenchDemo, read_recording
    from kimodo_vla.model.features import KimodoFeatures
    from kimodo_vla.core.types import yaw_matrix, yaw_of
    source, output = Path(source), Path(output)
    old = torch.load(source, map_location='cpu', weights_only=True)
    demo = ToolBenchDemo(read_recording(Path(recording), episode), old['instruction'], execution_frames=execution)
    observations = demo.observations()
    features = KimodoFeatures(50)
    motion, mask, _, _ = features.history(observations, yaw_matrix(float(yaw_of(observations[0].root_rot))))
    hand = features.hand_history(observations)
    assert motion.shape==old['target_motion'].shape and hand.shape==old['hand'].shape
    assert torch.isfinite(motion).all() and torch.isfinite(hand).all()
    new = dict(old, observed_motion=motion.float(), observed_mask=mask[0].clone(),
        observed_hand=hand.float(), history_execution_frames=execution)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix('.tmp')
    torch.save(new,temporary);temporary.replace(output)
    images = output.with_suffix('.frames.npy')
    if not images.exists():images.symlink_to(source.with_suffix('.frames.npy').resolve())
    return {'episode':output.stem,'frames':len(hand)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execution-frames',type=int,required=True);p.add_argument('--workers',type=int,default=24)
    p.add_argument('--manifest',type=Path,default=Path('/root/workspace/ToolBook/manifest.json'))
    a=p.parse_args();assert 1<a.execution_frames<=50
    from kimodo_vla.benchmarks.toolbench.demos import canonical
    raw=a.manifest.parent/'raw';jobs=[]
    manifest=json.loads(a.manifest.read_text())
    for row in manifest['episodes']:
        data=a.manifest.parent/row['data'];recording=data.parents[2]
        episode=int(data.stem.split('_')[-1]);name='_'.join(recording.relative_to(raw).parts)
        rel=Path(canonical(recording.parent.name))/f'{name}_{episode:06d}.pt'
        source=a.source/rel;assert source.is_file(),source
        jobs.append((str(recording),episode,str(source),str(a.out/rel),a.execution_frames))
    a.out.mkdir(parents=True,exist_ok=True)
    settings={'execution_frames':a.execution_frames,'prediction_frames':50,'source':str(a.source.resolve()),
        'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'episodes':len(jobs)}
    old=a.out/'preparation.json'
    if old.exists():assert json.loads(old.read_text())==settings,'different history preparation'
    old.write_text(json.dumps(settings,indent=2)+'\n')
    with mp.get_context('spawn').Pool(a.workers) as pool:
        results=list(pool.imap_unordered(worker,jobs))
    assert len(results)==3003
    (a.out/'history_audit.json').write_text(json.dumps(dict(passed=True,
        frames=sum(r['frames'] for r in results),image_arrays='symlinked; unchanged targets and source images',**settings),indent=2)+'\n')
    print(f'Validated {len(results)} histories at execution_frames={a.execution_frames}',flush=True)


if __name__=='__main__':main()
