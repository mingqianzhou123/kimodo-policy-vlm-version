"""Signed, aligned root / torso / arm decomposition and exact replay audit."""
import argparse
import json
from pathlib import Path
import numpy as np
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.oracle.geometry import XmlFK, rotation_wxyz, angle_deg
from kimodo_vla.oracle.recorded_reference import read_rows, sha

from paths import G1_XML
ARMS=[i for i,n in enumerate(G1_JOINT_NAMES) if any(k in n for k in ('shoulder','elbow','wrist'))]
RIGHT=[i for i in ARMS if G1_JOINT_NAMES[i].startswith('right_')]


def vector_stats(v, scale=100):
    v=np.asarray(v)*scale
    n=np.linalg.norm(v,axis=-1)
    bias=v.mean(axis=0)
    return dict(mean_norm=n.mean(axis=0).tolist(),p95_norm=np.percentile(n,95,axis=0).tolist(),
        max_norm=n.max(axis=0).tolist(),bias_xyz=bias.tolist(),bias_norm=np.linalg.norm(bias,axis=-1).tolist(),
        demeaned_rms=np.sqrt(np.mean(np.sum((v-bias)**2,axis=-1),axis=0)).tolist(),
        first_last_half_bias_change=np.linalg.norm(v[len(v)//2:].mean(axis=0)-v[:len(v)//2].mean(axis=0),axis=-1).tolist())


def components(e,fk):
    a=e['after']; q=np.array(a['joints']); rq=np.array(e['reference_joints'])
    p=np.array(a['root_pos']); rp=np.array(e['reference_root_world'])
    R=rotation_wxyz(a['root_quat']); Rr=rotation_wxyz(e['reference_root_quat'])
    wrist=np.array(a['wrist_pos']); ref=np.array(e['reference_wrist_world'])
    rel=(ref-rp)@Rr
    rooted=p+rel@R.T
    qmix=q.copy();qmix[ARMS]=rq[ARMS]
    mixed=fk(qmix,p,R)[0]; full=fk(q,p,R)[0]
    c=dict(root_translation=np.broadcast_to(p-rp,(2,3)),root_rotation=rel@(R-Rr).T,
        root_relative=wrist-rooted,torso=mixed-rooted,arm=full-mixed,fk_residual=wrist-full,
        tracking=wrist-ref)
    combined=c['root_translation']+c['root_rotation']+c['torso']+c['arm']+c['fk_residual']
    if not np.allclose(combined,c['tracking'],atol=1e-10,rtol=0):
        raise ValueError('Error decomposition identity failed')
    return c,angle_deg(R,Rr)


def summarize(run):
    run=Path(run); rows=read_rows(run/'execution.jsonl')
    fk=XmlFK(G1_XML,G1_JOINT_NAMES)
    parts=[components(e,fk) for e in rows]
    target=json.loads((run/'target.json').read_text())
    times=np.array([e['after']['time'] for e in rows]); start=rows[0]['before']['time']
    output=dict(run=run.name,frames=len(rows),units='cm for vectors; degrees for angles')
    for window,mask in dict(post_arrival=times>target['arrival_time']+1e-6,steady_4_8s=times>=start+4).items():
        if not mask.any():raise ValueError('No frames in diagnostic window')
        values={k:np.array([p[0][k] for p in parts])[mask] for k in parts[0][0]}
        error=values['tracking'];direction=error/np.maximum(np.linalg.norm(error,axis=-1,keepdims=True),1e-12)
        summary={k:vector_stats(v) for k,v in values.items()}
        summary['signed_contribution_cm']={k:(100*np.sum(v*direction,axis=-1)).mean(axis=0).tolist()
            for k,v in values.items() if k not in ('tracking','root_relative')}
        summary['root_rotation_deg']=dict(mean=float(np.mean([parts[i][1] for i in np.flatnonzero(mask)])),
            max=float(max(parts[i][1] for i in np.flatnonzero(mask))))
        measured=np.array([r['after']['joints'] for r in rows])[mask]
        ref=np.array([r['reference_joints'] for r in rows])[mask]
        decoded=np.array([r['decoder_joint_target'] for r in rows])[mask]
        summary['right_joints']={}
        for name,diff in dict(actual_reference=measured-ref,decoder_reference=decoded-ref,actual_decoder=measured-decoded).items():
            deg=np.degrees(diff)
            summary['right_joints'][name]={G1_JOINT_NAMES[i]:dict(bias=float(deg[:,i].mean()),
                rmse=float(np.sqrt(np.mean(deg[:,i]**2))),std=float(deg[:,i].std()),
                p95_abs=float(np.percentile(abs(deg[:,i]),95)),max_abs=float(abs(deg[:,i]).max())) for i in RIGHT}
        summary['reference_joint_range_deg']=float(np.degrees(np.ptp(ref,axis=0)).max())
        summary['frames']=int(mask.sum());output[window]=summary
    return output


def audit(run,fixture,source):
    run,fixture,source=map(Path,(run,fixture,source))
    manifest=json.loads((fixture/'replay.json').read_text())
    for name,h in manifest['source_sha256'].items():
        if sha(source/name)!=h:raise ValueError(f'Source changed: {name}')
    for name,h in manifest['fixture_sha256'].items():
        if sha(fixture/name)!=h:raise ValueError(f'Fixture changed: {name}')
    meta=json.loads((run/'run.json').read_text())
    if meta['generator_loaded'] or meta['vlm_model'] or meta['reference_source']!='replay':
        raise ValueError('Replay unexpectedly uses a generator')
    for key in ('tracker_sha256','arena_sha256'):
        if meta[key]!=manifest[key]:raise ValueError(f'Environment mismatch: {key}')
    source_meta=json.loads((source/'run.json').read_text())
    for key in ('marker_reach_rad','camera_lateral_m','vision_camera'):
        if meta[key]!=source_meta[key]:
            raise ValueError(f'Scene configuration changed: {key}')
    for key in ('arena/oracle_bridge.py','scripts/oracle_sim_bootstrap.py'):
        if meta['source_sha256'][key]!=source_meta['source_sha256'][key]:
            raise ValueError(f'Instrumentation changed: {key}')
    if json.loads((run/'sim-contract.json').read_text())!=json.loads((source/'sim-contract.json').read_text()):
        raise ValueError('Simulation dt/joint/body contract changed')
    if sha(run/'target.json')!=manifest['fixture_sha256']['target.json']:
        raise ValueError('Evaluation target changed')
    a=read_rows(run/'execution.jsonl');b=read_rows(source/'execution.jsonl')
    if len(a)!=len(b) or len(a)!=manifest['consumed_frames']:raise ValueError('Wrong frame count')
    out={}
    with np.load(fixture/'reference.npz') as data:
        commands=read_rows(run/'commands.jsonl')
        if not np.array_equal(np.array([c['action'] for c in commands]),data['actions']):
            raise ValueError('Emitted actions changed')
        for key in ('reference_joints','reference_root_world','reference_root_quat','reference_wrist_world'):
            delta=np.array([r[key] for r in a])-data[key]
            out[key+'_max_abs_delta']=float(abs(delta).max())
            if abs(delta).max()>1e-6:raise ValueError(f'Consumed reference mismatch: {key}: {abs(delta).max()}')
        if np.max(abs(np.array([r['after']['time'] for r in a])-data['after_times']))>1e-5:
            raise ValueError('Consumed frame timestamps differ')
    out['actual_wrist_replay_vs_original_cm']=vector_stats(np.array([r['after']['wrist_pos'] for r in a])-np.array([r['after']['wrist_pos'] for r in b]))
    out['passed']=True
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--fixture',type=Path);p.add_argument('--source',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=summarize(a.run)
    if a.fixture:result['replay_audit']=audit(a.run,a.fixture,a.source)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    s=result['post_arrival'];print(a.run.name,'right cm:',{k:round(s[k]['mean_norm'][1],4) for k in ('tracking','root_translation','root_rotation','root_relative','torso','arm')})
