"""Align generated, consumed-reference and actual wrist positions at 50 Hz.

Keeps coordinate integration error separate from the tracker's error. Reads
saved predictions only; does not modify generated actions or simulator logs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from kimodo_vla.model.features import KimodoFeatures, G1_XML_PATH
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.motion.g1_reference import _interpolate_motion, rot6d_row_to_matrix
from kimodo_vla.oracle.geometry import Frame, XmlFK
from summarize_oracle import stats


def audit(run):
    read = lambda name: [json.loads(x) for x in (run/name).read_text().splitlines()]
    plans, execution = read('plans.jsonl'), read('execution.jsonl')
    if any(p.get('variant','original') != 'original' for p in plans):
        raise ValueError('Processed reference: use scripts/summarize_continuity.py for time-aligned errors')
    target=json.loads((run/'target.json').read_text())
    frame=Frame(target['origin'],np.array(target['heading']))
    goal=np.array(target['targets_world'])[:,0]
    features=KimodoFeatures(50)
    fk=XmlFK(G1_XML_PATH,G1_JOINT_NAMES)
    generated={}
    for p in plans:
        data=np.load(run/f"plan-{p['plan']:03d}.npz")
        decoded=features.representation.inverse(torch.from_numpy(data['motion_features'])[None],
                                               is_normalized=False)
        # All requests are on a 0.5-second control grid after the 0.1-second warmup.
        # Thus the first sent command is future t=0.02, not t=0.
        step=round((p['time']-execution[0]['before']['time'])*50)
        offset=step*.6-p['cut']
        _,root=_interpolate_motion(decoded['local_rot_mats'][0],decoded['root_positions'][0],
                                  offset+torch.arange(1,26)*.6)
        actions=data['actions']
        rotations=rot6d_row_to_matrix(torch.from_numpy(actions[:,3:9])).numpy()
        roots=frame.world(root.numpy())
        np.testing.assert_allclose(roots[:,2],actions[:,2],atol=2e-6,
                                   err_msg='Reconstructed command time grid differs')
        wrists=np.array([fk(a[9:38],rp,frame.heading@rr)[0]
                         for a,rp,rr in zip(actions,roots,rotations)])
        generated[p['plan']]=(roots,wrists)
    rows=[]
    for e in execution:
        available=[p for p in plans if p['time']<=e['before']['time']+1e-7]
        if not available: continue
        p=available[-1]
        i=round((e['before']['time']-p['time'])*50)
        if not 0<=i<25: raise ValueError('Execution cannot be aligned with action chunk')
        roots,wrists=generated[p['plan']]
        ref=np.array(e['reference_wrist_world']); actual=np.array(e['after']['wrist_pos'])
        rows.append({'time':e['after']['time'],'plan':p['plan'],'action_index':i,
            'after_arrival':e['after_arrival'],
            'generated_wrist_target_m':np.linalg.norm(wrists[i]-goal,axis=-1).tolist(),
            'transport_wrist_m':np.linalg.norm(ref-wrists[i],axis=-1).tolist(),
            'transport_root_m':float(np.linalg.norm(np.array(e['reference_root_world'])-roots[i])),
            'tracking_wrist_m':np.linalg.norm(actual-ref,axis=-1).tolist(),
            'actual_target_m':np.linalg.norm(actual-goal,axis=-1).tolist(),
            'transport_delta_world':(ref-wrists[i]).tolist()})
    (run/'aligned-stages.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    keys=('generated_wrist_target_m','transport_wrist_m','transport_root_m',
          'tracking_wrist_m','actual_target_m')
    report={group:{k:stats(subset,k) for k in keys} for group,subset in
            [('all_generated_actions',rows),('after_arrival',[r for r in rows if r['after_arrival']])]}
    report['units']='metres; errors are vector norms and must not be added as scalars'
    (run/'aligned-summary.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    print(json.dumps(audit(p.parse_args().run),indent=2))
