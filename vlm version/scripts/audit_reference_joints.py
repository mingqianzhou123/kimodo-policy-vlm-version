"""Audit transmitted joint references, including chunk seams and XML limits.

Raw scalar differences are intentional: this joint29 interface differentiates
scalar joint angles, without unwrapping them. These are commanded rates, not
measured robot velocities or a dynamic-feasibility certificate.
"""
import argparse,json
from pathlib import Path
import numpy as np
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.oracle.geometry import XmlFK


def audit(run,steps=200):
    if (run/'commands.jsonl').exists():
        rows=[json.loads(x) for x in (run/'commands.jsonl').read_text().splitlines()]
        q=np.asarray([r['joints'] for r in rows])[:steps]
        plans=np.asarray([r['plan'] for r in rows])[:steps]
    else:
        paths=sorted(run.glob('plan-*.npz'))
        chunks=[np.load(p)['actions'][:,9:38] for p in paths]
        q=np.concatenate(chunks)[:steps]
        plans=np.concatenate([np.full(len(c),i) for i,c in enumerate(chunks)])[:steps]
    fk=XmlFK(Path(__file__).resolve().parents[1]/'kimodo_vla/skeleton/assets/g1skel34/xml/g1.xml',G1_JOINT_NAMES)
    bad=(q<fk.joint_limits[:,0]-1e-5)|(q>fk.joint_limits[:,1]+1e-5)
    differences=np.abs(np.diff(q,axis=0));largest=differences.max(axis=1)
    i,j=np.unravel_index(differences.argmax(),differences.shape)
    result={'run':run.name,'frames':len(q),'control_hz':50,
        'frames_outside_xml_limits':int(bad.any(axis=1).sum()),
        'violations_by_joint':{n:int(v) for n,v in zip(G1_JOINT_NAMES,bad.sum(axis=0)) if v},
        'max_scalar_step_deg':float(np.rad2deg(largest.max())),
        'p95_max_joint_command_rate_rad_s':float(np.quantile(largest*50,.95)),
        'chunk_seam_max_joint_steps_deg':np.rad2deg(largest[np.diff(plans)!=0]).tolist(),
        'largest_step':{'from_index':int(i),'joint':G1_JOINT_NAMES[j],
            'angles_rad':q[i:i+2,j].tolist()},
        'dynamic_feasibility_claimed':False}
    (run/'joint-reference-audit.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--steps',type=int,default=200);a=p.parse_args()
    print(json.dumps(audit(a.run,a.steps),indent=2))
