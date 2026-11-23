"""Check a kinematic reachability witness; never use IK to correct policy output."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from kimodo_vla.oracle.geometry import XmlFK,rotation_wxyz,angle_deg
from kimodo_vla.model.features import G1_XML_PATH
from kimodo_vla.core.types import G1_JOINT_NAMES


def verify(run):
    sample=json.loads((run/'execution.jsonl').read_text().splitlines()[0])['before']
    target=json.loads((run/'target.json').read_text())
    goal=np.array(target['targets_world'])[:,0]; goal_rot=np.array(target['wrist_rotation_world'])
    fk=XmlFK(G1_XML_PATH,G1_JOINT_NAMES)
    q=np.array(sample['joints']); root=np.array(sample['root_pos']); rot=rotation_wxyz(sample['root_quat'])
    ids=[i for i,n in enumerate(G1_JOINT_NAMES) if any(x in n for x in ('shoulder','elbow','wrist'))]
    def residual(arm):
        candidate=q.copy(); candidate[ids]=arm
        pos,r=fk(candidate,root,rot)
        angle=Rotation.from_matrix(np.swapaxes(goal_rot,-1,-2)@r).as_rotvec()
        return np.r_[(pos-goal).ravel(),(.1*angle).ravel()]
    initial = np.array(target['witness_joint_pos'])[ids] if target.get('witness_joint_pos') else q[ids]
    fit=least_squares(residual,initial,bounds=(fk.joint_limits[ids,0],fk.joint_limits[ids,1]),
                      ftol=1e-12,gtol=1e-12,xtol=1e-12,max_nfev=500)
    q[ids]=fit.x;pos,r=fk(q,root,rot)
    error=np.linalg.norm(pos-goal,axis=-1)
    angles=[angle_deg(x,y) for x,y in zip(r,goal_rot)]
    result={'wrist_error_m':error.tolist(),'wrist_angle_deg':angles,
        'kinematic_witness_found':bool(np.all(error<1e-4) and max(angles)<.1),
        'within_joint_limits':bool(np.all(q>=fk.joint_limits[:,0]-1e-6) and np.all(q<=fk.joint_limits[:,1]+1e-6)),
        'joint_names':G1_JOINT_NAMES,'witness_joint_pos':q.tolist(),
        'root_pos':root.tolist(),'root_quat':sample['root_quat'],
        'collision_checked':False,'dynamic_feasibility_checked':False,'applied_to_policy':False}
    (run/'reachability.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    print(json.dumps(verify(p.parse_args().run),indent=2))
