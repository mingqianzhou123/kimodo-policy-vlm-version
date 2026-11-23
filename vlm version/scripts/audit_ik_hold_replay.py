"""Verify default IK numerics against preserved implementation on saved inputs."""
import importlib.util,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from kimodo_vla.oracle.policy import OraclePolicy
from kimodo_vla.oracle.projection import project_trajectory
from kimodo_vla.oracle.geometry import XmlFK,Frame,rotation_wxyz,BASIS
from kimodo_vla.model.features import KimodoFeatures,G1_XML_PATH
from kimodo_vla.core.types import G1_JOINT_NAMES


def audit():
    original=Path('snapshots/ik-before-hold-stability-20261117/kimodo_vla/oracle/projection.py')
    spec=importlib.util.spec_from_file_location('old_projection',original);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[];roundoff=[]
    for mode in ('hold','reach'):
        run=Path('runs')/f'continuous-ik-{mode}-20261117-r1'
        read=lambda f:[json.loads(x) for x in (run/f).read_text().splitlines()]
        target=json.loads((run/'target.json').read_text());initial=read('execution.jsonl')[0]['before']
        p=OraclePolicy.__new__(OraclePolicy);p.features=KimodoFeatures(50);p.frame=Frame(target['origin'],np.array(target['heading']))
        p.initial_root=np.array(initial['root_pos']);p.initial_rotation=rotation_wxyz(initial['root_quat']);p.initial_joints=np.array(initial['joints'])
        p.goal_joints=p.initial_joints if target['witness_joint_pos'] is None else np.array(target['witness_joint_pos'])
        names=p.features.representation.skeleton.bone_order_names
        p.ids=[[names.index(f'{s}_{j}_skel') for j in ('wrist_yaw','hand_roll')] for s in ('left','right')]
        one,one_rot=p.path_goals(np.array([2.]))
        batch,batch_rot=p.path_goals(2.+np.arange(1,26)/50)
        a=p.frame.world(one[0,:,0]);b=p.frame.world(batch[0,:,0])
        ar=p.frame.world_to_model.T@one_rot[0]@BASIS
        br=(p.frame.world_to_model.T@batch_rot@BASIS)[0]
        roundoff.append({'mode':mode,'same_goal_single_vs_batch_position_m':np.linalg.norm(a-b,axis=-1).tolist(),
            'same_goal_rotation_matrix_max_delta':float(abs(ar-br).max()),
            'same_goal_rotation_rad':np.linalg.norm(Rotation.from_matrix(np.swapaxes(ar,-1,-2)@br).as_rotvec(),axis=-1).tolist()})
        fk=XmlFK(G1_XML_PATH,G1_JOINT_NAMES);support=target['support_target'];commands=read('commands.jsonl');plans=read('plans.jsonl')
        for index in (0,3,15):
            plan=plans[index];data=np.load(run/f'plan-{index:03d}.npz');c=next(x for x in commands if x['plan']==index)
            pos,rot=p.path_goals(plan['time']-initial['time']+np.arange(1,26)/50)
            args=(fk,data['pre_ik_actions'][:,9:38],c['boundary_joints'],p.initial_root,p.initial_rotation,
                  p.frame.world(pos[:,:,0]),p.frame.world_to_model.T@rot@BASIS,
                  np.array(support['ankle_world']),np.array(support['ankle_rotation_world']))
            before,_=module.project_trajectory(*args);after,_=project_trajectory(*args)
            np.testing.assert_array_equal(before,after)
            rows.append({'mode':mode,'plan':index,'reference_joint_max_delta_rad':float(abs(before-after).max())})
    out=Path('runs/ik-hold-review');out.mkdir(exist_ok=True)
    (out/'default-ik-replay.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
    (out/'goal-roundoff.json').write_text(json.dumps(roundoff,indent=2));print(json.dumps(roundoff))

if __name__=='__main__':audit()
