import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from kimodo_vla.oracle.fixed_reference import FixedReference
from kimodo_vla.core.types import G1_JOINT_NAMES


def sample(policy, time=0.):
    q=np.zeros(29);r=Rotation.from_euler('xyz',[.04,-.03,.8]).as_matrix()
    root=np.array([1.,-2.,.8]); wp,wr=policy.fk(q,root,r)
    return {'time':time,'joints':q.tolist(),'root_pos':root.tolist(),
            'reference_root_pos':root.tolist(),'root_quat':Rotation.from_matrix(r).as_quat()[[3,0,1,2]].tolist(),
            'wrist_pos':wp.tolist(),'wrist_rot':wr.tolist()}


def test_fixed_hold_has_no_generator_and_never_chases_measured_drift(tmp_path):
    p=FixedReference(tmp_path,'hold');s=sample(p)
    first=p({'observation':{'oracle_history':[s]}})
    s['time']=.5;s['joints']=(np.zeros(29)+.1).tolist();s['root_pos'][0]+=.3
    second=p({'observation':{'oracle_history':[s]}})
    np.testing.assert_allclose(first['action_chunk'],second['action_chunk'],atol=1e-12)
    assert not hasattr(p,'model')
    assert all(x['model_loaded'] is False for x in map(json.loads,(tmp_path/'plans.jsonl').read_text().splitlines()))


def test_fixed_reach_is_smooth_bounded_and_has_exact_fk_goal(tmp_path):
    p=FixedReference(tmp_path,'reach');s=sample(p);p.initialize(s)
    t=np.linspace(0,2,201);q=p.joints_at(t)
    assert np.all(q>=p.fk.joint_limits[:,0]) and np.all(q<=p.fk.joint_limits[:,1])
    np.testing.assert_allclose(q[0],p.initial)
    np.testing.assert_allclose(q[t>=1.2],np.broadcast_to(p.goal,q[t>=1.2].shape))
    pos,rot=p.fk(p.goal,p.root,p.rotation)
    np.testing.assert_allclose(pos,np.array(p.target['targets_world'])[:,0])
    np.testing.assert_allclose(rot,p.target['wrist_rotation_world'])
    assert np.max(np.abs((p.joints_at(1e-4)-p.initial)/1e-4))<1e-6
    assert np.max(np.abs((p.goal-p.joints_at(1.2-1e-4))/1e-4))<1e-6


def test_chunk_timing_and_nonzero_heading_delta_reconstruct_absolute_reference(tmp_path):
    p=FixedReference(tmp_path,'reach');s=sample(p)
    s['reference_root_pos']=[.97,-2.02,.79]
    response=p({'observation':{'oracle_history':[s]}})
    a=np.array(response['action_chunk'])
    delta_local=p.rotation.T@(p.root-np.array(s['reference_root_pos']))
    np.testing.assert_allclose(a[0,:2],delta_local[:2])
    np.testing.assert_allclose(a[1:,:2],0.)
    np.testing.assert_allclose(a[:,9:38],p.joints_at(np.arange(1,26)/50))
    s['time']=.5;s['reference_root_pos']=p.root.tolist()
    b=np.array(p({'observation':{'oracle_history':[s]}})['action_chunk'])
    np.testing.assert_allclose(b[:,9:38],p.joints_at(np.arange(26,51)/50))
