import json
from pathlib import Path
import numpy as np
from arena.dynamics_override import intervene,FIELDS
from kimodo_vla.core.types import G1_JOINT_NAMES

ROOT=Path(__file__).resolve().parents[1]

def test_combined_profile_locks_aligned_actuators_and_only_changes_sensor_load():
    c=json.loads((ROOT/'runs/dynamics-actuator-20261119-r1-base-center/actuator-config.json').read_text())
    spec=json.loads((ROOT/'configs/sonic-dynamics.json').read_text())
    before=c['before_first_control'];s={k:np.array(before[k]) for k in FIELDS}
    a=intervene(s,c['joint_names'],c['body_names'],'actuator',spec)
    combined=intervene(s,c['joint_names'],c['body_names'],'actuator-load',spec)
    for k in FIELDS:
        if k not in ('masses','inertias'):np.testing.assert_array_equal(combined[k],a[k])
    ids=[c['body_names'].index(n) for n in spec['sensor-load']['bodies']]
    other=[i for i in range(len(c['body_names'])) if i not in ids]
    for k in ('masses','inertias'):np.testing.assert_array_equal(combined[k][other],a[k][other])
    np.testing.assert_allclose(combined['masses'][ids],.001)
    for k in spec['actuator']:np.testing.assert_array_equal(a[k].astype(np.float32),np.array(c['active_first_physics_step'][k],dtype=np.float32))

def test_support_height_comes_from_reference_not_tracking_error():
    from kimodo_vla.oracle.recorded_reference import read_rows
    spec=json.loads((ROOT/'configs/root-dynamics.json').read_text())
    rows=read_rows(ROOT/'runs/dynamics-actuator-20261119-r1-base-center/execution.jsonl')
    feet=np.array(rows[0]['reference_foot_world'])
    assert abs(spec['support_height_m']-(feet[:,2].mean()+spec['sole_min_z_m']))<1e-12
    assert 0.02<spec['support_height_m']<0.022

def test_root_foot_identity_separates_support_from_posture_and_rotation():
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from analyze_root_dynamics import decompose
    from scipy.spatial.transform import Rotation
    r=np.array([.01,-.02,-.024]);rr=np.zeros(3)
    lr=np.array([[.02,.1,-.75],[.02,-.1,-.75]])
    la=lr+np.array([.003,-.004,.005]);R=Rotation.from_euler('xyz',[.02,.03,.01]).as_matrix();Rr=np.eye(3)
    residual=np.array([[1e-5,2e-5,0],[-1e-5,0,2e-5]])
    f=r+la@R.T+residual;fr=rr+lr
    p=decompose(r,rr,f,fr,la,lr,R,Rr)
    np.testing.assert_allclose(sum(p[k] for k in p if k!='root_error'),np.broadcast_to(r,(2,3)),atol=1e-12)
    np.testing.assert_allclose(p['negative_fk_residual'],-residual,atol=1e-12)
