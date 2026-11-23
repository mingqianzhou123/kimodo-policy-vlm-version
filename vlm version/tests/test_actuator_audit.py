import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from kimodo_vla.core.types import G1_JOINT_NAMES

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_actuator_audit import pd_estimate,gravity_from_links


def test_static_gravity_balance_and_stale_gain_estimate():
    # Two N.m gravity compensation with a 20 N.m/rad drive needs -0.1 rad error.
    q=np.array([-.1]);target=np.array([0.]);zero=np.zeros(1)
    actual=pd_estimate(q,zero,target,zero,zero,np.array([20.]),np.array([1.]))
    cached=pd_estimate(q,zero,target,zero,zero,np.array([40.]),np.array([5.]))
    np.testing.assert_allclose(actual,[2.]);np.testing.assert_allclose(cached,[4.])
    np.testing.assert_allclose(q-target+actual/20,0,atol=1e-12)


def test_gravity_uses_descendant_mass_world_axis_and_com():
    # Independent disconnected pendulums allow auditing all 29 canonical names.
    names=[f'b{i}' for i in range(29)]
    cfg=dict(body_names=names,sim_cfg={'gravity':[0,0,-10]},
        active_first_physics_step={'masses':[2.]*29,'coms':[[.3,0,0,0,0,0,1]]*29})
    joints=[dict(name=n,body0=['/base'],body1=['/'+names[i]],attributes={
        'physics:axis':'Y','physics:localRot1':[1,0,0,0],'physics:localPos1':'(0, 0, 0)'})
        for i,n in enumerate(G1_JOINT_NAMES)]
    poses=np.zeros((29,7));poses[:,6]=1
    np.testing.assert_allclose(gravity_from_links(cfg,joints,poses),-6)
    # Tilt every link 90 degrees: COM lies above/below the joint and gravity moment vanishes.
    poses[:,3:7]=Rotation.from_euler('y',90,degrees=True).as_quat()
    np.testing.assert_allclose(gravity_from_links(cfg,joints,poses),0,atol=1e-12)


def test_fixed_payload_enters_parent_gravity_moment():
    names=[f'b{i}' for i in range(29)]+['payload']
    cfg=dict(body_names=names,sim_cfg={'gravity':[0,0,-10]},active_first_physics_step={
        'masses':[0.]*29+[3.],'coms':[[0,0,0,0,0,0,1]]*30})
    joints=[dict(name=n,body0=['/base'],body1=['/'+names[i]],attributes={
        'physics:axis':'Y','physics:localRot1':[1,0,0,0],'physics:localPos1':[0,0,0]})
        for i,n in enumerate(G1_JOINT_NAMES)]
    joints.append(dict(name='fixed_payload',body0=['/b0'],body1=['/payload'],attributes={}))
    poses=np.zeros((30,7));poses[:,6]=1;poses[-1,0]=.5
    result=gravity_from_links(cfg,joints,poses)
    np.testing.assert_allclose(result[0],-15);np.testing.assert_allclose(result[1:],0)
