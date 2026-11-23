import numpy as np
import torch
from scipy.spatial.transform import Rotation
from kimodo_vla.model.features import KimodoFeatures,G1_XML_PATH
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.oracle.geometry import XmlFK
from kimodo_vla.oracle.continuity import condition_boundary,blend_reference,bounded_hinge_trajectory


def pose(f,q):
    return f.codec.decode_joint_configuration_pose(q,np.tile([0.,0.,.8],(len(q),1)),
        root_rotation_matrices=np.tile(np.eye(3),(len(q),1,1)),joint_names=G1_JOINT_NAMES)


def test_handoff_matches_consumed_reference_at_zero_and_prediction_after_interval():
    f=KimodoFeatures(50);a=pose(f,np.zeros((1,29)));b=pose(f,np.ones((4,29))*.3)
    r,p=blend_reference(b['local_rot_mats'],b['root_positions']+.1,a['local_rot_mats'][0],
        a['root_positions'][0],[0.,.02,.2,.3],.2)
    torch.testing.assert_close(r[0],a['local_rot_mats'][0],atol=2e-6,rtol=0)
    torch.testing.assert_close(p[0],a['root_positions'][0])
    torch.testing.assert_close(r[2:],b['local_rot_mats'][2:],atol=2e-6,rtol=0)
    torch.testing.assert_close(p[2:],b['root_positions'][2:]+.1)
    assert torch.max(abs(r[1]-r[0]))<.005


def test_conditions_keep_real_last_history_and_current_pose_not_expired_goal():
    f=KimodoFeatures(50);history=torch.ones(48,417,dtype=torch.bool)
    values=torch.ones(50,417);mask=torch.ones_like(values,dtype=torch.bool)
    current=torch.arange(417,dtype=torch.float32)
    condition_boundary(history,values,mask,current,f.representation)
    assert not history[:-1].any() and history[-1].all()
    end=f.representation.slice_dict['global_rot_data'].stop
    torch.testing.assert_close(values[0,:end],current[:end])
    torch.testing.assert_close(values[15],torch.ones(417))


def test_bounded_projection_handles_pi_and_limits_across_chunk_seam():
    f=KimodoFeatures(50);lim=XmlFK(G1_XML_PATH,G1_JOINT_NAMES).joint_limits
    q=np.zeros((40,29));j=G1_JOINT_NAMES.index('right_hip_yaw_joint')
    q[:,j]=np.deg2rad(np.linspace(175,185,40));q[:,17]=1.5
    raw=pose(f,q)['local_rot_mats'];prev=np.zeros(29);prev[j]=2.5
    first=bounded_hinge_trajectory(raw[:25],f.codec,lim,prev,2.)
    second=bounded_hinge_trajectory(raw[25:],f.codec,lim,first[-1],2.)
    out=np.r_[first,second]
    assert np.all(out>=lim[:,0]-1e-7) and np.all(out<=lim[:,1]+1e-7)
    assert np.max(abs(np.diff(np.vstack([prev,out]),axis=0)))<=.0400001
    assert np.all(out[:,j]>2.4)  # no +/-pi branch teleport to the opposite limit


def test_bounded_projection_preserves_slow_legal_reference():
    f=KimodoFeatures(50);lim=XmlFK(G1_XML_PATH,G1_JOINT_NAMES).joint_limits
    q=np.zeros((25,29));q[:,11]=np.linspace(.001,.025,25)
    result=bounded_hinge_trajectory(pose(f,q)['local_rot_mats'],f.codec,lim,np.zeros(29),2.)
    np.testing.assert_allclose(result,q,atol=2e-6)
