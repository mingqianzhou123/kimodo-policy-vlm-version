import numpy as np
import torch
from scipy.spatial.transform import Rotation
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.model.features import KimodoFeatures,G1_XML_PATH
from kimodo_vla.model.sparse_conditions import hand_keyframe
from kimodo_vla.oracle.geometry import XmlFK
from scripts.diagnose_joint_conversion import project_axis,angle


def test_legal_joint_poses_roundtrip_including_limit_edges():
    f=KimodoFeatures(50);limits=XmlFK(G1_XML_PATH,G1_JOINT_NAMES).joint_limits
    rng=np.random.default_rng(42)
    q=np.vstack((limits[:,0],limits[:,1],rng.uniform(limits[:,0],limits[:,1],(64,29)))).astype('float32')
    pose=f.codec.decode_joint_configuration_pose(q,np.tile([0.,0.,.8],(len(q),1)),
        root_rotation_matrices=np.tile(np.eye(3),(len(q),1,1)),joint_names=G1_JOINT_NAMES)
    result=f.codec.encode(pose['local_rot_mats'],pose['root_positions'])[:,9:38].numpy()
    np.testing.assert_allclose(result,q,atol=3e-6)


def test_sparse_values_equal_independent_full_pose_features():
    f=KimodoFeatures(50);rep=f.representation
    q=np.zeros((5,29));q[:,G1_JOINT_NAMES.index('left_shoulder_pitch_joint')]=-.3
    pose=f.codec.decode_joint_configuration_pose(q,np.tile([.4,-.2,.8],(5,1)),
        root_rotation_matrices=np.tile(np.eye(3),(5,1,1)),joint_names=G1_JOINT_NAMES)
    full=f.features(pose['local_rot_mats'],pose['root_positions'])
    gr,pos,_=rep.skeleton.fk(pose['local_rot_mats'],pose['root_positions'])
    positions={};rotations={}
    for side in ('left','right'):
        ids=[rep.skeleton.bone_order_names.index(f'{side}_{j}_skel') for j in ('wrist_yaw','hand_roll')]
        positions[side]=pos[2,ids];rotations[side]=gr[2,ids[0]]
    values,mask=hand_keyframe(rep,frames=5,frame=2,smooth_root_xz=full[2,[0,2]],
        hand_positions=positions,wrist_rotations=rotations)
    assert mask.sum()==26
    torch.testing.assert_close(values[mask],full[mask],atol=1e-6,rtol=1e-6)


def test_axis_projection_matches_brute_force_and_preserves_legal_angles():
    axis=np.array([0.,1.,0.]);r=Rotation.from_euler('xyz',[[.4,1.1,-.3],[0.,.2,0.]]).as_matrix()
    near,bounded=project_axis(r,axis,[-.5,.5])
    assert abs(near[1]-.2)<1e-12 and abs(bounded[1]-.2)<1e-12
    grid=np.linspace(-.5,.5,20001);pr=Rotation.from_rotvec(grid[:,None]*axis).as_matrix()
    brute=angle(np.swapaxes(pr,-1,-2)@r[0]);chosen=Rotation.from_rotvec(axis*bounded[0]).as_matrix()
    assert angle(chosen.T@r[0]) <= brute.min()+1e-6


def test_scalar_branch_jump_is_not_a_large_matrix_rotation():
    axis=np.array([0.,0.,1.]);r=Rotation.from_rotvec(np.deg2rad([179.,181.])[:,None]*axis).as_matrix()
    q,_=project_axis(r,axis,[-3.,3.])
    assert abs(np.rad2deg(np.diff(q)[0]))>350
    assert abs(angle(r[0].T@r[1])-2.)<1e-6
