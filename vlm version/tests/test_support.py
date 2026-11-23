import numpy as np
import torch
from kimodo_vla.model.features import KimodoFeatures,G1_XML_PATH
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.oracle.geometry import XmlFK,Frame
from kimodo_vla.oracle.support import support_channels,condition_support


def standing(f,q):
    pose=f.codec.decode_joint_configuration_pose(q[None],np.array([[0.,0.,.8]]),
        root_rotation_matrices=np.eye(3)[None],joint_names=G1_JOINT_NAMES)
    return pose,f.features(pose['local_rot_mats'],pose['root_positions'])[0]


def test_support_mask_preserves_boundary_and_hand_goals_and_none_is_noop():
    f=KimodoFeatures(50);_,initial=standing(f,np.zeros(29));rep=f.representation
    for mode in ('none','root','root-feet'):
        v=torch.randn(50,417);m=torch.zeros_like(v,dtype=torch.bool);old=v.clone()
        channels=condition_support(v,m,initial,rep,mode)
        torch.testing.assert_close(v[0],old[0]);assert not m[0].any()
        torch.testing.assert_close(v[:,~channels],old[:,~channels])
        if mode!='none':torch.testing.assert_close(v[1:,channels],initial[channels].expand(49,-1))
        assert int(channels.sum())=={'none':0,'root':17,'root-feet':69}[mode]
        for side in ('left','right'):
            j=rep.skeleton.bone_order_names.index(f'{side}_wrist_yaw_skel')
            i=rep.slice_dict['local_joints_positions'].start+3*j
            assert not channels[i:i+3].any()


def test_reach_witness_is_compatible_with_fixed_root_and_both_feet():
    f=KimodoFeatures(50);q=np.zeros(29);_,initial=standing(f,q)
    for side in ('left','right'):q[G1_JOINT_NAMES.index(f'{side}_shoulder_pitch_joint')]=-.15
    _,goal=standing(f,q);channels=support_channels(f.representation,'root-feet')
    torch.testing.assert_close(initial[channels],goal[channels],atol=2e-6,rtol=0)


def test_ankle_fk_agrees_with_independent_xml_and_world_transform():
    f=KimodoFeatures(50);q=np.linspace(-.2,.2,29);pose,_=standing(f,q)
    _,p,_=f.representation.skeleton.fk(pose['local_rot_mats'],pose['root_positions'])
    ids=[f.representation.skeleton.bone_order_names.index(f'{s}_ankle_roll_skel') for s in ('left','right')]
    fk=XmlFK(G1_XML_PATH,G1_JOINT_NAMES)
    feet,_=fk(q,[0,0,.8],np.eye(3),body_names=['left_ankle_roll_link','right_ankle_roll_link'])
    np.testing.assert_allclose(Frame([0,0,0],np.eye(3)).world(p[0,ids].numpy()),feet,atol=2e-6)


def test_sparse_feet_only_conditions_selected_frames_with_dense_root():
    f=KimodoFeatures(50);_,initial=standing(f,np.zeros(29));rep=f.representation
    v=torch.zeros(50,417);m=torch.zeros_like(v,dtype=torch.bool)
    condition_support(v,m,initial,rep,'root-feet-keyframes',keyframes=[0,15,30,49])
    root=support_channels(rep,'root');foot=support_channels(rep,'root-feet') & ~root
    assert m[1:,root].all() and not m[0].any()
    assert m[:,foot].any(-1).nonzero().flatten().tolist()==[15,30,49]
    assert int(m.sum())==49*17+3*52
