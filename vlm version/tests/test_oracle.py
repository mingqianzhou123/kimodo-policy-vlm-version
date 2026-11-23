import numpy as np
import torch
from scipy.spatial.transform import Rotation
from test_v61_interfaces import make_policy, infer
from kimodo_vla.oracle.geometry import Frame, XmlFK
from kimodo_vla.model.features import KimodoFeatures, G1_XML_PATH
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.model.native_policy import ConstraintGuidance
from kimodo_vla.oracle.policy import OraclePolicy


def test_constraint_guidance_removes_text_and_only_changes_condition_mask():
    class Probe(torch.nn.Module):
        def forward(self, **kw):
            assert not kw['text_feat'].any()
            assert not kw['text_feat_pad_mask'].any()
            return kw['x'] + kw['motion_mask'] * kw['observed_motion']
    g = ConstraintGuidance(Probe(),2.)
    result = g(x=torch.tensor([3.,4.]), motion_mask=torch.tensor([1.,0.]),
        observed_motion=torch.tensor([5.,7.]),text_feat=torch.ones(1,1,4),
        text_feat_pad_mask=torch.ones(1,1,dtype=torch.bool))
    torch.testing.assert_close(result,torch.tensor([13.,4.]))


def test_native_sampling_never_encodes_images_or_requires_controlnet():
    model = make_policy()
    model.controlnet = None
    def fail(*args):
        raise AssertionError('Native A must not encode an image')
    model._encode_images = fail
    out = infer(model)
    assert torch.isfinite(out['motion_features']).all()
    assert len(model.denoiser.calls) == 2


def test_native_cfg_keeps_noisy_history_outside_condition_branch():
    model=make_policy()
    model.controlnet=None
    model.native_noisy_history=True
    infer(model)
    call=model.denoiser.calls[0]
    assert torch.all(call['observed_motion'][0,1:3] == -2)
    assert not torch.all(call['x'][0,1:3] == -2)


def test_coordinate_roundtrip_with_nonzero_origin_and_heading():
    h = Rotation.from_euler('z', 1.1).as_matrix()
    f = Frame([12.,-5.,.8], h)
    p = np.array([[12.3,-4.8,1.0], [11.8,-4.7,.9]])
    np.testing.assert_allclose(f.world(f.points(p)),p,atol=1e-12)
    assert f.origin[2] == 0
    np.testing.assert_allclose(f.points(p)[:,1], p[:,2])


def test_independent_xml_fk_agrees_with_codec_positions_and_projection():
    features = KimodoFeatures(50)
    fk = XmlFK(G1_XML_PATH, G1_JOINT_NAMES)
    q = np.linspace(-.12,.18,29).astype(np.float32)
    root = np.array([.4,-.2,.85],dtype=np.float32)
    r = Rotation.from_euler('xyz',[.03,-.04,.7]).as_matrix().astype(np.float32)
    pose = features.codec.decode_joint_configuration_pose(q[None],root[None],
        root_rotation_matrices=r[None],joint_names=G1_JOINT_NAMES)
    gr,pos,_ = features.representation.skeleton.fk(pose['local_rot_mats'],pose['root_positions'])
    ids = [features.representation.skeleton.bone_order_names.index(f'{s}_wrist_yaw_skel') for s in ('left','right')]
    skel_world = pos[0,ids].numpy() @ features.codec.mujoco_to_kimodo.numpy()
    xml_pos,_ = fk(q,root,r)
    np.testing.assert_allclose(skel_world,xml_pos,atol=2e-5)
    action = features.codec.encode(pose['local_rot_mats'],pose['root_positions'],
        previous_root_position=pose['root_positions'][0],hand_binary=torch.zeros(1,2)).numpy()
    np.testing.assert_allclose(action[0,9:38],q,atol=2e-5)


def test_reach_goal_has_an_explicit_joint_limited_fk_witness(tmp_path):
    policy=OraclePolicy.__new__(OraclePolicy)
    policy.features=KimodoFeatures(50)
    policy.fk=XmlFK(G1_XML_PATH,G1_JOINT_NAMES)
    policy.ids=[[policy.features.representation.skeleton.bone_order_names.index(f'{s}_{j}_skel')
                 for j in ('wrist_yaw','hand_roll')] for s in ('left','right')]
    policy.mode,policy.run='reach',tmp_path
    q=np.zeros(29);root=np.array([.3,-.4,.8]);r=np.eye(3)
    wrist,wr=policy.fk(q,root,r)
    policy.initialize({'root_pos':root.tolist(),'root_quat':[1,0,0,0],'time':0.,
                       'joints':q.tolist(),'wrist_pos':wrist.tolist(),'wrist_rot':wr.tolist()})
    witness=np.array(policy.target_info['witness_joint_pos'])
    assert np.all(witness>=policy.fk.joint_limits[:,0])
    assert np.all(witness<=policy.fk.joint_limits[:,1])
    pos,rot=policy.fk(witness,root,r)
    np.testing.assert_allclose(policy.targets[:,0],pos,atol=1e-7)
    np.testing.assert_allclose(policy.target_rot_world,rot,atol=1e-7)
    assert np.all(np.linalg.norm(pos-wrist,axis=-1)>.01)
