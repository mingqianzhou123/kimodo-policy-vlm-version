import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from kimodo_vla.vlm.geometry import parse_points,backproject,load_capture,MOTION_TEXT


def test_point_contract_rejects_hidden_targets_and_extra_commands():
    good={'left':[250,500],'right':[750,500],'motion_text':MOTION_TEXT}
    assert parse_points(json.dumps(good))==good
    for change in ({'left':None},{'left':[float('nan'),2]},{'left':[True,2]},
                   {'left':[1001,0]},{'joint_angles':[0]*29},{'motion_text':'close grippers'}):
        with pytest.raises(ValueError):parse_points(json.dumps({**good,**change}))


def test_backproject_optical_z_with_rotated_camera_and_image_edges():
    K=np.array([[100.,0,100],[0,100,50],[0,0,1]])
    R=Rotation.from_euler('z',90,degrees=True).as_matrix();p=np.array([1,2,3.])
    depth=np.full((101,201),2.)
    world,info=backproject([1000,1000],depth,K,p,R)
    np.testing.assert_allclose(world,[0,4,5],atol=1e-12)
    assert info['pixel']==[200,100]
    np.testing.assert_allclose(backproject([500,500],depth,K,p,R)[0],[1,2,5])


@pytest.mark.parametrize('bad',[np.nan,np.inf,0.,10.])
def test_bad_depth_is_rejected_without_background_fallback(bad):
    depth=np.full((5,5),1.);depth[2,2]=bad
    with pytest.raises(ValueError):backproject([500,500],depth,np.eye(3),np.zeros(3),np.eye(3))


def test_depth_edge_is_rejected():
    depth=np.full((5,5),1.);depth[2,3]=2.
    with pytest.raises(ValueError):backproject([500,500],depth,np.eye(3),np.zeros(3),np.eye(3))


def test_capture_rejects_ray_distance(tmp_path):
    p=tmp_path/'capture.npz'
    np.savez(p,rgb=np.zeros((5,5,3),dtype=np.uint8),depth=np.ones((5,5)),
             depth_kind='distance_to_camera',camera_convention='ros_optical')
    with pytest.raises(ValueError):load_capture(p)


def test_visual_path_uses_cartesian_goal_not_witness_joints():
    from kimodo_vla.vlm.policy import VisualTargetPolicy
    from kimodo_vla.oracle.geometry import Frame
    policy=VisualTargetPolicy.__new__(VisualTargetPolicy)
    policy.frame=Frame([1.,2.,3.],Rotation.from_euler('z',.4).as_matrix())
    policy.initial_visual_targets=np.arange(12,dtype=float).reshape(2,2,3)/100
    policy.targets=policy.initial_visual_targets+np.array([.05,0,.01])
    policy.target_rot_model=np.tile(np.eye(3),(2,1,1))
    pos,rot=policy.path_goals(np.array([0.,.6,1.2,8.]))
    world=policy.frame.world(pos)
    np.testing.assert_allclose(world[0],policy.initial_visual_targets,atol=1e-12)
    np.testing.assert_allclose(world[1],(policy.initial_visual_targets+policy.targets)/2,atol=1e-12)
    np.testing.assert_allclose(world[2:],np.tile(policy.targets,(2,1,1,1)),atol=1e-12)
    np.testing.assert_array_equal(rot[0],rot[-1])


def test_paused_capture_refreshes_sensor_without_advancing_physics(tmp_path):
    from types import SimpleNamespace
    import torch
    from arena.vlm_capture import capture
    rendered=[0]
    class Camera:
        def __init__(self):
            self.cfg=SimpleNamespace(update_latest_camera_pose=False)
            self.outdated=False
            self.data=SimpleNamespace(output={'rgb':torch.zeros(1,5,5,3,dtype=torch.uint8),
                'distance_to_image_plane':torch.ones(1,5,5,1)},
                intrinsic_matrices=torch.eye(3)[None],pos_w=torch.zeros(1,3),
                quat_w_ros=torch.tensor([[1.,0.,0.,0.]]))
        def reset(self):self.outdated=True
        def update(self,dt,force_recompute=False):
            # IsaacLab only updates outdated buffers, even with force_recompute.
            if self.outdated:
                self.data.output['rgb'].fill_(rendered[0]);self.outdated=False
    camera=Camera()
    sim=SimpleNamespace(current_time=1.25,forward=lambda:None,
        render=lambda:rendered.__setitem__(0,rendered[0]+1))
    result=capture(SimpleNamespace(scene={'front_camera':camera},sim=sim),tmp_path)
    data=load_capture(result['capture_path'])
    assert rendered[0]>0 and np.all(data['rgb']==rendered[0])
    assert float(data['time'])==sim.current_time==1.25
