"""Controlled visible-target fixture. Truth is evaluator-only, never policy input."""
import json
from pathlib import Path
import numpy as np
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.oracle.geometry import XmlFK,rotation_wxyz


def create_markers(env,run,camera_name,reach_rad=.15,camera_lateral_m=0.):
    from pxr import UsdGeom,Gf
    robot=env.scene['robot'];data=robot.data
    ids=[list(robot.joint_names).index(n) for n in G1_JOINT_NAMES]
    q=data.joint_pos[0,ids].detach().cpu().numpy().copy()
    for side in ('left','right'):q[G1_JOINT_NAMES.index(f'{side}_shoulder_pitch_joint')]-=reach_rad
    root=data.root_state_w[0].detach().cpu().numpy()
    xml=Path(__file__).resolve().parents[2]/'kimodo_vla/skeleton/assets/g1skel34/xml/g1.xml'
    fk=XmlFK(xml,G1_JOINT_NAMES)
    if np.any(q<fk.joint_limits[:,0]) or np.any(q>fk.joint_limits[:,1]):raise ValueError('Fixture target is not joint limited')
    goals,_=fk(q,root[:3],rotation_wxyz(root[3:7]))
    camera=env.scene[camera_name];camera.cfg.update_latest_camera_pose=True
    env.sim.forward();camera.reset();camera.update(0.,force_recompute=True)
    quat=camera.data.quat_w_ros[0].detach().cpu().numpy()
    R=rotation_wxyz(quat);stage=env.sim.get_initial_stage()
    for side,goal,color in zip(('left','right'),goals,((1.,0.,0.),(0.,0.,1.))):
        cube=UsdGeom.Cube.Define(stage,f'/World/VLMTarget_{side}');cube.CreateSizeAttr(1.)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        transform=UsdGeom.Xformable(cube);transform.ClearXformOpOrder()
        transform.AddTranslateOp().Set(Gf.Vec3d(*(goal+R[:,2]*.0005).tolist()))
        transform.AddOrientOp().Set(Gf.Quatf(float(quat[0]),Gf.Vec3f(*quat[1:].tolist())))
        transform.AddScaleOp().Set(Gf.Vec3f(.035,.035,.001))
    # Keep marker orientation fixed across views; move only the sensor along its
    # original optical X axis. No robot/root joint is changed.
    original_position=camera.data.pos_w[0].detach().cpu().numpy().copy()
    if camera_lateral_m:
        camera.set_world_poses(positions=(original_position+R[:,0]*camera_lateral_m)[None],
            orientations=quat[None],convention='ros')
    # Only rendering geometry: no collision, mass, physics or ground-truth target injection.
    truth={'targets_world':goals.tolist(),'fixture':'red-left blue-right 35 mm visual squares',
           'camera_name':camera_name,'collision':False,'used_by_policy':False,
           'initial_wrist_world':data.body_pos_w[0,[list(robot.body_names).index(f'{side}_wrist_yaw_link') for side in ('left','right')]].detach().cpu().numpy().tolist(),
           'reach_rad':reach_rad,'camera_lateral_m':camera_lateral_m,
           'original_camera_position':original_position.tolist(),
           'expected_camera_position':(original_position+R[:,0]*camera_lateral_m).tolist()}
    (Path(run)/'visual-evaluation-truth.json').write_text(json.dumps(truth,indent=2))
