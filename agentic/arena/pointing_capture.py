"""Compose static scenes in the simulator and capture first-person RGB-D frames for the pointing test.

Objects are visual copies outside physics, placed by writing their pose, so every pose is exactly
the one recorded. Truth (poses, meshes, target identity) goes to ``truth/`` and never
enters the files a model reads in ``inputs/``.
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np

PARK = (60., 60., -20.)  # unused copies wait here, far outside every view


def yaw_quat(deg):
    half = np.radians(deg) / 2
    return np.array([np.cos(half), 0., 0., np.sin(half)])


def quat_matrix(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])


def resting_pose(mesh_min_z, x, y, support_z, yaw_deg):
    """Pose that rests the object's lowest point on a horizontal support, turned about the vertical."""
    return np.array([x, y, support_z - mesh_min_z]), yaw_quat(yaw_deg)


def add_instances(scene_cfg, scene, assets):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    root = os.environ['PROJECT_ROOT']
    for i, (name, asset_name) in enumerate(scene['instances'].items()):
        asset = assets[asset_name]
        setattr(scene_cfg, f'pointing_{name}', AssetBaseCfg(
            prim_path=f'/World/envs/env_.*/Pointing_{name}',
            init_state=AssetBaseCfg.InitialStateCfg(pos=(PARK[0]+3*i, PARK[1], PARK[2]), rot=(1., 0., 0., 0.)),
            # Rendering only: these copies take no part in physics.
            spawn=UsdFileCfg(usd_path=f"{root}/{asset['usd']}", scale=(asset['scale'],)*3,
                             rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=False),
                             collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False))))


def extract_mesh(root):
    """Rendered triangles of one object, in metres, in the frame of its root prim (scale applied)."""
    from pxr import Usd, UsdGeom
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    M = np.array(cache.GetLocalToWorldTransform(root))  # row vectors: world = local @ M[:3,:3] + M[3,:3]
    axes = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=1, keepdims=True)
    vertices, faces, offset = [], [], 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        image = UsdGeom.Imageable(prim)
        if image.ComputeVisibility() == UsdGeom.Tokens.invisible or image.ComputePurpose() not in ('default', 'render'):
            continue
        mesh = UsdGeom.Mesh(prim)
        points, counts, index = mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts:
            continue
        P = np.array(cache.GetLocalToWorldTransform(prim))
        world = np.asarray(points, float) @ P[:3, :3] + P[3, :3]
        vertices.append((world - M[3, :3]) @ axes.T)
        index, start = np.asarray(index), 0
        for count in counts:
            for k in range(1, count-1):
                faces.append((offset+index[start], offset+index[start+k], offset+index[start+k+1]))
            start += count
        offset += len(points)
    if not faces:
        raise ValueError(f'No rendered mesh under {root.GetPath()}')
    return np.concatenate(vertices).astype(np.float32), np.asarray(faces, np.int32)


def set_pose(prim, position, quat):
    from pxr import Gf, Sdf
    prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*map(float, position)))
    orient = prim.GetAttribute('xformOp:orient')
    w, xyz = float(quat[0]), [float(v) for v in quat[1:]]
    orient.Set(Gf.Quatd(w, Gf.Vec3d(*xyz)) if orient.GetTypeName() == Sdf.ValueTypeNames.Quatd else Gf.Quatf(w, Gf.Vec3f(*xyz)))


def place_robot(env, x, y, yaw_deg):
    import torch
    robot = env.scene['robot']
    root = robot.data.default_root_state.clone()
    root[:, :2] = torch.tensor([x, y], device=root.device)
    root[:, 3:7] = torch.tensor(yaw_quat(yaw_deg), device=root.device, dtype=root.dtype)
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
    return root[0, :7].detach().cpu().numpy()


def probe(env, stage, spec, out):
    """Layout facts needed to write frame specifications: furniture extents and asset sizes."""
    from PIL import Image
    from pxr import Usd, UsdGeom, UsdPhysics
    boxes = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    extents = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0')):
        path = prim.GetPath().pathString
        if path.count('/') > 7 or '/Robot/' in path or not prim.IsA(UsdGeom.Xformable):
            continue
        box = boxes.ComputeWorldBound(prim).ComputeAlignedRange()
        low, high = np.array(box.GetMin()), np.array(box.GetMax())
        if np.isfinite(low).all() and .05 < max(high[:2]-low[:2]) < 8:
            extents[path] = [np.round(low, 3).tolist(), np.round(high, 3).tolist()]
    assets = {}
    for name, asset in spec['assets'].items():
        asset_stage = Usd.Stage.Open(os.environ['PROJECT_ROOT']+'/'+asset['usd'])
        vertices, faces = extract_mesh(asset_stage.GetDefaultPrim())
        assets[name] = {'meters_per_unit': UsdGeom.GetStageMetersPerUnit(asset_stage), 'up_axis': str(UsdGeom.GetStageUpAxis(asset_stage)),
                        'min': vertices.min(0).round(4).tolist(), 'max': vertices.max(0).round(4).tolist(), 'triangles': len(faces),
                        'rigid_bodies': sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in asset_stage.Traverse())}
    robot = env.scene['robot']
    state = {key: env.scene[key].data.root_state_w[0, :7].detach().cpu().numpy().round(4).tolist()
             for key in env.scene.keys() if hasattr(getattr(env.scene[key], 'data', None), 'root_state_w')}
    (out/'probe.json').write_text(json.dumps({'scene_keys': list(env.scene.keys()), 'root_states': state,
        'default_root_state': robot.data.default_root_state[0, :7].detach().cpu().numpy().round(4).tolist(),
        'extents': extents, 'assets': assets}, indent=1))
    world = env.scene['world_camera'] if 'world_camera' in env.scene.keys() else None
    if world is not None:
        for _ in range(6):
            env.sim.render()
        world.update(0., force_recompute=True)
        Image.fromarray(world.data.output['rgb'][0].detach().cpu().numpy()[..., :3]).save(out/'probe-world.png')


def run(env, spec, scene_name, out):
    from PIL import Image
    from pxr import UsdGeom
    from arena.vlm_capture import capture
    scene, stage = spec['scenes'][scene_name], env.sim.get_initial_stage()
    for folder in ('inputs', 'truth/meshes'):
        (out/folder).mkdir(parents=True, exist_ok=True)
    if os.environ.get('POINTING_PROBE') == '1':
        probe(env, stage, spec, out)
    for path in scene.get('hide', []):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise ValueError(f'Nothing to hide at {path}')
        UsdGeom.Imageable(prim).MakeInvisible()
    prims, lowest = {}, {}
    for name, asset_name in scene['instances'].items():
        prims[name] = stage.GetPrimAtPath(f'/World/envs/env_0/Pointing_{name}')
        if not prims[name].IsValid():
            raise ValueError(f'Copy {name} was not spawned')
        asset, target = spec['assets'][asset_name], out/'truth/meshes'/f'{asset_name}.npz'
        vertices, faces = extract_mesh(prims[name])
        vertices = vertices @ quat_matrix(asset.get('base_quat', [1, 0, 0, 0])).T  # into the frame whose pose is recorded
        box = np.asarray(asset.get('graspable_box', [[-np.inf]*3, [np.inf]*3]), float)
        centres = vertices[faces].mean(1)
        graspable = ((centres >= box[0]) & (centres <= box[1])).all(1)
        if not graspable.any():
            raise ValueError(f'{asset_name}: no triangle inside graspable_box')
        if not target.exists():
            np.savez_compressed(target, vertices=vertices, faces=faces, graspable=graspable)
        lowest[asset_name] = float(vertices[:, 2].min())
    records = []
    for frame in spec['frames']:
        if frame['scene'] != scene_name:
            continue
        place_robot(env, *frame['robot'])
        objects = {}
        for i, name in enumerate(scene['instances']):
            if name in frame['objects']:
                # The mesh was stored after base_quat, so the recorded pose is the yaw-only resting pose.
                asset_name = scene['instances'][name]
                position, quat = resting_pose(lowest[asset_name], *frame['objects'][name])
                set_pose(prims[name], position, quat_multiply(quat, np.asarray(spec['assets'][asset_name].get('base_quat', [1, 0, 0, 0]), float)))
                objects[name] = {'asset': asset_name, 'position': position.tolist(), 'quat_wxyz': quat.tolist()}
            else:
                set_pose(prims[name], (PARK[0]+3*i, PARK[1], PARK[2]), [1, 0, 0, 0])
        if frame['target'] not in objects:
            raise ValueError(f"{frame['id']}: target is not placed")
        # One 5 ms physics step publishes the new robot pose to the camera; nothing else moves in that time.
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        robot_pose = env.scene['robot'].data.root_state_w[0, :7].detach().cpu().numpy()
        capture(env, out/'inputs')
        with np.load(out/'inputs/vlm-capture.npz', allow_pickle=False) as f:
            data = {k: f[k].copy() for k in f.files}
        (out/'inputs/vlm-capture.npz').unlink()
        Image.fromarray(data['rgb']).save(out/'inputs'/f"{frame['id']}.png")
        # Geometry for back-projection. The model sees only the PNG and the instruction.
        np.savez_compressed(out/'inputs'/f"{frame['id']}.npz", **{k: v for k, v in data.items() if k != 'rgb'})
        records.append({'id': frame['id'], 'scene': scene_name, 'category': frame['category'], 'target': frame['target'],
                        'instruction': frame['instruction'], 'objects': objects, 'robot_root_pose': robot_pose.tolist(),
                        'robot_stance': frame['robot']})
    (out/f'truth/frames-{scene_name}.json').write_text(json.dumps(records, indent=1))


def install():
    """Replaces the oracle bridge: add the object copies to the scene, capture, then stop before any control."""
    spec, scene_name, out = json.loads(Path(os.environ['POINTING_SPEC']).read_text()), os.environ['POINTING_SCENE'], Path(os.environ['POINTING_OUT'])
    parse = importlib.import_module('isaaclab_tasks.utils.parse_cfg')
    original_parse = parse.parse_env_cfg

    def parse_env_cfg(*args, **kwargs):
        cfg = original_parse(*args, **kwargs)
        add_instances(cfg.scene, spec['scenes'][scene_name], spec['assets'])
        return cfg

    parse.parse_env_cfg = parse_env_cfg
    factory = importlib.import_module('action_provider.create_action_provider')

    def create(env, args):
        run(env, spec, scene_name, out)
        (out/f'captured-{scene_name}').write_text('ok')
        raise SystemExit(0)

    factory.create_action_provider = create
