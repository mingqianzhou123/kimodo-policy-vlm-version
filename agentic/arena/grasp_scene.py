"""Process-local box contact reporting and a read-only audit of what the box rests on.

Nothing here changes mass, friction, geometry or any shared asset. The box only gains a
contact report, so that "held by the hands" and "off its support" are measured forces.
"""
import json
import os
from pathlib import Path

import numpy as np

SENSOR = 'box_contact'


def hand_side(body_name):
    """'left'/'right' for a hand, None for every other robot body.

    A hand is everything from the last arm joint outwards: the wrist-yaw link that carries
    it, the palm, the camera base on the palm and the fingers. Forearms, elbows and the
    torso are not hands, so a box resting on them is not held by the hands.
    """
    if '_hand_' in body_name or body_name.endswith('_wrist_yaw_link'):
        return body_name.split('_', 1)[0]
    return None


def split_forces(net, matrix, names, floor_n=.05):
    """Box contact forces by source: each hand, the rest of the robot, everything else.

    ``net`` is the total contact force on the box [3]; ``matrix`` holds the force from each
    filtered robot body [M, 3], named by ``names``. The remainder is whatever is not the
    robot: the support surface, the floor, the shelf. ``bodies`` lists who is pressing.
    """
    net, matrix = np.asarray(net, dtype=float), np.asarray(matrix, dtype=float)
    if net.shape != (3,) or matrix.shape != (len(names), 3):
        raise ValueError('Unexpected contact force shapes')
    matrix = np.nan_to_num(matrix)
    sides = [hand_side(n) for n in names]
    group = lambda side: np.linalg.norm(matrix[[s == side for s in sides]].sum(0)) if side in sides else 0.
    each = np.linalg.norm(matrix, axis=1)
    return {'left_n': float(group('left')), 'right_n': float(group('right')),
            'robot_other_n': float(group(None)),
            'external_n': float(np.linalg.norm(net - matrix.sum(0))),
            'bodies': {n: round(float(f), 2) for n, f in zip(names, each) if f >= floor_n}}


def install_grasp_scene():
    from isaaclab.scene import InteractiveScene
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
    import omni.usd
    original = InteractiveScene.__init__
    run = Path(os.environ['ORACLE_RUN'])

    def init(scene, cfg, *args, **kwargs):
        cfg.box.spawn.activate_contact_sensors = True
        original(scene, cfg, *args, **kwargs)
        stage = omni.usd.get_context().get_stage()
        root = '/World/envs/env_0'
        reporters = [p for p in Usd.PrimRange(stage.GetPrimAtPath(root + '/Box'))
                     if p.HasAPI(PhysxSchema.PhysxContactReportAPI)]
        if len(reporters) != 1 or not reporters[0].HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f'Expected one rigid box body with a contact report, got {reporters}')
        robot = scene['robot']
        bodies = {p.GetName(): str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(root + '/Robot'))
                  if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        names = sorted(bodies)
        pattern = lambda path: path.replace('/env_0/', '/env_.*/')
        sensor = ContactSensor(ContactSensorCfg(prim_path=pattern(str(reporters[0].GetPath())),
            update_period=0., history_length=1, track_air_time=False, debug_vis=False,
            filter_prim_paths_expr=[pattern(bodies[n]) for n in names]))
        scene._sensors[SENSOR] = sensor
        scene.grasp_filter_bodies = names
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
        bound = lambda prim: cache.ComputeWorldBound(prim).ComputeAlignedRange()
        box = bound(stage.GetPrimAtPath(root + '/Box'))
        center = (np.array(box.GetMin()) + np.array(box.GetMax())) / 2
        colliders = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root + '/Room')):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            b = bound(prim)
            low, high = np.array(b.GetMin()), np.array(b.GetMax())
            if np.any(low[:2] > center[:2] + 2.5) or np.any(high[:2] < center[:2] - 2.5) or low[2] > 1.5:
                continue
            colliders.append({'path': str(prim.GetPath()), 'min': low.tolist(), 'max': high.tolist()})
        (run / 'grasp-scene.json').write_text(json.dumps({
            'box_body': str(reporters[0].GetPath()), 'box_spawn_aabb': [list(box.GetMin()), list(box.GetMax())],
            'filter_bodies': names, 'robot_body_names': list(getattr(robot, 'body_names', []) or []),
            'colliders_near_box': colliders,
            'note': 'AABBs are at spawn, before the per-episode object reset moves the box'}, indent=2))

    InteractiveScene.__init__ = init


def read_contact(env):
    """Current box contact forces split by source, from the sensor added above."""
    sensor = env.scene.sensors[SENSOR]
    matrix = sensor.data.force_matrix_w
    if matrix is None:
        raise RuntimeError('Box contact sensor has no filtered force matrix')
    return split_forces(sensor.data.net_forces_w[0, 0].detach().cpu().numpy(),
                        matrix[0, 0].detach().cpu().numpy(), env.scene.grasp_filter_bodies)


def place_at_stance(env):
    """Diagnostic start: put the robot at the grasp stance before it settles, so that an
    attempt needs no walking. The stance is the one the policy would walk to."""
    from kimodo_vla.oracle.grasp_policy import stance
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/grasp-loop.json').read_text())['policy']
    robot = env.scene['robot']
    state = robot.data.root_state_w.clone()
    goal = stance(env.scene['box'].data.root_com_pos_w[0].detach().cpu().numpy(), config)
    state[0, 0], state[0, 1] = float(goal[0]), float(goal[1])
    state[0, 7:] = 0.
    robot.write_root_pose_to_sim(state[:, :7])
    robot.write_root_velocity_to_sim(state[:, 7:])
    (Path(os.environ['ORACLE_RUN']) / 'start-at-stance.json').write_text(json.dumps({'root_xy': goal.tolist()}))


def load_layout():
    """The placement asked for through GRASP_LAYOUT, resolved once and kept in the run folder.

    GRASP_LAYOUT is JSON: either a map cell {"cell": {depth, lateral, yaw, extra}}, a seeded
    placement in the fixed region {"region_seed": N, "speed_mps": V}, or a seeded placement over the
    wide range with the robot left at the task's start {"wide_seed": N, "speed_mps": V}. None when
    it is not set.
    """
    spec = os.environ.get('GRASP_LAYOUT')
    if not spec:
        return None
    from arena.box_layout import in_region, place, supported, wide
    run = Path(os.environ['ORACLE_RUN'])
    saved = run / 'layout.json'
    if saved.is_file():
        return json.loads(saved.read_text())
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/grasp-loop.json').read_text())
    spec = json.loads(spec)
    if 'cell' in spec:
        c = spec['cell']
        layout = place(c['depth'], c['lateral'], c['yaw'], c['extra'], config)
    elif 'wide_seed' in spec:
        layout = {**wide(spec['wide_seed'], config), 'speed_mps': float(spec.get('speed_mps', 0.))}
    else:
        layout = {**in_region(spec['region_seed'], config), 'speed_mps': float(spec.get('speed_mps', 0.))}
    table = next(c for c in json.loads((run / 'grasp-scene.json').read_text())['colliders_near_box']
                 if 'PackingTable' in c['path'])
    corners = [[table['min'][0], table['min'][1]], [table['max'][0], table['max'][1]]]
    ends = [layout['box_xy'][1]] + layout.get('segment_y', [])
    if not all(supported([layout['box_xy'][0], y], layout['yaw_deg'], config['scene']['box_half_m'], corners) for y in ends):
        raise ValueError(f'Box placement not fully over the table: {layout}')
    layout['spec'] = spec
    saved.write_text(json.dumps(layout, indent=1))
    return layout


def place_layout(env):
    """Put the box and the robot where the layout says, before the robot settles."""
    from scipy.spatial.transform import Rotation
    layout = load_layout()
    box = env.scene['box']
    state = box.data.root_state_w.clone()
    state[0, 0], state[0, 1] = layout['box_xy']
    x, y, z, w = Rotation.from_euler('z', layout['yaw_deg'], degrees=True).as_quat()
    state[0, 3:7] = state.new_tensor([w, x, y, z])
    state[0, 7:] = 0.
    box.write_root_pose_to_sim(state[:, :7])
    box.write_root_velocity_to_sim(state[:, 7:])
    if layout['robot_xy'] is None:   # wide placements: the robot starts where the task puts it
        return
    robot = env.scene['robot']
    state = robot.data.root_state_w.clone()
    state[0, 0], state[0, 1] = layout['robot_xy']
    state[0, 7:] = 0.
    robot.write_root_pose_to_sim(state[:, :7])
    robot.write_root_velocity_to_sim(state[:, 7:])


def drive_box(env, rest, y, vy):
    """Impose the driven box pose for the coming control step: ``rest`` is its pose [7] when the
    drive started (x, height and orientation stay), y and the y-velocity come from the drive."""
    box = env.scene['box']
    pose = rest.clone()
    pose[0, 1] = float(y)
    velocity = pose.new_zeros((1, 6))
    velocity[0, 1] = float(vy)
    box.write_root_pose_to_sim(pose)
    box.write_root_velocity_to_sim(velocity)
