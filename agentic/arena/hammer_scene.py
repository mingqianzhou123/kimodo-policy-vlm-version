"""DoubleDesk hammer scene, process-local: contact report on the hammer, placement and records.

Nothing here changes mass, friction, geometry or any shared asset. The hammer (scene key
``object_l``) only gains a contact report, so that "held by the hand" and "off its support"
are measured forces, exactly as for the box (``grasp_scene``).
"""
import json
import os
from pathlib import Path

import numpy as np

from arena.grasp_scene import split_forces

KEY = 'object_l'
SENSOR = 'hammer_contact'
CONFIG = Path(__file__).resolve().parents[1] / 'configs/hammer-grasp.json'


def config():
    return json.loads(CONFIG.read_text())


def install_hammer_scene():
    from isaaclab.scene import InteractiveScene
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
    import omni.usd
    original = InteractiveScene.__init__
    run = Path(os.environ['ORACLE_RUN'])

    def init(scene, cfg, *args, **kwargs):
        getattr(cfg, KEY).spawn.activate_contact_sensors = True
        original(scene, cfg, *args, **kwargs)
        stage = omni.usd.get_context().get_stage()
        root = '/World/envs/env_0'
        reporters = [p for p in Usd.PrimRange(stage.GetPrimAtPath(root + '/ObjectL'))
                     if p.HasAPI(PhysxSchema.PhysxContactReportAPI) and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(reporters) != 1:
            raise ValueError(f'Expected one rigid hammer body with a contact report, got {reporters}')
        bodies = {p.GetName(): str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(root + '/Robot'))
                  if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        names = sorted(bodies)
        pattern = lambda path: path.replace('/env_0/', '/env_.*/')
        scene._sensors[SENSOR] = ContactSensor(ContactSensorCfg(prim_path=pattern(str(reporters[0].GetPath())),
            update_period=0., history_length=1, track_air_time=False, debug_vis=False,
            filter_prim_paths_expr=[pattern(bodies[n]) for n in names]))
        scene.hammer_filter_bodies = names
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
        bound = lambda prim: cache.ComputeWorldBound(prim).ComputeAlignedRange()
        body = reporters[0]
        local = cache.ComputeRelativeBound(body, body).ComputeAlignedRange()
        spawn = bound(stage.GetPrimAtPath(root + '/ObjectL'))
        centre = (np.array(spawn.GetMin()) + np.array(spawn.GetMax())) / 2
        colliders = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root + '/Room')):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            b = bound(prim)
            low, high = np.array(b.GetMin()), np.array(b.GetMax())
            if np.any(low[:2] > centre[:2] + 2.) or np.any(high[:2] < centre[:2] - 2.) or low[2] > 1.5 or high[2] < .3:
                continue
            colliders.append({'path': str(prim.GetPath()), 'min': low.tolist(), 'max': high.tolist()})
        (run / 'hammer-scene.json').write_text(json.dumps({
            'hammer_body': str(body.GetPath()), 'hammer_local_bounds': [list(local.GetMin()), list(local.GetMax())],
            'hammer_spawn_aabb': [list(spawn.GetMin()), list(spawn.GetMax())], 'filter_bodies': names,
            'colliders_near_hammer': colliders,
            'note': 'AABBs are at spawn, before the per-episode object reset moves the hammer'}, indent=2))

    InteractiveScene.__init__ = init


def read_contact(env):
    """Current hammer contact forces split by source (each hand, rest of robot, everything else)."""
    sensor = env.scene.sensors[SENSOR]
    matrix = sensor.data.force_matrix_w
    if matrix is None:
        raise RuntimeError('Hammer contact sensor has no filtered force matrix')
    return split_forces(sensor.data.net_forces_w[0, 0].detach().cpu().numpy(),
                        matrix[0, 0].detach().cpu().numpy(), env.scene.hammer_filter_bodies)


def table_rect(run):
    """(x, y) min and max of the hammer table's top, from the scene record and the configuration."""
    cfg = config()['scene']
    colliders = json.loads((Path(run) / 'hammer-scene.json').read_text())['colliders_near_hammer']
    table = next((c for c in colliders if c['path'] == cfg['table_prim']), None)
    if table is None:
        raise ValueError(f"Table {cfg['table_prim']} not found near the hammer")
    return [table['min'][:2], table['max'][:2]], table


def load_layout():
    """The placement asked for through GRASP_LAYOUT, resolved once and kept in the run folder.

    GRASP_LAYOUT is JSON: a map cell {"cell": {depth, lateral, yaw, extra}} or a seeded
    placement in the fixed region {"region_seed": N}; either may carry an "inject" or a "motion"
    request (see ``hammer_loop``). None when it is not set.
    """
    spec = os.environ.get('GRASP_LAYOUT')
    if not spec:
        return None
    from arena.hammer_layout import in_region, place, supported
    run = Path(os.environ['ORACLE_RUN'])
    saved = run / 'layout.json'
    if saved.is_file():
        return json.loads(saved.read_text())
    cfg, spec = config(), json.loads(spec)
    if 'cell' in spec:
        c = spec['cell']
        layout = place(c['depth'], c['lateral'], c['yaw'], c['extra'], cfg)
    else:
        layout = in_region(spec['region_seed'], cfg)
    rect, table = table_rect(run)
    if abs(table['max'][2] - cfg['scene']['table_top_z_m']) > .01:
        raise ValueError(f"Table top {table['max'][2]} differs from the configuration")
    f = np.radians(cfg['scene']['facing_deg'])
    forward = np.array([np.cos(f), np.sin(f)])
    near = min(np.asarray([[x, y] for x in (rect[0][0], rect[1][0]) for y in (rect[0][1], rect[1][1])]) @ forward)
    if abs(near - np.asarray(cfg['scene']['table_edge_point_m']) @ forward) > .01:
        raise ValueError(f'The table near edge ({near}) differs from the configuration')
    if not np.allclose(rect, cfg['scene']['table_rect_m'], atol=.01):
        raise ValueError(f'The table top {rect} differs from the configuration')
    if not supported(layout['hammer_pos'], layout['hammer_yaw_rad'], cfg, rect):
        raise ValueError(f'Hammer placement not fully over the table: {layout}')
    layout['spec'] = spec
    saved.write_text(json.dumps(layout, indent=1))
    return layout


def set_hammer(env, xyz, yaw):
    """Teleport the hammer to a pose at rest."""
    from scipy.spatial.transform import Rotation
    hammer = env.scene[KEY]
    state = hammer.data.root_state_w.clone()
    state[0, 0:3] = state.new_tensor(xyz)
    x, y, z, w = Rotation.from_euler('z', yaw).as_quat()
    state[0, 3:7] = state.new_tensor([w, x, y, z])
    state[0, 7:] = 0.
    hammer.write_root_pose_to_sim(state[:, :7])
    hammer.write_root_velocity_to_sim(state[:, 7:])


def drive_hammer(env, xy, velocity_xy):
    """Impose the driven hammer position and velocity for the coming control step (height and
    orientation stay as they are)."""
    hammer = env.scene[KEY]
    state = hammer.data.root_state_w.clone()
    state[0, 0], state[0, 1] = float(xy[0]), float(xy[1])
    state[0, 7:] = 0.
    state[0, 7], state[0, 8] = float(velocity_xy[0]), float(velocity_xy[1])
    hammer.write_root_pose_to_sim(state[:, :7])
    hammer.write_root_velocity_to_sim(state[:, 7:])


def place_hammer_layout(env):
    """Put the hammer and the robot where the layout says, before the robot settles."""
    import torch
    layout = load_layout()
    set_hammer(env, layout['hammer_pos'], layout['hammer_yaw_rad'])
    robot = env.scene['robot']
    state = robot.data.root_state_w.clone()
    state[0, 0], state[0, 1] = layout['robot_xy']
    yaw = layout['robot_yaw_rad']
    state[0, 3:7] = torch.tensor([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)], dtype=state.dtype, device=state.device)
    state[0, 7:] = 0.
    robot.write_root_pose_to_sim(state[:, :7])
    robot.write_root_velocity_to_sim(state[:, 7:])
