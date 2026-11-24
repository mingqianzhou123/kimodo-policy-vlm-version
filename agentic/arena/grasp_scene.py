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
