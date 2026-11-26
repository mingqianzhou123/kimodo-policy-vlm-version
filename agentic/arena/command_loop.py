"""Simulator side of the command loop (``kimodo_vla/agent/loop.py``), for P&PBox, SitSofa and VisionNavi.

Installed inside the oracle bridge, so every request still carries the measured history. On each
request it adds the ground-truth objects (``scene_truth``, world frame: x, y on the floor, z up,
metres) and, when the previous reply asked for one, a camera capture of this instant
(``vision``). After every control step it appends one row to ``loop/steps.jsonl``: measured root
and wrists, hand command, object poses, contact forces and Arena's own success check. When a
reply says ``end`` it stops the episode the way the grasp loop does. Nothing here changes mass,
friction, geometry or any shared asset; the box only gains a contact report (``grasp_scene``).
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np

from kimodo_vla.oracle.geometry import rotation_wxyz

TASK_MODULES = {'pp_box': 'tasks.g1_tasks.move_pickplace_box_g1_29dof_dex3_wholebody.mdp.rewards',
                'sit_sofa': 'tasks.g1_tasks.move_sit_sofa_g1_29dof_dex3_wholebody.mdp.rewards',
                'vision_navi': 'tasks.g1_tasks.move_small_warehouse_vision_navigation_g1_29dof_dex3_wholebody.mdp.rewards'}
NAVI_OBSTACLES = ('Obstacle01_A', 'Obstacle01_B', 'Obstacle01_C', 'Obstacle02_A', 'Obstacle02_B', 'Obstacle02_C')
# Arena's own list of bodies that count as sitting (move_sit_sofa .../mdp/rewards.py), read here as the hip contact.
HIP_BODIES = ('pelvis', 'pelvis_contour_link', 'imu_in_pelvis', 'waist_yaw_link', 'waist_roll_link', 'waist_pitch_link',
              'torso_link', 'left_hip_pitch_link', 'right_hip_pitch_link', 'left_hip_roll_link', 'right_hip_roll_link',
              'left_hip_yaw_link', 'right_hip_yaw_link')
SUPPORT_BODIES = ('ankle', 'foot')     # touching the floor is expected
CONTACT_FLOOR_N = 5.
SUCCESS_EVERY = 5                      # Arena's success check every this many control steps (it walks the stage)
FALL_HEIGHT_M, FALL_TILT_DEG = .40, 60.


def yaw_deg(quat_wxyz):
    r = rotation_wxyz(quat_wxyz)
    return float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))


def region(name, low, high, **extra):
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    return {'name': name, 'kind': 'region', 'min': low.tolist(), 'max': high.tolist(),
            'center': ((low+high)/2).tolist(), 'size': (high-low).tolist(), **extra}


def classify_contacts(names, forces, floor=CONTACT_FLOOR_N):
    """Robot contact forces by body: hip/pelvis bodies, and every other non-foot body pressing at least ``floor``."""
    norms = np.linalg.norm(np.asarray(forces, dtype=float), axis=-1)
    hip = float(sum(f for n, f in zip(names, norms) if n in HIP_BODIES))
    other = {n: round(float(f), 1) for n, f in zip(names, norms)
             if f >= floor and n not in HIP_BODIES and not any(s in n for s in SUPPORT_BODIES)}
    return hip, other


class Hooks:
    def __init__(self, env, provider, run, task):
        self.env, self.provider, self.run, self.task = env, provider, Path(run), task
        (self.run/'loop').mkdir(parents=True, exist_ok=True)
        self.steps = (self.run/'loop'/'steps.jsonl').open('a', buffering=1)
        self.capture_next, self.captures, self.static, self.count = False, 0, None, 0
        self.rewards = importlib.import_module(TASK_MODULES[task])
        robot = env.scene['robot']
        self.body_names = list(robot.body_names)
        self.wrist_ids = [self.body_names.index(f'{s}_wrist_yaw_link') for s in ('left', 'right')]
        sensor = env.scene.sensors['contact_forces']
        self.sensor_names = list(sensor.body_names)
        self.last_success = None

    # ---- objects --------------------------------------------------------------------------
    def stage_bounds(self, path):
        import omni.usd
        from pxr import Usd, UsdGeom
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        return np.array(box.GetMin(), dtype=float), np.array(box.GetMax(), dtype=float)

    def parts(self, root, depth=None):
        """Axis-aligned bounds of every prim under ``root`` (``depth`` levels down only, if given)."""
        import omni.usd
        from pxr import Usd, UsdGeom
        stage = omni.usd.get_context().get_stage()
        top = stage.GetPrimAtPath(root)
        if not top or not top.IsValid():
            return []
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
        out, base = [], str(top.GetPath()).count('/')
        for prim in Usd.PrimRange(top):
            level = str(prim.GetPath()).count('/')-base
            if level == 0 or (depth is not None and level > depth):
                continue
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if box.IsEmpty():
                continue
            low, high = np.array(box.GetMin(), dtype=float), np.array(box.GetMax(), dtype=float)
            out.append({'path': str(prim.GetPath()), 'name': prim.GetName(), 'min': low.tolist(), 'max': high.tolist()})
        return out

    def static_objects(self):
        """Fixed scenery, read once after the episode's reset placed it."""
        if self.static is not None:
            return self.static
        objects, extra = [], {'room_assets': self.parts('/World/envs/env_0/Room/Assets', depth=1)}
        if self.task == 'pp_box':
            scene = json.loads((self.run/'grasp-scene.json').read_text())
            table = next(c for c in scene['colliders_near_box'] if 'PackingTable' in c['path'])
            objects.append(region('table', table['min'], table['max'], top_z=table['max'][2]))
            surfaces = self.rewards._get_shelf_support_surfaces_world(self.env)[0]
            base = min(s[4] for s in surfaces)
            target_top = base+self.rewards._SHELF_ACTIVE_LAYER_Z_FROM_BASE_M
            widest = max(surfaces, key=lambda s: (s[1]-s[0])*(s[3]-s[2]))
            shelf = self.stage_bounds('/World/envs/env_0/Shelf')
            if shelf is not None:
                objects.append(region('shelf', *shelf))
            objects.append(region('shelf_target_layer', [widest[0], widest[2], target_top], [widest[1], widest[3], target_top],
                note='place the box so that its centre is over this area and its bottom rests at top z'))
            extra.update({'shelf_parts': self.parts('/World/envs/env_0/Shelf'),
                     'shelf_support_surfaces': [list(s) for s in surfaces], 'shelf_base_top_z': base,
                     'shelf_target_top_z': target_top, 'box_size_m': list(self.rewards._BOX_SIZE_M)})
        elif self.task == 'vision_navi':
            # Arena's own target region (the yellow box, else the target sign's ring) and the six laid-out obstacles.
            region_kind, values = self.rewards._get_stage_target_regions_world(
                self.env, self.rewards._get_target_positions(self.env))[0]
            if region_kind == 'circle':
                cx, cy, r = values
                objects.append(region('target', [cx-r, cy-r, 0.], [cx+r, cy+r, 0.], shape='circle', radius=r,
                                      note='stand with both feet inside this circle'))
            else:
                x0, x1, y0, y1 = values[:4]
                objects.append(region('target', [x0, y0, 0.], [x1, y1, 0.], shape='box', note='stand with both feet inside this area'))
            for name in NAVI_OBSTACLES:
                bounds = self.stage_bounds(f'/World/envs/env_0/{name}')
                if bounds is not None:
                    objects.append(region(name.lower(), *bounds, kind_of='obstacle'))
            extra.update({'target_region': [region_kind, list(values)]})
        else:
            seat = self.rewards._get_sofa_seat_boxes_world(self.env)[0]
            if seat is None:
                raise RuntimeError('Arena found no sofa seat on the stage')
            x0, x1, y0, y1, z0, z1 = seat
            objects.append(region('sofa_seat', [x0, y0, z0], [x1, y1, z1], top_z=z1))
            extra.update({'seat_box_world': list(seat), 'sit_bodies': list(HIP_BODIES)})
            # Furniture within 3 m of the seat, as obstacles (names from the scene file).
            centre = np.array([(x0+x1)/2, (y0+y1)/2])
            for part in extra['room_assets']:
                low, high = np.array(part['min']), np.array(part['max'])
                size = high-low
                if np.linalg.norm((low[:2]+high[:2])/2-centre) <= 3. and size[2] > .1 and max(size[:2]) < 4.:
                    objects.append(region(part['name'], low, high, kind_of='furniture'))
            for key in ('carton_obstacle_01', 'carton_obstacle_02'):
                bounds = self.stage_bounds(f'/World/envs/env_0/{"CartonObstacle0" + key[-1]}')
                if bounds is not None:
                    extra[f'{key}_size_m'] = (bounds[1]-bounds[0]).tolist()
        self.static = (objects, extra)
        return self.static

    def movable(self):
        out = []
        keys = {'pp_box': ['box'], 'sit_sofa': ['carton_obstacle_01', 'carton_obstacle_02']}.get(self.task, [])
        for key in keys:
            if key not in self.env.scene.keys():
                continue
            data = self.env.scene[key].data
            pos = data.root_com_pos_w[0].detach().cpu().numpy() if key == 'box' else data.root_pos_w[0].detach().cpu().numpy()
            quat = data.root_quat_w[0].detach().cpu().numpy()
            size = (list(self.rewards._BOX_SIZE_M) if key == 'box'
                    else self.static_objects()[1].get(f'{key}_size_m'))
            out.append({'name': key, 'kind': 'object', 'position': pos.tolist(), 'yaw_deg': yaw_deg(quat),
                        'quat_wxyz': quat.tolist(), 'size': size})
        return out

    def truth(self):
        objects, extra = self.static_objects()
        truth = {'frame': 'world: x, y on the floor, z up, metres; yaw counter-clockwise from +x',
                 'time': float(self.env.sim.current_time), 'objects': self.movable()+objects}
        if not (self.run/'loop'/'scene.json').exists():
            (self.run/'loop'/'scene.json').write_text(json.dumps({'task': self.task, **truth, **extra}, indent=1))
        return truth

    # ---- camera ---------------------------------------------------------------------------
    def capture(self):
        """One RGB image of this paused instant (``arena/vlm_capture`` without the depth image, which this
        mode does not enable), with the camera's pose; one file per decision."""
        name = os.environ.get('COMMAND_CAMERA', 'front_camera')
        camera = self.env.scene[name]
        camera.cfg.update_latest_camera_pose = True
        self.env.sim.forward()
        for _ in range(6):
            self.env.sim.render()
        camera.reset()
        camera.update(0., force_recompute=True)
        data = camera.data
        cpu = lambda x: x[0].detach().cpu().numpy().copy()
        path = self.run/'loop'/f'capture-{self.captures:03d}.npz'
        now = float(self.env.sim.current_time)
        np.savez_compressed(path, rgb=cpu(data.output['rgb'])[..., :3], K=cpu(data.intrinsic_matrices),
                            camera_position=cpu(data.pos_w), camera_rotation=rotation_wxyz(cpu(data.quat_w_ros)),
                            time=now, camera_name=name, camera_convention='ros_optical')
        self.captures += 1
        return {'capture_path': str(path), 'time': now, 'camera_name': name}

    # ---- every control step ---------------------------------------------------------------
    def after_step(self):
        env = self.env
        data = env.scene['robot'].data
        root = data.root_state_w[0].detach().cpu().numpy()
        tilt = float(np.degrees(np.arccos(np.clip(rotation_wxyz(root[3:7])[2, 2], -1, 1))))
        sensor = env.scene.sensors['contact_forces']
        hip, other = classify_contacts(self.sensor_names, sensor.data.net_forces_w[0].detach().cpu().numpy())
        row = {'time': float(env.sim.current_time), 'root_pos': root[:3].tolist(), 'root_quat': root[3:7].tolist(),
               'tilt_deg': tilt, 'fallen': bool(root[2] < FALL_HEIGHT_M or tilt > FALL_TILT_DEG),
               'wrist_pos': data.body_pos_w[0, self.wrist_ids].detach().cpu().numpy().tolist(),
               'wrist_quat': data.body_quat_w[0, self.wrist_ids].detach().cpu().numpy().tolist(),
               'hands_closed': [bool(self.provider._left_hand_binary_state), bool(self.provider._right_hand_binary_state)],
               'hip_contact_n': hip, 'other_contacts_n': other,
               'objects': [{k: o[k] for k in ('name', 'position', 'yaw_deg', 'quat_wxyz')} for o in self.movable()]}
        if self.task == 'pp_box':
            from arena.grasp_scene import read_contact
            box = read_contact(env)
            row['box_contact'] = {k: box[k] for k in ('left_n', 'right_n', 'robot_other_n', 'external_n')}
        if self.count % SUCCESS_EVERY == 0:
            self.last_success = bool(self.rewards.compute_success_mask(env)[0].item())
            row['arena_success'] = self.last_success
        self.count += 1
        self.steps.write(json.dumps(row)+'\n')

    def stop(self):
        controller = getattr(self.env, 'kvla_controller', None)
        if controller is not None:
            controller.is_running = False


def install_command_hooks():
    """Wrap the bridge's provider: objects and captures on requests, one record per step, stop on 'end'."""
    task = os.environ['COMMAND_TASK']
    factory = importlib.import_module('action_provider.create_action_provider')
    inner = factory.create_action_provider

    def create(env, args):
        provider = inner(env, args)
        hooks = Hooks(env, provider, Path(os.environ['ORACLE_RUN']), task)
        client = provider._lerobot_http_client
        post, get_action = client._post_json, provider.get_action

        def wrapped_post(path, payload):
            if path == '/infer':
                payload['observation']['scene_truth'] = hooks.truth()
                if hooks.capture_next:
                    payload['observation']['vision'] = hooks.capture()
                    hooks.capture_next = False
            response = post(path, payload)
            if path == '/infer':
                loop = response.get('command_loop') or {}
                hooks.capture_next = bool(loop.get('capture'))
                if loop.get('end'):
                    hooks.stop()
            return response

        def act(current_env):
            action = get_action(current_env)
            hooks.after_step()
            return action

        client._post_json, provider.get_action = wrapped_post, act
        return provider
    factory.create_action_provider = create
    # The evaluation loop runs while its controller does; remember it so the loop's 'end' can stop it.
    control = importlib.import_module('layeredcontrol.robot_control_system')
    set_provider = control.RobotController.set_action_provider

    def remember(controller, provider):
        controller.env.kvla_controller = controller
        return set_provider(controller, provider)
    control.RobotController.set_action_provider = remember
