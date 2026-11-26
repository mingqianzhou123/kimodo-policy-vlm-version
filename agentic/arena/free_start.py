"""Start the measured episode settled on both feet at a chosen spot on the open floor.

The task's reset pose faces a packing table 0.94 m ahead, which leaves no room to walk. This
moves the robot (position only, facing unchanged) before the same double-support settling the
standing experiments use, with the same aligned actuators. No markers, no actuator audit.
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np

from arena.actuator_audit import array
from arena.dynamics_override import install_dynamics_override
from arena.settled_start import FEET, LIMIT, WINDOW, stable
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.oracle.geometry import rotation_wxyz


def start_xy():
    """World x, y from KVLA_START_XY ("x,y"), or None to keep the task's reset position."""
    text = os.environ.get('KVLA_START_XY')
    if not text:
        return None
    xy = np.array([float(v) for v in text.split(',')])
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError('KVLA_START_XY must be "x,y" in metres')
    return xy


def relocate(env, provider, xy):
    """Put the root at world xy with its height, orientation and joints unchanged, at rest."""
    import torch
    robot = env.scene['robot']
    state = robot.data.root_state_w.clone()
    state[:, :2] = torch.as_tensor(xy, dtype=state.dtype, device=state.device)
    state[:, 7:] = 0.
    robot.write_root_pose_to_sim(state[:, :7])
    robot.write_root_velocity_to_sim(state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.joint_pos.clone(), torch.zeros_like(robot.data.joint_vel))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(0.)
    moved = array(robot.data.root_state_w[0, :2])
    if not np.allclose(moved, xy, atol=1e-4):
        raise RuntimeError(f'Robot was not moved: {moved} instead of {xy}')
    quat, here = provider._get_current_robot_root_pose_for_vla()
    provider._lerobot_vla_runtime.reset(body_xy_world=here, target_root_quat_wxyz=quat)
    provider._vla_initial_robot_quat_wxyz = quat.copy()
    provider._latest_pose_payload = {}


def settle(env, provider, warm_action, run):
    """Hold the measured joints until both feet carry load and nothing moves for 0.5 s."""
    data = env.scene['robot'].data
    robot = env.scene['robot']
    xy = start_xy()
    if xy is not None:
        relocate(env, provider, xy)
    feet = [list(robot.body_names).index(n) for n in FEET]
    joints = [list(robot.joint_names).index(n) for n in G1_JOINT_NAMES]
    sensor = env.scene.sensors['contact_forces']
    contacts = [list(sensor.body_names).index(n) for n in FEET]
    initial = array(data.root_state_w[0])
    if not np.allclose(initial[3:7], [1, 0, 0, 0], atol=1e-6):
        raise ValueError('Unexpected reset orientation')
    hold = np.r_[0., 0., initial[2], np.eye(3)[:, :2].ravel(), array(data.joint_pos[0, joints]), 0., 0.]
    consecutive = 0
    with (run/'settling.jsonl').open('x') as log:
        for i in range(LIMIT):
            if provider._lerobot_action_chunk_queue:
                raise ValueError('Unexpected pending VLA action during settling')
            provider._lerobot_action_chunk_queue.append(hold.copy())
            start = float(env.sim.current_time)
            warm_action(env)
            end = float(env.sim.current_time)
            if not np.isclose(end-start, .02, atol=1e-6) or provider._lerobot_action_chunk_queue:
                raise ValueError('SONIC warm-up did not consume one standing frame')
            root = array(data.root_state_w[0])
            row = dict(frame=i+1, time=end, root_height_m=root[2], root_xy_m=root[:2].tolist(),
                       tilt_deg=float(np.degrees(np.arccos(np.clip(rotation_wxyz(root[3:7])[2, 2], -1, 1)))),
                       vertical_force_n=array(sensor.data.net_forces_w[0, contacts, 2]).tolist(),
                       ankle_height_m=array(data.body_pos_w[0, feet, 2]).tolist(),
                       root_speed_m_s=float(np.linalg.norm(root[7:10])),
                       root_angular_speed_rad_s=float(np.linalg.norm(root[10:13])),
                       foot_speed_m_s=np.linalg.norm(array(data.body_lin_vel_w[0, feet]), axis=1).tolist())
            consecutive = consecutive+1 if stable(row) else 0
            log.write(json.dumps(dict(row, consecutive=consecutive))+'\n')
            if consecutive == WINDOW:
                # Reset only the action coordinate integrator. Preserve SONIC's
                # physically warmed observation history, drives and robot state.
                quat, here = provider._get_current_robot_root_pose_for_vla()
                provider._lerobot_vla_runtime.reset(body_xy_world=here, target_root_quat_wxyz=quat)
                provider._vla_initial_robot_quat_wxyz = quat.copy()
                provider._latest_pose_payload = {}
                if hasattr(provider, '_heft_reset'):   # its reference restarts with the integrator
                    provider._heft_reset()
                return
    raise RuntimeError('Double support did not stabilize within 5 s')


def install(install_bridge):
    factory = importlib.import_module('action_provider.create_action_provider')
    install_dynamics_override()
    if os.environ.get('KVLA_TRACKER') == 'heft':   # HEFT replaces SONIC in the provider, settling included
        from arena.heft_bridge import install_heft
        install_heft()
    aligned = factory.create_action_provider

    def remember_warm_action(env, args):
        provider = aligned(env, args)
        provider._settling_action = provider.get_action
        return provider

    factory.create_action_provider = remember_warm_action
    install_bridge()
    bridged = factory.create_action_provider

    def create(env, args):
        provider = bridged(env, args)
        action = provider.get_action
        ready = False

        def act(current_env):
            nonlocal ready
            if not ready:
                settle(current_env, provider, provider._settling_action, Path(os.environ['ORACLE_RUN']))
                ready = True
            return action(current_env)

        provider.get_action = act
        return provider
    factory.create_action_provider = create
