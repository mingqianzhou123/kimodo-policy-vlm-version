"""Settle on the task floor with SONIC before starting the measured episode."""
import importlib
import json
import os
from pathlib import Path

import numpy as np

from arena.actuator_audit import array, install_actuator_audit
from arena.dynamics_override import install_dynamics_override
from arena.vlm_markers import create_markers
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.oracle.geometry import rotation_wxyz

FEET = ['left_ankle_roll_link', 'right_ankle_roll_link']
WINDOW = 25
LIMIT = 250


def stable(row):
    return (min(row['vertical_force_n']) > 10
            and max(abs(np.asarray(row['ankle_height_m']) - .035)) <= .005
            and row['root_height_m'] > .65 and row['tilt_deg'] < 15
            and row['root_speed_m_s'] < .02 and row['root_angular_speed_rad_s'] < .10
            and max(row['foot_speed_m_s']) < .01)


def settle(env, provider, warm_action, run, markers=True, before=None):
    if before is not None:
        before(env)
    data = env.scene['robot'].data
    robot = env.scene['robot']
    feet = [list(robot.body_names).index(n) for n in FEET]
    joints = [list(robot.joint_names).index(n) for n in G1_JOINT_NAMES]
    sensor = env.scene.sensors['contact_forces']
    contacts = [list(sensor.body_names).index(n) for n in FEET]
    initial = array(data.root_state_w[0])
    # All five archived cases start at identity heading and orientation.
    if not np.allclose(initial[3:7], [1, 0, 0, 0], atol=1e-6):
        raise ValueError('Unexpected reset orientation')
    hold = np.r_[0., 0., initial[2], np.eye(3)[:, :2].ravel(), array(data.joint_pos[0,joints]), 0., 0.]
    if markers:
        create_markers(env, run, 'front_camera', float(os.environ['VLM_MARKER_REACH_RAD']),
                       float(os.environ['VLM_CAMERA_LATERAL_M']))
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
            row = dict(frame=i+1, time=end, root_height_m=root[2],
                       tilt_deg=float(np.degrees(np.arccos(np.clip(rotation_wxyz(root[3:7])[2,2], -1, 1)))),
                       vertical_force_n=array(sensor.data.net_forces_w[0,contacts,2]).tolist(),
                       ankle_height_m=array(data.body_pos_w[0,feet,2]).tolist(),
                       root_speed_m_s=float(np.linalg.norm(root[7:10])),
                       root_angular_speed_rad_s=float(np.linalg.norm(root[10:13])),
                       foot_speed_m_s=np.linalg.norm(array(data.body_lin_vel_w[0,feet]),axis=1).tolist())
            consecutive = consecutive+1 if stable(row) else 0
            log.write(json.dumps(dict(row, consecutive=consecutive))+'\n')
            if consecutive == WINDOW:
                # Reset only the action coordinate integrator. Preserve SONIC's
                # physically warmed observation history, drives and robot state.
                quat, xy = provider._get_current_robot_root_pose_for_vla()
                provider._lerobot_vla_runtime.reset(body_xy_world=xy, target_root_quat_wxyz=quat)
                provider._vla_initial_robot_quat_wxyz = quat.copy()
                provider._latest_pose_payload = {}
                return
    raise RuntimeError('Double support did not stabilize within 5 s')


def install(install_bridge, *, markers=True, audit=True, before=None):
    """The defaults are the 2026-11-23 reach configuration; the grasp loop needs neither extra."""
    factory = importlib.import_module('action_provider.create_action_provider')
    install_dynamics_override()
    aligned = factory.create_action_provider

    def remember_warm_action(env, args):
        provider = aligned(env, args)
        provider._settling_action = provider.get_action
        return provider

    factory.create_action_provider = remember_warm_action
    install_bridge()
    if audit:
        install_actuator_audit()
    audited = factory.create_action_provider

    def create(env, args):
        provider = audited(env, args)
        action = provider.get_action
        ready = False

        def act(current_env):
            nonlocal ready
            if not ready:
                settle(current_env, provider, provider._settling_action, Path(os.environ['ORACLE_RUN']), markers, before)
                ready = True
            return action(current_env)

        provider.get_action = act
        return provider
    factory.create_action_provider = create
