"""Process-local Arena instrumentation; does not edit the shared simulator source."""
import importlib
import json
import os
from pathlib import Path
import numpy as np
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.oracle.geometry import XmlFK, rotation_wxyz, angle_deg


def install_oracle_bridge():
    factory = importlib.import_module('action_provider.create_action_provider')
    original_create = factory.create_action_provider

    def create(env, args):
        provider = original_create(env, args)
        if args.action_source != 'sonic_wholebody':
            # Exact spelling is checked against the provider's actual capabilities below.
            if not hasattr(provider, '_sonic_idx'):
                raise ValueError('Oracle experiment requires SONIC')
        client = provider._lerobot_http_client
        if client is None:
            raise ValueError('Oracle experiment requires HTTP')
        get_action, post_json = provider.get_action, client._post_json
        body_names = list(env.scene['robot'].body_names)
        wrist_ids = [body_names.index(f'{s}_wrist_yaw_link') for s in ('left','right')]
        foot_names = [f'{s}_ankle_roll_link' for s in ('left','right')]
        foot_ids = [body_names.index(n) for n in foot_names]
        joint_names = list(env.scene['robot'].joint_names)
        joint_ids = [joint_names.index(n) for n in G1_JOINT_NAMES]
        sonic_ids=provider._sonic_idx.detach().cpu().numpy().tolist()
        sonic_to_canonical=[sonic_ids.index(i) for i in joint_ids]
        xml = Path(__file__).resolve().parents[2] / 'kimodo_vla/skeleton/assets/g1skel34/xml/g1.xml'
        fk = XmlFK(xml, G1_JOINT_NAMES)
        pending, target = [], None
        request_error, last_time = None, None
        run = Path(os.environ['ORACLE_RUN'])
        vision = None
        grasp = os.environ.get('GRASP_LOOP') == '1'
        # GRASP_TASK=hammer: the one-handed DoubleDesk hammer grasp (arena/hammer_loop.py); unset, the box.
        hammer = grasp and os.environ.get('GRASP_TASK') == 'hammer'
        # KVLA_TRACK_OBJECT names the scene object logged as box_pos / box_quat outside the grasp loops
        # (experiment 2's replays of SitSofa); unset, the box as before.
        target_key = 'object_l' if hammer else 'box' if grasp else os.environ.get('KVLA_TRACK_OBJECT', 'box')
        loop = drive_rest = None
        if hammer:
            from arena.hammer_loop import HammerGraspLoop
            from arena.hammer_scene import config as hammer_config, load_layout as hammer_layout, read_contact
            loop = HammerGraspLoop(run, hammer_config(), stop=lambda: setattr(env.kvla_controller, 'is_running', False),
                                   env=env, provider=provider, layout=hammer_layout())
            apply_hands = provider._apply_hand_binary_targets
            # The policy's hand channel is ignored: the right hand follows the gate, the left stays open.
            provider._apply_hand_binary_targets = lambda left_closed, right_closed: loop.command_hands(apply_hands)
        elif grasp:
            from arena.grasp_loop import GraspLoop
            from arena.grasp_scene import drive_box, load_layout, read_contact
            config = json.loads((Path(__file__).resolve().parents[1]/'configs/grasp-loop.json').read_text())
            table = next((c for c in json.loads((run/'grasp-scene.json').read_text())['colliders_near_box']
                          if 'PackingTable' in c['path']), None)
            if table is None or abs(table['min'][0]-config['scene']['table_edge_x_m']) > .01:
                raise ValueError('The table edge in the scene differs from the grasp configuration')
            # KVLA_GRASP_KEEP_RUNNING=1 (run_oracle.py --carry-goal): a success no longer ends the episode.
            keep = os.environ.get('KVLA_GRASP_KEEP_RUNNING') == '1'
            loop = GraspLoop(run, config, stop=(lambda: None) if keep else lambda: setattr(env.kvla_controller, 'is_running', False),
                             layout=load_layout())
            apply_hands = provider._apply_hand_binary_targets
            # The policy's hand channel is ignored: both hands follow the measured-arrival gate.
            provider._apply_hand_binary_targets = lambda left_closed, right_closed: apply_hands(
                left_closed=loop.closed, right_closed=loop.closed)

        def sample():
            data = env.scene['robot'].data
            root = data.root_state_w[0].detach().cpu().numpy()
            wp = data.body_pos_w[0, wrist_ids].detach().cpu().numpy()
            wq = data.body_quat_w[0, wrist_ids].detach().cpu().numpy()
            wr = np.stack([rotation_wxyz(q) for q in wq])
            box = env.scene[target_key].data.root_state_w[0].detach().cpu().numpy()
            reference = getattr(provider, '_latest_pose_payload', None)
            reference_root = (np.asarray(reference['body_pos']).reshape(3)
                              if reference and 'body_pos' in reference else root[:3])
            measured_joints=data.joint_pos[0,joint_ids].detach().cpu().numpy()
            return {'time': float(env.sim.current_time), 'root_pos': root[:3].tolist(),
                'reference_root_pos': reference_root.tolist(),
                'reference_root_quat': (np.asarray(reference['body_quat_w']).reshape(4).tolist()
                    if reference and 'body_quat_w' in reference else root[3:7].tolist()),
                'reference_joints': (np.asarray(reference['joint_pos']).reshape(29).tolist()
                    if reference and 'joint_pos' in reference else measured_joints.tolist()),
                'root_quat': root[3:7].tolist(),
                'joints': measured_joints.tolist(),
                'wrist_pos': wp.tolist(), 'wrist_rot': wr.tolist(),
                'foot_pos': data.body_pos_w[0,foot_ids].detach().cpu().numpy().tolist(),
                'foot_rot': [rotation_wxyz(q).tolist() for q in data.body_quat_w[0,foot_ids].detach().cpu().numpy()],
                'wrist_velocity': data.body_lin_vel_w[0, wrist_ids].detach().cpu().numpy().tolist(),
                'box_pos': box[:3].tolist(), 'box_quat': box[3:7].tolist(),
                **({'box_com': env.scene[target_key].data.root_com_pos_w[0].detach().cpu().numpy().tolist(),
                    # torque the actuators applied, canonical joint order (for the proprioceptive checks)
                    'applied_torque': data.applied_torque[0, joint_ids].detach().cpu().numpy().tolist()}
                   if grasp else {}),
                'hand_joint_pos': {n: float(data.joint_pos[0,i]) for i,n in enumerate(joint_names)
                                   if i not in joint_ids},
                'gripper_command': [int(provider._left_hand_binary_state), int(provider._right_hand_binary_state)]}

        def post(path, payload):
            nonlocal target, request_error, last_time, vision
            if path == '/infer':
                if not pending:
                    raise ValueError('No measured history at request')
                payload['observation']['oracle_history'] = pending.copy()
                if os.environ.get('VLM_CAPTURE')=='1':
                    if vision is None:
                        from arena.vlm_capture import capture
                        camera=os.environ.get('VLM_CAMERA','front_camera')
                        if os.environ.get('VLM_MARKERS')=='1':
                            from arena.vlm_markers import create_markers
                            create_markers(env,run,camera,
                                reach_rad=float(os.environ.get('VLM_MARKER_REACH_RAD','.15')),
                                camera_lateral_m=float(os.environ.get('VLM_CAMERA_LATERAL_M','0')))
                        vision=capture(env,run,camera)
                    payload['observation']['vision']=vision
            try:
                response = post_json(path, payload)
            except Exception as exc:
                request_error = exc
                raise
            if path == '/infer':
                target = response['oracle_target']
                if loop is not None:
                    loop.on_plan(response['grasp'], pending[-1])
                pending.clear()
            elif path == '/reset':
                pending.clear()
                target = last_time = request_error = None
                vision = None
            return response

        def act(current_env):
            nonlocal last_time, drive_rest
            if hammer:
                loop.act_on_scene(env, float(env.sim.current_time))
            elif loop is not None:
                drive = loop.drive(float(env.sim.current_time))
                if drive is not None:
                    if drive_rest is None:
                        drive_rest = env.scene['box'].data.root_state_w[:, :7].clone()
                    drive_box(env, drive_rest, *drive)
            before = sample()
            if last_time is not None and not np.isclose(before['time']-last_time,.02,atol=1e-6):
                raise ValueError(f'Unexpected control timing {last_time} -> {before["time"]}')
            last_time = before['time']
            pending.append(before)
            if loop is not None:
                loop.before_step(before)
            action = get_action(current_env)
            if request_error is not None:
                raise RuntimeError('Oracle HTTP failed') from request_error
            after = sample()
            record = {'before': before, 'after': after, 'control_dt': after['time']-before['time'],
                      'grasp_attempted': False, 'contact_sensor_available': False}
            if loop is not None:
                record.update(contact_sensor_available=True, grasp_attempted=loop.gate.fired_at is not None,
                              grasp=loop.after_step(after, read_contact(env)))
            if getattr(provider, '_heft_reference', None) is not None:   # the frame HEFT tracked (arena/heft_bridge.py)
                record['heft_reference'] = provider._heft_reference
                record['heft_checks'] = dict(provider._heft_checks)
            payload = provider._latest_pose_payload
            if payload and all(k in payload for k in ('body_pos','body_quat_w','joint_pos')):
                refp, refr = fk(np.asarray(payload['joint_pos']).reshape(29),
                               np.asarray(payload['body_pos']).reshape(3),
                               rotation_wxyz(np.asarray(payload['body_quat_w']).reshape(4)))
                measured = np.asarray(after['wrist_pos'])
                ref_feet, ref_feet_rot = fk(np.asarray(payload['joint_pos']).reshape(29),
                    np.asarray(payload['body_pos']).reshape(3),
                    rotation_wxyz(np.asarray(payload['body_quat_w']).reshape(4)), body_names=foot_names)
                measured_feet_fk,_ = fk(after['joints'],after['root_pos'],rotation_wxyz(after['root_quat']),
                                       body_names=foot_names)
                record.update(reference_foot_world=ref_feet.tolist(),reference_foot_rotation=ref_feet_rot.tolist(),
                    measured_xml_vs_sim_foot_m=np.linalg.norm(measured_feet_fk-after['foot_pos'],axis=-1).tolist())
                measured_fk, _ = fk(after['joints'], after['root_pos'], rotation_wxyz(after['root_quat']))
                refq=np.asarray(payload['joint_pos']).reshape(29)
                ref_at_actual_root,_=fk(refq,after['root_pos'],rotation_wxyz(after['root_quat']))
                record.update(reference_wrist_world=refp.tolist(),
                    reference_joints=refq.tolist(),
                    reference_root_quat=np.asarray(payload['body_quat_w']).reshape(4).tolist(),
                    root_anchored_reference_wrist_world=ref_at_actual_root.tolist(),
                    posture_wrist_m=np.linalg.norm(measured-ref_at_actual_root,axis=-1).tolist(),
                    root_pose_wrist_contribution_m=np.linalg.norm(ref_at_actual_root-refp,axis=-1).tolist(),
                    reference_root_world=np.asarray(payload['body_pos']).reshape(3).tolist(),
                    tracking_wrist_m=np.linalg.norm(measured-refp,axis=-1).tolist(),
                    measured_xml_vs_sim_wrist_m=np.linalg.norm(measured_fk-measured,axis=-1).tolist())
                decoder=np.asarray(provider._latest_decoder_target).reshape(29)[sonic_to_canonical]
                record['decoder_joint_target']=decoder.tolist()
                record['decoder_actual_joint_rmse_rad']=float(np.sqrt(np.mean((decoder-np.array(after['joints']))**2)))
                record['reference_actual_joint_rmse_rad']=float(np.sqrt(np.mean((refq-np.array(after['joints']))**2)))
                if target is not None:
                    goal = np.asarray(target['targets_world'])[:,0]
                    record['measured_target_m'] = np.linalg.norm(measured-goal,axis=-1).tolist()
                    record['consumed_reference_target_m'] = np.linalg.norm(refp-goal,axis=-1).tolist()
                    record['measured_target_angle_deg'] = [angle_deg(a,b) for a,b in
                        zip(np.asarray(after['wrist_rot']),np.asarray(target['wrist_rotation_world']))]
                    record['after_arrival'] = after['time'] >= target['arrival_time']
            with (run/'execution.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            return action

        provider.get_action, client._post_json = act, post
        (run/'sim-contract.json').write_text(json.dumps({'joint_names': joint_names,
            'body_names': body_names,'wrist_ids':wrist_ids,'joint_ids':joint_ids,
            'task': args.task,'physics_dt':env.physics_dt}, indent=2))
        return provider
    factory.create_action_provider = create
    if os.environ.get('GRASP_LOOP') == '1':
        # The evaluation loop runs while its controller does; remember it so a physical
        # success can end the episode without touching Arena's own reward.
        control = importlib.import_module('layeredcontrol.robot_control_system')
        set_provider = control.RobotController.set_action_provider

        def remember(controller, provider):
            controller.env.kvla_controller = controller
            return set_provider(controller, provider)
        control.RobotController.set_action_provider = remember
