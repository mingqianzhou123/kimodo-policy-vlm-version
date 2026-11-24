"""Experiment A: measured pose history, fixed oracle hand targets, native Kimodo."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from kimodo_vla.core.types import G1_JOINT_NAMES, yaw_matrix, yaw_of
from kimodo_vla.model.features import KimodoFeatures, G1_XML_PATH
from kimodo_vla.model.native_policy import NativeKimodoPolicy
from kimodo_vla.model.planner import load_text_embeddings, _episode_noise_seed
from kimodo_vla.model.sparse_conditions import hand_keyframe
from kimodo_vla.motion.g1_reference import _interpolate_motion, rot6d_row_to_matrix
from .geometry import BASIS, SIDES, Frame, XmlFK, rotation_wxyz, angle_deg
from .continuity import condition_boundary, blend_reference, bounded_hinge_trajectory
from .support import condition_support
from .hand_path import joint_goal_path, execution_frames
from .projection import project_trajectory


class OraclePolicy:
    def __init__(self, checkpoint, text_cache, run, mode, device='cuda', diffusion_steps=50,
                 constraint_cfg=2., text_mode='none', root_anchor='reference',
                 variant='original',handoff_seconds=.2,joint_speed=2.,support_constraints='none',hand_constraints='keyframes',trajectory_projection='none'):
        if variant not in ('original','handoff','bounded'):raise ValueError('Unknown oracle variant')
        if handoff_seconds<=0 or joint_speed<=0:raise ValueError('Positive handoff duration and joint speed required')
        self.variant,self.handoff_seconds,self.joint_speed=variant,handoff_seconds,joint_speed
        if support_constraints not in ('none','root','root-feet','root-feet-keyframes'):
            raise ValueError('Unknown support constraints')
        if support_constraints!='none' and variant!='bounded':
            raise ValueError('Support experiments require the preserved bounded variant')
        if support_constraints!='none' and mode not in ('hold','reach'):
            raise ValueError('Standing support is only defined for hold/reach')
        self.support_constraints=support_constraints
        if hand_constraints not in ('keyframes','continuous'):
            raise ValueError('Unknown hand constraints')
        if hand_constraints!='keyframes' and (variant!='bounded' or support_constraints!='root' or mode not in ('hold','reach')):
            raise ValueError('Continuous experiment requires bounded + root + hold/reach')
        self.hand_constraints=hand_constraints
        if trajectory_projection not in ('none','ik','ik-hold'):
            raise ValueError('Unknown trajectory projection')
        if trajectory_projection!='none' and hand_constraints!='continuous':
            raise ValueError('IK experiment requires continuous hand conditions')
        self.trajectory_projection=trajectory_projection
        shared = getattr(self, 'shared_model', None)  # a subclass may hand over a loaded model (offline sweeps)
        self.model = shared if shared is not None else NativeKimodoPolicy(checkpoint,
            constraint_cfg=constraint_cfg if text_mode == 'none' else None).to(device).eval()
        self.text_mode, self.constraint_cfg = text_mode, constraint_cfg
        self.root_anchor = root_anchor
        self.features = KimodoFeatures(50.)
        self.texts = load_text_embeddings(text_cache)
        self.run = Path(run)
        self.mode, self.steps = mode, diffusion_steps
        self.fk = XmlFK(G1_XML_PATH, G1_JOINT_NAMES)
        self.ids = [[self.features.representation.skeleton.bone_order_names.index(f'{s}_{j}_skel')
                     for j in ('wrist_yaw', 'hand_roll')] for s in SIDES]
        self.reset(None)

    def reset(self, seed=None):
        self.seed = 1000 if seed is None else int(seed)
        self.history, self.plan_index = [], 0
        self.frame = self.targets = None
        self.start_time = None
        return {'status': 'ok'}

    def initialize(self, sample):
        root = np.asarray(sample['root_pos'])
        heading = yaw_matrix(float(yaw_of(rotation_wxyz(sample['root_quat']))))
        self.frame = Frame(root, heading)
        self.start_time = sample['time']
        pose = self.features.codec.decode_joint_configuration_pose(
            np.array([sample['joints']]), self.frame.points([root]) @ BASIS,
            root_rotation_matrices=np.array([heading.T @ rotation_wxyz(sample['root_quat'])]),
            joint_names=G1_JOINT_NAMES)
        gr, pos, _ = self.features.representation.skeleton.fk(pose['local_rot_mats'], pose['root_positions'])
        self.standing_features = self.features.features(pose['local_rot_mats'],pose['root_positions'])[0]
        names = self.features.representation.skeleton.bone_order_names
        self.ankle_ids = [names.index(f'{side}_ankle_roll_skel') for side in SIDES]
        self.foot_names = [f'{side}_ankle_roll_link' for side in SIDES]
        initial_feet,initial_feet_rot = self.fk(sample['joints'],root,rotation_wxyz(sample['root_quat']),
                                               body_names=self.foot_names)
        self.support_target = {'root_world':root.tolist(),'root_rotation_world':rotation_wxyz(sample['root_quat']).tolist(),
            'ankle_world':np.asarray(sample.get('foot_pos',initial_feet)).tolist(),
            'ankle_rotation_world':np.asarray(sample.get('foot_rot',initial_feet_rot)).tolist(),
            'foot_definition':'ankle_roll_link origins; native toe endpoints are virtual, not sole contacts',
            'standing_features':self.standing_features.tolist()}
        skel = self.frame.world(pos[0].numpy()[self.ids])
        actual = np.asarray(sample['wrist_pos'])
        self.targets = skel.copy()
        # Wrist goal is simulator truth; hand-end is a virtual skeleton endpoint,
        # not a claimed Dex3 contact point. Preserve its measured-pose offset.
        self.targets += (actual - skel[:, 0])[:, None, :]
        if self.mode == 'offset':
            self.targets += (heading @ np.array([.05, 0, 0]))[None, None]
        elif self.mode == 'box':
            box = np.asarray(sample['box_pos'])
            for i, sign in enumerate((1, -1)):
                target = box + heading @ np.array([-.06, sign * .20, .08])
                self.targets[i] += target - self.targets[i, 0]
        self.target_rot_model = gr[0].numpy()[np.asarray(self.ids)[:, 0]]
        self.target_rot_world = np.asarray(sample['wrist_rot'])
        witness = None
        if self.mode == 'reach':
            # A nontrivial oracle with an explicit feasible configuration.
            # Unlike translating points by an arbitrary distance, FK certifies
            # the joint-limited kinematic endpoint (not collision/dynamic safety).
            q = np.array(sample['joints'])
            for side in SIDES:
                q[G1_JOINT_NAMES.index(f'{side}_shoulder_pitch_joint')] -= .15
            if not np.all((q>=self.fk.joint_limits[:,0]) & (q<=self.fk.joint_limits[:,1])):
                raise ValueError('Oracle reach configuration exceeds joint limits')
            witness = q.tolist()
            goal_pose = self.features.codec.decode_joint_configuration_pose(q[None],
                self.frame.points([root]) @ BASIS,
                root_rotation_matrices=np.array([heading.T @ rotation_wxyz(sample['root_quat'])]),
                joint_names=G1_JOINT_NAMES)
            goal_gr, goal_pos, _ = self.features.representation.skeleton.fk(
                goal_pose['local_rot_mats'],goal_pose['root_positions'])
            self.targets = self.frame.world(goal_pos[0].numpy()[self.ids])
            self.target_rot_model = goal_gr[0].numpy()[np.asarray(self.ids)[:,0]]
            goal_wrist, self.target_rot_world = self.fk(q,root,rotation_wxyz(sample['root_quat']))
            self.targets += (goal_wrist-self.targets[:,0])[:,None,:]
        self.initial_joints = np.asarray(sample['joints'])
        self.goal_joints = self.initial_joints.copy() if witness is None else np.asarray(witness)
        self.initial_root = root.copy()
        self.initial_rotation = rotation_wxyz(sample['root_quat'])
        self.root_goal = self.frame.points(root)[[0, 2]]
        self.arrival = self.start_time + 1.2
        self.target_info = {'mode': self.mode, 'targets_world': self.targets.tolist(),
            'wrist_rotation_world': self.target_rot_world.tolist(), 'arrival_time': self.arrival,
            'virtual_endpoint': True, 'contact_template_validated': False,
            'initial_skeleton_vs_sim_wrist_m': np.linalg.norm(skel[:, 0]-actual, axis=1).tolist(),
            'witness_joint_pos': witness,
            'goal_displacement_m': np.linalg.norm(self.targets[:,0]-actual,axis=-1).tolist(),
            'origin': self.frame.origin.tolist(), 'heading': heading.tolist()}
        self.target_info['support_target']=self.support_target
        self.target_info['hand_path']='quintic interpolation of legal initial/reach witness joints, fixed episode clock'
        self.target_info['initial_joints']=self.initial_joints.tolist()
        (self.run / 'target.json').write_text(json.dumps(self.target_info, indent=2))

    def path_goals(self, elapsed):
        q=joint_goal_path(self.initial_joints,self.goal_joints,elapsed)
        pose=self.features.codec.decode_joint_configuration_pose(q,
            np.tile(self.frame.points(self.initial_root)@BASIS,(len(q),1)),
            root_rotation_matrices=np.tile(self.frame.heading.T@self.initial_rotation,(len(q),1,1)),
            joint_names=G1_JOINT_NAMES)
        gr,pos,_=self.features.representation.skeleton.fk(pose['local_rot_mats'],pose['root_positions'])
        return pos.numpy()[:,self.ids],gr.numpy()[:,np.asarray(self.ids)[:,0]]

    @torch.inference_mode()
    def __call__(self, request):
        samples = request['observation']['oracle_history']
        if not samples:
            raise ValueError('Measured oracle history is required')
        if self.frame is None:
            self.initialize(samples[0])
        if self.history and samples[0]['time'] <= self.history[-1]['time']:
            raise ValueError('Non-monotonic oracle history')
        self.history.extend(samples)
        self.history = self.history[-180:]
        obs = samples[-1]
        t = obs['time'] - self.start_time
        step = int(round(t * 50))
        if len(self.history) < 6:
            rot = self.frame.heading.T @ rotation_wxyz(obs['root_quat'])
            action = np.r_[0., 0., obs['root_pos'][2], rot[:, :2].ravel(), obs['joints'], 0., 0.]
            return {'action_chunk': np.tile(action, (5, 1)).tolist(), 'oracle_target': self.target_info}
        joints = np.asarray([x['joints'] for x in self.history])
        roots_world = np.asarray([x['root_pos'] for x in self.history])
        roots = (roots_world-self.frame.origin) @ self.frame.heading
        rotations = np.stack([self.frame.heading.T @ rotation_wxyz(x['root_quat']) for x in self.history])
        first_step = int(round((self.history[0]['time']-self.start_time)*50))
        local, root = self.features.motion(joints, roots, rotations, first_step)
        features = self.features.features(local, root)
        hist = features[:-1][-100:]
        hist_mask = torch.ones_like(hist, dtype=torch.bool)
        cut = round(t*30)
        arrival_frame = round((self.arrival-self.start_time)*30)-cut
        keyframes = [arrival_frame] if arrival_frame > 0 else [0, 15, 30, 49]
        if self.variant!='original' and arrival_frame<=0:keyframes=[15,30,49]
        if not all(0 <= x < 50 for x in keyframes):
            raise ValueError(f'Target outside canvas: {keyframes}')
        values = torch.zeros(50, 417)
        mask = torch.zeros_like(values, dtype=torch.bool)
        positions = self.frame.points(self.targets)
        for keyframe in keyframes:
            v, m = hand_keyframe(self.model.representation, frames=50, frame=keyframe,
                smooth_root_xz=self.root_goal, hand_positions=dict(zip(SIDES, positions)),
                wrist_rotations=dict(zip(SIDES, self.target_rot_model)))
            values[m], mask[m] = v[m], True
        dense_frames=[]
        if self.hand_constraints=='continuous':
            dense_frames=execution_frames(step,cut)
            path_pos,path_rot=self.path_goals((cut+np.array(dense_frames))/30)
            for k,pp,rr in zip(dense_frames,path_pos,path_rot):
                v,m=hand_keyframe(self.model.representation,frames=50,frame=k,
                    smooth_root_xz=self.root_goal,hand_positions=dict(zip(SIDES,pp)),
                    wrist_rotations=dict(zip(SIDES,rr)))
                values[m],mask[m]=v[m],True
        if self.variant!='original':
            condition_boundary(hist_mask,values,mask,features[-1],self.model.representation)
        support_mask=condition_support(values,mask,self.standing_features,self.model.representation,
                                       self.support_constraints,keyframes=keyframes)
        text_key = str(request['task']).strip()
        text = self.texts[text_key] if self.text_mode == 'task' else torch.zeros(1,4096)
        generator = torch.Generator(device=self.model.device).manual_seed(_episode_noise_seed(self.seed, self.plan_index))
        begin = time.perf_counter()
        output = self.model.predict_future(text_key, torch.zeros(1, 3, 1, 1), hist,
            history_feature_mask=hist_mask, text_feat=text.to(self.model.device),
            future_condition=values, future_condition_mask=mask,
            diffusion_steps=self.steps, generator=generator)
        elapsed = time.perf_counter()-begin
        output = {k: v.cpu() if torch.is_tensor(v) else v for k, v in output.items()}
        k = keyframes[0]
        fk_model = output['posed_joints'].numpy()[:, self.ids]
        feature_pos = self.model.representation.inverse(output['motion_features'].to(self.model.device)[None],
            is_normalized=False, posed_joints_from='positions')['posed_joints'][0].cpu().numpy()[:, self.ids]
        fk_world = self.frame.world(fk_model)
        raw_world = self.frame.world(feature_pos)
        # Exact control grid; send 0.5 s, retaining full 50-frame prediction for audit.
        offset = step*.6-cut
        times = offset+torch.arange(26)*.6
        before = output['history_last_root_position'].reshape(1, 3)
        cr, cp = _interpolate_motion(torch.cat((output['local_rot_mats'][:1], output['local_rot_mats'])),
            torch.cat((before, output['root_positions'])), times+1)
        # Arena integrates deltas onto its last consumed reference, not onto
        # this new prediction's frame zero. Preserve the absolute goal gauge.
        anchor = (torch.as_tensor(self.frame.points(obs['reference_root_pos']),dtype=cp.dtype)
                  if self.root_anchor == 'reference' else cp[0])
        actions = self.features.codec.encode(cr[1:], cp[1:], previous_root_position=anchor,
                                            hand_binary=torch.zeros(25, 2)).numpy()
        raw_actions=actions.copy();raw_root=cp[1:].clone()
        _,raw_pose_all,_=self.features.representation.skeleton.fk(cr[1:],cp[1:])
        raw_pose=self.frame.world(raw_pose_all.numpy()[:,np.asarray(self.ids)[:,0]])
        raw_feet=self.frame.world(raw_pose_all.numpy()[:,self.ankle_ids])
        sent_root=raw_root.clone();handoff_actions=raw_actions.copy()
        if self.variant!='original':
            boundary=self.features.codec.decode_joint_configuration_pose(np.array([obs['reference_joints']]),
                np.array([self.frame.points(obs['reference_root_pos'])@BASIS]),
                root_rotation_matrices=np.array([self.frame.heading.T@rotation_wxyz(obs['reference_root_quat'])]),
                joint_names=G1_JOINT_NAMES)
            sent_local,sent_root=blend_reference(cr[1:],cp[1:],boundary['local_rot_mats'][0],
                boundary['root_positions'][0],np.arange(1,26)/50,self.handoff_seconds)
            handoff_actions=self.features.codec.encode(sent_local,sent_root,
                previous_root_position=boundary['root_positions'][0],hand_binary=torch.zeros(25,2)).numpy()
            actions=handoff_actions.copy()
            if self.variant=='bounded':
                actions[:,9:38]=bounded_hinge_trajectory(sent_local,self.features.codec,self.fk.joint_limits,
                    obs['reference_joints'],self.joint_speed)
        def wrist_fk(action,model_root):
            rotation=rot6d_row_to_matrix(torch.from_numpy(action[:,3:9])).numpy()
            return np.asarray([self.fk(a[9:38],self.frame.world(p),self.frame.heading@r)[0]
                               for a,p,r in zip(action,model_root.numpy(),rotation)])
        raw_projected=wrist_fk(raw_actions,raw_root)
        handoff_projected=wrist_fk(handoff_actions,sent_root)
        pre_ik_actions=actions.copy();pre_ik_root=sent_root.clone()
        pre_ik_wrist=wrist_fk(actions,sent_root)
        projection_info=None
        if self.trajectory_projection!='none':
            goal_pos,goal_rot=self.path_goals(t+np.arange(1,26)/50)
            goal_world=self.frame.world(goal_pos[:,:,0])
            goal_rot_world=self.frame.world_to_model.T@goal_rot@BASIS
            hold_options={}
            if self.trajectory_projection=='ik-hold':
                prev_pos,prev_rot=self.path_goals(np.array([t]))
                hold_options=dict(stabilize_hold=True,previous_hands=self.frame.world(prev_pos[0,:,0]),
                    previous_hand_rotations=self.frame.world_to_model.T@prev_rot[0]@BASIS)
            projected_q,projection_info=project_trajectory(self.fk,actions[:,9:38],obs['reference_joints'],
                self.initial_root,self.initial_rotation,goal_world,goal_rot_world,
                np.array(self.support_target['ankle_world']),np.array(self.support_target['ankle_rotation_world']),
                speed=self.joint_speed,**hold_options)
            projected_pose=self.features.codec.decode_joint_configuration_pose(projected_q,
                np.tile(self.frame.points(self.initial_root)@BASIS,(25,1)),
                root_rotation_matrices=np.tile(self.frame.heading.T@self.initial_rotation,(25,1,1)),
                joint_names=G1_JOINT_NAMES)
            sent_root=projected_pose['root_positions']
            actions=self.features.codec.encode(projected_pose['local_rot_mats'],sent_root,
                previous_root_position=boundary['root_positions'][0],hand_binary=torch.zeros(25,2)).numpy()
            # Preserve the bounded scalar branch exactly rather than re-extract it.
            actions[:,9:38]=projected_q
        sent_projected=wrist_fk(actions,sent_root)
        def support_fk(action,model_root):
            rotation=rot6d_row_to_matrix(torch.from_numpy(action[:,3:9])).numpy()
            world_rot=self.frame.heading@rotation
            feet=np.asarray([self.fk(a[9:38],self.frame.world(p),r,body_names=self.foot_names)[0]
                             for a,p,r in zip(action,model_root.numpy(),world_rot)])
            return feet,world_rot
        raw_projected_feet,raw_root_rot=support_fk(raw_actions,raw_root)
        sent_feet,sent_root_rot=support_fk(actions,sent_root)
        command_rows=[{'plan':self.plan_index,'action_index':i,'time':obs['time']+(i+1)/50,
            'variant':self.variant,'raw_model_wrist':raw_pose[i].tolist(),
            'raw_projected_wrist':raw_projected[i].tolist(),'handoff_wrist':handoff_projected[i].tolist(),
            'sent_wrist':sent_projected[i].tolist(),'pre_ik_wrist':pre_ik_wrist[i].tolist(),
            'trajectory_projection':self.trajectory_projection,'pre_ik_joints':pre_ik_actions[i,9:38].tolist(),'sent_joints':actions[i,9:38].tolist(),
            'raw_joints':raw_actions[i,9:38].tolist(),'handoff_joints':handoff_actions[i,9:38].tolist(),
            'sent_root_world':self.frame.world(sent_root[i].numpy()).tolist(),
            'raw_root_world':self.frame.world(raw_root[i].numpy()).tolist(),
            'raw_root_rotation_world':raw_root_rot[i].tolist(),'sent_root_rotation_world':sent_root_rot[i].tolist(),
            'raw_model_feet':raw_feet[i].tolist(),'raw_projected_feet':raw_projected_feet[i].tolist(),
            'sent_feet':sent_feet[i].tolist(),
            'boundary_joints':obs.get('reference_joints',obs['joints'])} for i in range(25)]
        with (self.run/'commands.jsonl').open('a') as f:f.write(''.join(json.dumps(x)+'\n' for x in command_rows))
        # Independent XML FK after projection to the 29 one-axis joint angles.
        all_actions = self.features.codec.encode(output['local_rot_mats'], output['root_positions'],
            previous_root_position=before[0], hand_binary=torch.zeros(50, 2)).numpy()
        rots = rot6d_row_to_matrix(torch.from_numpy(all_actions[:, 3:9])).numpy()
        projected_pos, projected_rot = [], []
        for a, p, r in zip(all_actions, output['root_positions'].numpy(), rots):
            pp, rr = self.fk(a[9:38], self.frame.world(p), self.frame.heading @ r)
            projected_pos.append(pp); projected_rot.append(rr)
        projected_pos, projected_rot = np.array(projected_pos), np.array(projected_rot)
        record = {'plan': self.plan_index, 'time': obs['time'], 'cut': cut,
            'keyframes': keyframes,'dense_hand_frames':dense_frames,'hand_constraints':self.hand_constraints, 'inference_s': elapsed, 'text_key': text_key,
            'text_mode': self.text_mode, 'constraint_cfg': self.constraint_cfg,
            'diffusion_steps': self.steps,'trajectory_projection':self.trajectory_projection,'projection':projection_info,
            'variant':self.variant,'history_condition_frames':int(hist_mask.any(-1).sum()),
            'support_constraints':self.support_constraints,'support_channels':int(support_mask.sum()),
            'handoff_seconds':self.handoff_seconds,'joint_speed_rad_s':self.joint_speed,
            'root_anchor': self.root_anchor,
            'root_boundary_correction_m': float(torch.linalg.vector_norm(anchor-cp[0])),
            'coordinate_roundtrip_m': float(np.max(np.abs(self.frame.world(positions)-self.targets))),
            'feature_target_m': np.linalg.norm(raw_world[k]-self.targets, axis=-1).tolist(),
            'kimodo_fk_target_m': np.linalg.norm(fk_world[k]-self.targets, axis=-1).tolist(),
            'projected_wrist_target_m': np.linalg.norm(projected_pos[k]-self.targets[:, 0], axis=-1).tolist(),
            'projection_wrist_delta_m': np.linalg.norm(projected_pos[k]-fk_world[k, :, 0], axis=-1).tolist(),
            'projected_wrist_angle_deg': [angle_deg(a,b) for a,b in zip(projected_rot[k], self.target_rot_world)],
            'measured_wrist_target_m': np.linalg.norm(np.asarray(obs['wrist_pos'])-self.targets[:,0], axis=-1).tolist()}
        with (self.run / 'plans.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        np.savez_compressed(self.run / f'plan-{self.plan_index:03d}.npz',
            motion_features=output['motion_features'].numpy(), conditions=values.numpy(), mask=mask.numpy(),
            history=hist.numpy(), raw_positions_world=raw_world, kimodo_fk_world=fk_world,
            projected_wrist_world=projected_pos, actions=actions,
            raw_actions=raw_actions,handoff_actions=handoff_actions,pre_ik_actions=pre_ik_actions,
            pre_ik_root_model=pre_ik_root.numpy(),history_mask=hist_mask.numpy(),
            sent_root_model=sent_root.numpy(),raw_root_model=raw_root.numpy(),
            support_mask=support_mask.numpy(),standing_features=self.standing_features.numpy(),
            raw_future_root_world=self.frame.world(output['root_positions'].numpy()),
            raw_future_ankle_world=self.frame.world(output['posed_joints'].numpy()[:,self.ankle_ids]),
            projected_wrist_rotation=projected_rot)
        self.plan_index += 1
        return {'action_chunk': actions.tolist(), 'oracle_target': self.target_info, 'oracle_plan': record}
