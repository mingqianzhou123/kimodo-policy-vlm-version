"""Goals that move the whole body: a timed ground path for the root, optionally one hand target.

The standing backend (``OraclePolicy``) pins root and feet to the starting stance and repairs the
arms with an IK that assumes neither moves. Here the root follows a ground path, the feet are
free, and nothing repairs Kimodo's output: the reference sent to the tracker is Kimodo's motion
after the usual handoff blend and joint bounds.

A goal is written in the start frame: x forward, y left, z up from the floor, origin on the floor
under the pelvis at the first measured sample. Times are seconds from that sample.

    {"name": "forward-0.5",
     "root": {"xy": [0.5, 0.0], "start_s": 0.5, "duration_s": 2.5},
     "hand": {"side": "right", "position": [0.9, -0.2, 0.95], "approach": [1, 0, 0],
              "start_s": 3.5, "duration_s": 1.5}}

``root`` moves the ground point the pelvis hovers over while the facing direction stays as it
started, so a sideways or backward ``xy`` asks for a side step or a backward walk, not a turn.
``hand`` moves one wrist to a world point; ``approach`` is the direction the forearm points on
arrival. Either part may be left out.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.model.planner import _episode_noise_seed
from kimodo_vla.model.sparse_conditions import hand_keyframe
from kimodo_vla.motion.g1_reference import _interpolate_motion, rot6d_row_to_matrix
from .continuity import blend_reference, bounded_hinge_trajectory
from .geometry import BASIS, SIDES, angle_deg, rotation_wxyz
from .hand_path import execution_frames
from .policy import OraclePolicy

HISTORY_SOURCES = ('reference', 'measured', 'measured-last')
CANVAS = 50  # future frames Kimodo generates per plan, at 30 Hz
ROOT_CONDITIONS = ('path', 'waypoints')


def ease(u):
    """Minimum-jerk progress: 0 at u<=0, 1 at u>=1, zero velocity and acceleration at both ends."""
    u = np.clip(np.asarray(u, dtype=float), 0., 1.)
    return u**3 * (10 + u*(-15 + 6*u))


def approach_rotation(approach, side):
    """Wrist-link rotation whose x axis (along the forearm, toward the hand) is ``approach``.

    The link's y axis stays as close to world-left as the approach allows, which is the link's
    orientation with the forearm level and the palm turned inward. ``side`` is unused: both wrist
    links share that convention.
    """
    x = np.asarray(approach, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all() or np.linalg.norm(x) < 1e-6:
        raise ValueError('approach must be a non-zero direction')
    x = x/np.linalg.norm(x)
    y = np.array([0., 1., 0.]) - x[1]*x
    if np.linalg.norm(y) < 1e-3:
        raise ValueError('approach along the left-right axis leaves the palm direction undefined')
    y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def load_goal(source):
    """Validate a goal file or dict and fill in defaults; returns a plain dict."""
    goal = json.loads(Path(source).read_text()) if not isinstance(source, dict) else dict(source)
    unknown = set(goal) - {'name', 'root', 'hand', 'note'}
    if unknown:
        raise ValueError(f'Unknown goal fields: {sorted(unknown)}')
    if not goal.get('name'):
        raise ValueError('A goal needs a name')
    out = {'name': str(goal['name'])}
    if 'root' in goal:
        root = dict(goal['root'])
        xy = np.asarray(root.pop('xy'), dtype=float)
        start, duration = float(root.pop('start_s', .5)), float(root.pop('duration_s'))
        if root or xy.shape != (2,) or not np.isfinite(xy).all() or start < 0 or duration <= 0:
            raise ValueError('root needs xy [forward, left], duration_s > 0 and start_s >= 0')
        out['root'] = {'xy': xy.tolist(), 'start_s': start, 'duration_s': duration}
    if 'hand' in goal:
        hand = dict(goal['hand'])
        side = hand.pop('side')
        position = np.asarray(hand.pop('position'), dtype=float)
        approach = np.asarray(hand.pop('approach'), dtype=float)
        start, duration = float(hand.pop('start_s')), float(hand.pop('duration_s'))
        if hand or side not in SIDES or position.shape != (3,) or not np.isfinite(position).all() \
                or start < 0 or duration <= 0:
            raise ValueError('hand needs side, position [forward, left, up], approach, start_s and duration_s')
        approach_rotation(approach, side)
        out['hand'] = {'side': side, 'position': position.tolist(), 'approach': approach.tolist(),
                       'start_s': start, 'duration_s': duration}
    if len(out) == 1:
        raise ValueError('A goal needs a root path, a hand target, or both')
    return out


def root_progress(goal, elapsed):
    """Fraction of the root path covered at ``elapsed`` seconds; 0 without a root path."""
    root = goal.get('root')
    if root is None:
        return np.zeros_like(np.asarray(elapsed, dtype=float))
    return ease((np.asarray(elapsed, dtype=float)-root['start_s'])/root['duration_s'])


def end_time(goal):
    """When the last commanded motion ends, in seconds from the start."""
    return max(part['start_s']+part['duration_s'] for key, part in goal.items() if key in ('root', 'hand'))


class LocomotionPolicy(OraclePolicy):
    def __init__(self, *args, motion_goal, history_source='reference', noise_seed=None, sent_steps=25,
                 root_condition='path', **kwargs):
        if history_source not in HISTORY_SOURCES:
            raise ValueError('Unknown history source')
        if root_condition not in ROOT_CONDITIONS:
            raise ValueError('Unknown root condition')
        if not 5 <= sent_steps <= 80:
            raise ValueError('A plan covers 1.63 s; send between 5 and 80 control steps of it')
        self.sent = int(sent_steps)  # 50 Hz actions executed before the next plan
        # 'path' conditions the root's ground point on every future frame; 'waypoints' only on
        # frames 15/30/49, leaving the timing of steps between them to Kimodo.
        self.root_condition = root_condition
        self.goal = load_goal(motion_goal)
        self.history_source = history_source
        self.noise_seed = noise_seed  # repeats of one goal differ only in Kimodo's sampling noise
        super().__init__(*args, **kwargs)
        if (self.mode != 'hold' or self.variant != 'bounded' or self.support_constraints != 'none'
                or self.trajectory_projection != 'none'):
            raise ValueError('Moving goals use the bounded handoff with no standing support and no IK repair')

    def reset(self, seed=None):
        self.hand_start = None
        status = super().reset(seed)
        if self.noise_seed is not None:
            self.seed = int(self.noise_seed)
        return status

    def initialize(self, sample):
        super().initialize(sample)  # measured start frame, standing features, both wrists held
        heading, origin = self.frame.heading, self.frame.origin
        self.root_start = np.asarray(self.root_goal, dtype=float)
        self.root_end = self.root_start.copy()
        if 'root' in self.goal:
            forward, left = self.goal['root']['xy']
            self.root_end = self.frame.points(self.initial_root + heading @ np.array([forward, left, 0.]))[[0, 2]]
        self.heading_feature = self.standing_features[self.model.representation.slice_dict['global_root_heading']]
        info = dict(mode='motion-goal', target_source='motion_goal', motion_goal=self.goal,
                    history_source=self.history_source, noise_seed=self.seed, active_sides=[],
                    root_goal_world=self.frame.world([self.root_end[0], 0., self.root_end[1]])[:2].tolist(),
                    arrival_time=self.start_time+end_time(self.goal), witness_joint_pos=None,
                    hand_path='none')
        if 'hand' in self.goal:
            hand = self.goal['hand']
            i = SIDES.index(hand['side'])
            # The virtual hand-end point keeps its measured offset in the wrist link's frame.
            tip = np.asarray(sample['wrist_rot'])[i].T @ (self.targets[i, 1]-self.targets[i, 0])
            rotation = heading @ approach_rotation(hand['approach'], hand['side'])
            wrist = origin + heading @ np.asarray(hand['position'])
            self.hand_side, self.hand_index = hand['side'], i
            self.hand_goal, self.hand_goal_rotation, self.hand_tip = wrist, rotation, tip
            self.targets[i] = [wrist, wrist + rotation @ tip]
            self.target_rot_world = np.asarray(self.target_rot_world).copy()
            self.target_rot_world[i] = rotation
            self.target_rot_model = self.frame.rotations(self.target_rot_world)
            info.update(active_sides=[hand['side']], targets_world=self.targets.tolist(),
                        wrist_rotation_world=self.target_rot_world.tolist(),
                        goal_displacement_m=np.linalg.norm(self.targets[:, 0]-sample['wrist_pos'], axis=1).tolist(),
                        hand_path='minimum-jerk world line and rotation from the wrist measured when the hand '
                                  'phase begins; the other hand is unconstrained')
        self.arrival = info['arrival_time']
        self.target_info.update(info)
        (self.run/'target.json').write_text(json.dumps(self.target_info, indent=2))

    def root_xz(self, elapsed):
        """Model-frame ground point of the root path at ``elapsed`` seconds: [..., 2]."""
        w = np.asarray(root_progress(self.goal, elapsed))[..., None]
        return self.root_start + w*(self.root_end-self.root_start)

    def hand_pose(self, elapsed):
        """World wrist position [N, 3] and rotation [N, 3, 3] of the hand path at ``elapsed`` seconds."""
        hand = self.goal['hand']
        start_pos, start_rot = self.hand_start
        w = ease((np.asarray(elapsed, dtype=float)-hand['start_s'])/hand['duration_s'])
        position = start_pos + w[:, None]*(self.hand_goal-start_pos)
        turn = Slerp([0., 1.], Rotation.from_matrix(np.stack([start_rot, self.hand_goal_rotation])))
        return position, turn(w).as_matrix()

    def pose_features(self, source):
        """Kimodo features of the stored 50 Hz samples, from measured or consumed-reference poses."""
        keys = (('reference_joints', 'reference_root_pos', 'reference_root_quat') if source == 'reference'
                else ('joints', 'root_pos', 'root_quat'))
        joints = np.asarray([x[keys[0]] for x in self.history])
        roots = (np.asarray([x[keys[1]] for x in self.history])-self.frame.origin) @ self.frame.heading
        rotations = np.stack([self.frame.heading.T @ rotation_wxyz(x[keys[2]]) for x in self.history])
        first_step = int(round((self.history[0]['time']-self.start_time)*50))
        return self.features.features(*self.features.motion(joints, roots, rotations, first_step))

    def conditions(self, elapsed, step, cut, obs):
        """Sparse future conditions for one plan: (values, mask, hand keyframes, dense hand frames)."""
        representation = self.model.representation
        values = torch.zeros(CANVAS, representation.motion_rep_dim)
        mask = torch.zeros_like(values, dtype=torch.bool)
        frames = np.arange(1, CANVAS)
        path = self.root_xz((cut+frames)/30)
        root = representation.slice_dict['smooth_root_pos'].start
        heading = representation.slice_dict['global_root_heading']
        for k, xz in zip(frames, path):
            if self.root_condition == 'waypoints' and k not in (15, 30, 49):
                continue
            values[k, root], values[k, root+2] = float(xz[0]), float(xz[1])
            mask[k, root] = mask[k, root+2] = True
            values[k, heading], mask[k, heading] = self.heading_feature, True
        keyframes, dense = [], []
        hand = self.goal.get('hand')
        if hand is not None and elapsed >= hand['start_s']-1e-9:
            if self.hand_start is None:
                i = self.hand_index
                self.hand_start = (np.asarray(obs['wrist_pos'])[i], np.asarray(obs['wrist_rot'])[i])
                self.target_info['hand_start'] = {'time': obs['time'], 'wrist_world': self.hand_start[0].tolist(),
                                                  'wrist_rotation_world': self.hand_start[1].tolist()}
                (self.run/'target.json').write_text(json.dumps(self.target_info, indent=2))
            arrival = round((hand['start_s']+hand['duration_s'])*30)-cut
            keyframes = [arrival] if 0 < arrival < CANVAS else [] if arrival > 0 else [15, 30, 49]
            dense = execution_frames(step, cut, self.sent)
            for k in sorted(set(keyframes) | set(dense)):
                position, rotation = self.hand_pose(np.array([(cut+k)/30]))
                wrist = self.frame.points(np.stack([position[0], position[0] + rotation[0] @ self.hand_tip]))
                v, m = hand_keyframe(representation, frames=CANVAS, frame=int(k), smooth_root_xz=path[k-1],
                                     hand_positions={self.hand_side: wrist},
                                     wrist_rotations={self.hand_side: self.frame.rotations(rotation[0])})
                values[m], mask[m] = v[m], True
        return values, mask, keyframes, dense

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
        t = obs['time']-self.start_time
        step = int(round(t*50))
        if len(self.history) < 6:
            rot = self.frame.heading.T @ rotation_wxyz(obs['root_quat'])
            action = np.r_[0., 0., obs['root_pos'][2], rot[:, :2].ravel(), obs['joints'], 0., 0.]
            return {'action_chunk': np.tile(action, (5, 1)).tolist(), 'oracle_target': self.target_info}
        representation = self.model.representation
        features = self.pose_features('reference' if self.history_source == 'reference' else 'measured')
        hist = features[:-1][-100:]
        hist_mask = torch.ones_like(hist, dtype=torch.bool)
        if self.history_source == 'measured-last':
            hist_mask[:-1] = False
        cut = round(t*30)
        values, mask, keyframes, dense = self.conditions(t, step, cut, obs)
        # Frame zero is now: its pose is known, its velocities and contacts are left to the model.
        pose_end = representation.slice_dict['global_rot_data'].stop
        values[0, :pose_end], mask[0, :pose_end] = features[-1, :pose_end], True
        generator = torch.Generator(device=self.model.device).manual_seed(_episode_noise_seed(self.seed, self.plan_index))
        begin = time.perf_counter()
        output = self.model.predict_future(str(request['task']).strip(), torch.zeros(1, 3, 1, 1), hist,
            history_feature_mask=hist_mask, text_feat=torch.zeros(1, 4096).to(self.model.device),
            future_condition=values, future_condition_mask=mask,
            diffusion_steps=self.steps, generator=generator)
        elapsed = time.perf_counter()-begin
        output = {k: v.cpu() if torch.is_tensor(v) else v for k, v in output.items()}

        # Kimodo's 30 Hz frames on the 50 Hz control grid: the steps sent from this plan.
        times = step*.6-cut+torch.arange(self.sent+1)*.6
        before = output['history_last_root_position'].reshape(1, 3)
        local, root = _interpolate_motion(torch.cat((output['local_rot_mats'][:1], output['local_rot_mats'])),
                                          torch.cat((before, output['root_positions'])), times+1)
        raw_local, raw_root = local[1:], root[1:]
        zero_hands = torch.zeros(self.sent, 2)
        boundary = self.features.codec.decode_joint_configuration_pose(np.array([obs['reference_joints']]),
            np.array([self.frame.points(obs['reference_root_pos']) @ BASIS]),
            root_rotation_matrices=np.array([self.frame.heading.T @ rotation_wxyz(obs['reference_root_quat'])]),
            joint_names=G1_JOINT_NAMES)
        boundary_root = boundary['root_positions'][0]
        raw_actions = self.features.codec.encode(raw_local, raw_root, previous_root_position=boundary_root,
                                                 hand_binary=zero_hands).numpy()
        sent_local, sent_root = blend_reference(raw_local, raw_root, boundary['local_rot_mats'][0], boundary_root,
                                                np.arange(1, self.sent+1)/50, self.handoff_seconds)
        actions = self.features.codec.encode(sent_local, sent_root, previous_root_position=boundary_root,
                                             hand_binary=zero_hands).numpy()
        actions[:, 9:38] = bounded_hinge_trajectory(sent_local, self.features.codec, self.fk.joint_limits,
                                                    obs['reference_joints'], self.joint_speed)

        def world_pose(action, model_root):
            """World root rotation, wrist positions and rotations of each action's joint reference."""
            rotation = self.frame.heading @ rot6d_row_to_matrix(torch.from_numpy(action[:, 3:9])).numpy()
            wrists = [self.fk(a[9:38], self.frame.world(p), r) for a, p, r in zip(action, model_root.numpy(), rotation)]
            return rotation, np.array([w[0] for w in wrists]), np.array([w[1] for w in wrists])
        raw_rotation, raw_wrist, _ = world_pose(raw_actions, raw_root)
        sent_rotation, sent_wrist, _ = world_pose(actions, sent_root)
        sent_feet = np.array([self.fk(a[9:38], self.frame.world(p), r, body_names=self.foot_names)[0]
                              for a, p, r in zip(actions, sent_root.numpy(), sent_rotation)])
        raw_feet = np.array([self.fk(a[9:38], self.frame.world(p), r, body_names=self.foot_names)[0]
                             for a, p, r in zip(raw_actions, raw_root.numpy(), raw_rotation)])
        rows = [{'plan': self.plan_index, 'action_index': i, 'time': obs['time']+(i+1)/50,
                 'raw_projected_wrist': raw_wrist[i].tolist(), 'sent_wrist': sent_wrist[i].tolist(),
                 'raw_joints': raw_actions[i, 9:38].tolist(), 'sent_joints': actions[i, 9:38].tolist(),
                 'raw_root_world': self.frame.world(raw_root[i].numpy()).tolist(),
                 'sent_root_world': self.frame.world(sent_root[i].numpy()).tolist(),
                 'raw_root_rotation_world': raw_rotation[i].tolist(),
                 'sent_root_rotation_world': sent_rotation[i].tolist(),
                 'raw_feet': raw_feet[i].tolist(), 'sent_feet': sent_feet[i].tolist()} for i in range(self.sent)]
        with (self.run/'commands.jsonl').open('a') as f:
            f.write(''.join(json.dumps(x)+'\n' for x in rows))

        # The whole 50-frame prediction, decoded the way analyze_prediction.py decodes it.
        all_actions = self.features.codec.encode(output['local_rot_mats'], output['root_positions'],
            previous_root_position=before[0], hand_binary=torch.zeros(CANVAS, 2)).numpy()
        _, future_wrist, future_wrist_rotation = world_pose(all_actions, output['root_positions'])
        future_root = self.frame.world(output['root_positions'].numpy())
        root_frames = list(range(1, CANVAS)) if self.root_condition == 'path' else [15, 30, 49]
        path_world = self.frame.world(np.insert(self.root_xz((cut+np.arange(CANVAS))/30), 1, 0., axis=1))[:, :2]
        record = {'plan': self.plan_index, 'time': obs['time'], 'cut': cut, 'keyframes': keyframes,
                  'dense_hand_frames': dense, 'root_frames': root_frames, 'hand_constraints': 'continuous',
                  'history_source': self.history_source, 'sent_steps': self.sent, 'root_condition': self.root_condition, 'history_condition_frames': int(hist_mask.any(-1).sum()),
                  'inference_s': elapsed, 'constraint_cfg': self.constraint_cfg, 'diffusion_steps': self.steps,
                  'variant': self.variant, 'handoff_seconds': self.handoff_seconds,
                  'joint_speed_rad_s': self.joint_speed, 'support_constraints': 'none', 'trajectory_projection': 'none',
                  'root_path_progress': float(root_progress(self.goal, t)),
                  'root_boundary_correction_m': float(torch.linalg.vector_norm(boundary_root-root[0])),
                  # How far Kimodo's own output is from what it was asked for, before any execution.
                  'root_path_error_m': np.linalg.norm(future_root[1:, :2]-path_world[1:], axis=1).max().item(),
                  'measured_root_path_m': float(np.linalg.norm(np.asarray(obs['root_pos'])[:2]-path_world[0]))}
        if keyframes or dense:
            i, k = self.hand_index, (keyframes or dense)[-1]
            goal_position, goal_rotation = self.hand_pose(np.array([(cut+k)/30]))
            record.update(hand_frame=int(k),
                kimodo_wrist_goal_m=float(np.linalg.norm(future_wrist[k, i]-goal_position[0])),
                kimodo_wrist_goal_deg=angle_deg(future_wrist_rotation[k, i], goal_rotation[0]),
                measured_wrist_target_m=float(np.linalg.norm(np.asarray(obs['wrist_pos'])[i]-self.hand_goal)))
        with (self.run/'plans.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        np.savez_compressed(self.run/f'plan-{self.plan_index:03d}.npz',
            motion_features=output['motion_features'].numpy(), conditions=values.numpy(), mask=mask.numpy(),
            history=hist.numpy(), history_mask=hist_mask.numpy(), actions=actions, raw_actions=raw_actions,
            sent_root_model=sent_root.numpy(), raw_root_model=raw_root.numpy(),
            projected_wrist_world=future_wrist, projected_wrist_rotation=future_wrist_rotation,
            raw_future_root_world=future_root, root_path_world=path_world)
        self.plan_index += 1
        return {'action_chunk': actions.tolist(), 'oracle_target': self.target_info, 'oracle_plan': record}
