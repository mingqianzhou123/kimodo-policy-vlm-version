"""Goals that move the whole body: walk the root to a ground point, then optionally reach one hand.

The standing backend (``OraclePolicy``) pins root and feet to the starting stance and repairs the
arms with an IK that assumes neither moves. Here the feet are free while walking and nothing
repairs Kimodo's output: the reference sent to the tracker is Kimodo's motion after the usual
handoff blend and joint bounds.

SONIC is given joint angles and body orientation only, never the root's position, so the body
ends up wherever the generated steps physically carry it. Position is therefore closed here:
every walking plan starts from where the body was measured and heads for the goal from there.
An episode runs through these phases:

  ready  0.5 s standing on the settled start pose
  walk   plans of ``walk_steps`` control steps, each a root ground path from the measured position
         toward the goal, until the measured root is within tolerance or the time allowance ends
  stop   one plan, executed whole, that ends in the starting standing pose where the body now is
         (holding a mid-stride pose instead sent the robot running 3 m in one early run)
  reach  0.5 s plans with the pelvis held at that stance (position, rotation, zero velocity: the
         standing backend's "root" support) and one wrist on a path to its target; the target stays
         conditioned until the episode ends. Legs are not conditioned and the trunk is free, so the
         reach is arm plus waist. Leaving the pelvis free instead made every plan a different
         motion: in one near reach the reference's feet travelled 19 m in 3.5 s.
  lean   (lean goals) one plan from the start stance that ends in the leaned pose
  hold   (goals without a hand) the last reference pose, repeated

A goal is written in the start frame: x forward, y left, z up from the floor, origin on the floor
under the pelvis at the first measured sample.

    {"name": "reach-far", "root": {"xy": [0.6, 0.0], "speed_m_s": 0.35},
     "hand": {"side": "right", "position": [0.95, -0.2, 0.95], "approach": [1, 0, 0], "duration_s": 1.5}}

``root`` moves the ground point under the pelvis while the facing direction stays as it started,
so a sideways or backward ``xy`` asks for a side step or a backward walk, not a turn. ``hand``
moves one wrist to a point; ``approach`` is the direction the forearm points on arrival. Either
part may be left out. A third kind of goal stands still and leans the trunk forward:

    {"name": "lean-30", "lean": {"pitch_deg": 30, "by": "waist"}}

Its end pose is the start pose with the trunk pitched forward and the feet where they were, led by
the waist joint or, with "by": "hips", by the pelvis; Kimodo generates the way there.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.model.planner import _episode_noise_seed
from kimodo_vla.model.sparse_conditions import hand_keyframe
from kimodo_vla.motion.g1_reference import _interpolate_motion, rot6d_row_to_matrix
from .continuity import blend_reference, bounded_hinge_trajectory
from .geometry import BASIS, SIDES, angle_deg, rotation_wxyz
from .hand_path import execution_frames
from .policy import OraclePolicy
from .support import condition_support

CANVAS = 50        # future frames Kimodo generates per plan, at 30 Hz
READY_S = .5       # standing time before the first plan
REACH_STEPS = 25   # control steps executed from each reach plan
STAND_FROM = 30    # a stop plan is the standing pose from this frame on
STOP_STEPS = 80    # control steps executed from the stop plan: all of it, so the body ends standing
LEAN_FROM = 40     # a lean plan is the leaned pose from this frame on
LEAN_STEPS = 80    # control steps executed from the lean plan
NOT_YET = 1e6      # arrival time, in seconds from the start, until the final phase begins
ARM_IK_WITHIN_M = .25  # with arm_ik, the arm is solved for the wrist over the last 25 cm of its path
ARM_JOINTS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')


def ease(u):
    """Minimum-jerk progress: 0 at u<=0, 1 at u>=1, zero velocity and acceleration at both ends."""
    u = np.clip(np.asarray(u, dtype=float), 0., 1.)
    return u**3 * (10 + u*(-15 + 6*u))


def approach_rotation(approach):
    """Wrist-link rotation whose x axis (along the forearm, toward the hand) is ``approach``.

    The link's y axis stays as close to the start frame's left as the approach allows, which is
    the link's orientation with the forearm level and the palm turned inward, for either hand.
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
    unknown = set(goal) - {'name', 'root', 'hand', 'lean', 'note'}
    if unknown:
        raise ValueError(f'Unknown goal fields: {sorted(unknown)}')
    if not goal.get('name'):
        raise ValueError('A goal needs a name')
    out = {'name': str(goal['name'])}
    if 'root' in goal:
        root = dict(goal['root'])
        xy = np.asarray(root.pop('xy'), dtype=float)
        speed, tolerance = float(root.pop('speed_m_s', .35)), float(root.pop('tolerance_m', .05))
        if root or xy.shape != (2,) or not np.isfinite(xy).all() or not .05 <= speed <= 1.5 or tolerance <= 0:
            raise ValueError('root needs xy [forward, left]; optional speed_m_s in [0.05, 1.5] and tolerance_m > 0')
        out['root'] = {'xy': xy.tolist(), 'speed_m_s': speed, 'tolerance_m': tolerance}
    if 'hand' in goal:
        hand = dict(goal['hand'])
        side = hand.pop('side')
        position = np.asarray(hand.pop('position'), dtype=float)
        approach = np.asarray(hand.pop('approach'), dtype=float)
        duration = float(hand.pop('duration_s'))
        if hand or side not in SIDES or position.shape != (3,) or not np.isfinite(position).all() or duration <= 0:
            raise ValueError('hand needs side, position [forward, left, up], approach and duration_s')
        approach_rotation(approach)
        out['hand'] = {'side': side, 'position': position.tolist(), 'approach': approach.tolist(), 'duration_s': duration}
    if 'lean' in goal:
        lean = dict(goal['lean'])
        pitch = float(lean.pop('pitch_deg'))
        by = lean.pop('by', 'waist')
        if lean or len(out) > 1 or not 0 < pitch <= 45 or by not in ('waist', 'hips'):
            raise ValueError('lean needs pitch_deg in (0, 45], by "waist" or "hips", and cannot be combined with root or hand')
        out['lean'] = {'pitch_deg': pitch, 'by': by}
    if len(out) == 1:
        raise ValueError('A goal needs a root target, a hand target, both, or a lean')
    return out


def walk_allowance(goal):
    """Seconds a walk may take before it is stopped short: the distance at 0.05 m/s, plus 5 s.

    The path asks for ``speed_m_s``, but the tracker turns the generated steps into far less
    ground than that (0.07 m/s measured on the first forward walk).
    """
    return float(np.linalg.norm(goal['root']['xy']))/.05+5.


def episode_seconds(goal, hold=2.):
    """Longest an episode of this goal can need: every phase at its allowance, then ``hold`` seconds."""
    total = READY_S+.2  # the first plan is made at 0.6 s; keep the full hold after the last phase
    if 'root' in goal:
        total += walk_allowance(goal)+2*1.6  # the walk can overrun by one plan, then the stop plan
    if 'hand' in goal:
        total += goal['hand']['duration_s']
    if 'lean' in goal:
        total += LEAN_STEPS/50
    return total+hold


def ground_path(start, goal, speed0, cruise, accel=1., frames=CANVAS, dt=1/30):
    """Ground points [frames, 2] from ``start`` straight toward ``goal``: speed up to ``cruise``, brake to stop on it.

    Row 0 is ``start`` (now). ``speed0`` is the present speed toward the goal; the path never
    passes the goal and stays on it once there.
    """
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    distance = float(np.linalg.norm(goal-start))
    if distance < 1e-9:
        return np.tile(start, (frames, 1))
    direction = (goal-start)/distance
    covered, speed, out = 0., float(np.clip(speed0, 0., cruise)), [start]
    for _ in range(frames-1):
        allowed = min(cruise, np.sqrt(2*accel*max(distance-covered, 0.)))
        speed = min(allowed, speed+accel*dt)
        covered = min(distance, covered+speed*dt)
        out.append(start+direction*covered)
    return np.array(out)


def leaned_pose(fk, joints, root, rotation, pitch_deg, foot_names, by='waist', legs_back_deg=8.):
    """Joints, root position and rotation of a standing pose with the trunk pitched forward, feet unmoved.

    ``by='waist'``: the waist joint takes what its limit allows and the pelvis pitches about the
    hips for the rest. ``by='hips'``: the waist stays, the pelvis pitches by the whole angle, and
    the legs lean back ``legs_back_deg`` about the ankles so the hips move behind the feet as the
    trunk goes ahead of them. In both the hip joints turn so the feet keep their place and
    orientation; the root pose is solved from the left foot.
    """
    if by not in ('waist', 'hips'):
        raise ValueError('Unknown lean style')
    names = list(fk.joint_names)
    waist = names.index('waist_pitch_joint')
    hips = [names.index(f'{side}_hip_pitch_joint') for side in SIDES]
    ankles = [names.index(f'{side}_ankle_pitch_joint') for side in SIDES]
    joints, root, rotation = np.array(joints, dtype=float), np.asarray(root, dtype=float), np.asarray(rotation)
    foot, foot_rotation = (x[0] for x in fk(joints, root, rotation, body_names=foot_names[:1]))
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    heading = Rotation.from_euler('z', yaw).as_matrix()

    def pose(q):
        """Root pose that puts the left foot of joints ``q`` back where it stood, and the pelvis pitch gained."""
        at, turned = (x[0] for x in fk(q, np.zeros(3), np.eye(3), body_names=foot_names[:1]))
        new_rotation = foot_rotation @ turned.T
        gained = heading.T @ new_rotation @ rotation.T @ heading
        return foot-new_rotation @ at, new_rotation, np.arctan2(gained[0, 2], gained[2, 2])
    total = np.radians(pitch_deg)
    leaned = joints.copy()
    if by == 'waist':
        leaned[waist] += min(total, fk.joint_limits[waist, 1]-.02-joints[waist])
    else:
        # Which way the ankle joint leans the shank back is read off the kinematics, not assumed.
        probe = joints.copy()
        probe[ankles] += .01
        sign = 1. if pose(probe)[0] @ heading[:, 0] < root @ heading[:, 0] else -1.
        leaned[ankles] += sign*np.radians(legs_back_deg)
    # The hip joints make up whatever pelvis pitch is still missing; the pitch is linear in them.
    want = total-(leaned[waist]-joints[waist])
    zero = pose(leaned)[2]
    probe = leaned.copy()
    probe[hips] += .01
    leaned[hips] += (want-zero)/((pose(probe)[2]-zero)/.01)
    if np.any(leaned < fk.joint_limits[:, 0]) or np.any(leaned > fk.joint_limits[:, 1]):
        raise ValueError('Leaned pose exceeds joint limits')
    new_root, new_rotation, _ = pose(leaned)
    return leaned, new_root, new_rotation


def refine_arm(fk, side, joints, previous, roots, rotations, wrists, wrist_rotations, active, speed, rate=50.):
    """Solve one arm's seven joints so its wrist link meets a pose, frame by frame; everything else is kept.

    ``joints`` [T, 29] is the reference to refine with root pose ``roots`` / ``rotations``; only frames
    where ``active`` is set are changed. Each solution stays within the joint limits and within
    ``speed`` rad/s of the frame before it (``previous`` before the first). Kinematics only: no
    collision or balance check. Returns the refined joints and the wrist error left on each frame [T], metres
    (NaN where not active).
    """
    ids = [list(fk.joint_names).index(f'{side}_{name}_joint') for name in ARM_JOINTS]
    link = [f'{side}_wrist_yaw_link']
    out, last, left = np.array(joints, dtype=float), np.asarray(previous, dtype=float)[ids], np.full(len(joints), np.nan)
    for t in range(len(out)):
        if not active[t]:
            last = out[t, ids]
            continue
        low = np.maximum(fk.joint_limits[ids, 0], last-speed/rate)
        high = np.minimum(fk.joint_limits[ids, 1], last+speed/rate)
        prior = out[t, ids].copy()

        def residual(arm):
            q = out[t].copy()
            q[ids] = arm
            position, rotation = fk(q, roots[t], rotations[t], body_names=link)
            turn = Rotation.from_matrix(wrist_rotations[t].T @ rotation[0]).as_rotvec()
            return np.r_[100*(position[0]-wrists[t]), 20*turn, .1*(arm-prior)]
        fit = least_squares(residual, np.clip(last, low+1e-9, high-1e-9), bounds=(low, high), max_nfev=60,
                            ftol=1e-8, xtol=1e-8, gtol=1e-7)
        out[t, ids] = last = fit.x
        left[t] = np.linalg.norm(fk(out[t], roots[t], rotations[t], body_names=link)[0][0]-wrists[t])
    return out, left


class LocomotionPolicy(OraclePolicy):
    def __init__(self, *args, motion_goal, noise_seed=None, walk_steps=75, arm_ik=False, **kwargs):
        # Reach configuration two: over the last 25 cm of the wrist's path the reaching arm's joints
        # are solved for the path pose; legs and trunk stay as Kimodo generated them.
        self.arm_ik = bool(arm_ik)
        if not 25 <= walk_steps <= 80:
            raise ValueError('A plan covers 1.63 s; execute between 25 and 80 control steps of a walking plan')
        self.goal = load_goal(motion_goal)
        self.walk_steps = int(walk_steps)
        self.noise_seed = noise_seed  # repeats of one goal differ only in Kimodo's sampling noise
        super().__init__(*args, **kwargs)
        if (self.mode != 'hold' or self.variant != 'bounded' or self.support_constraints != 'none'
                or self.trajectory_projection != 'none'):
            raise ValueError('Moving goals use the bounded handoff with no standing support and no IK repair')

    def reset(self, seed=None):
        self.phase, self.phases, self.jumps = 'ready', [], []
        self.hand_start = self.hand_begin = self.stance_xz = self.walk_begin = None
        self.stop_sent = self.lean_sent = False
        status = super().reset(seed)
        if self.noise_seed is not None:
            self.seed = int(self.noise_seed)
        return status

    def initialize(self, sample):
        super().initialize(sample)  # measured start frame, standing features, both wrists held
        heading, origin = self.frame.heading, self.frame.origin
        representation = self.model.representation
        self.root_end = np.asarray(self.root_goal, dtype=float).copy()
        if 'root' in self.goal:
            forward, left = self.goal['root']['xy']
            self.root_end = self.frame.points(self.initial_root + heading @ np.array([forward, left, 0.]))[[0, 2]]
        self.root_slot = representation.slice_dict['smooth_root_pos'].start
        self.heading_slice = representation.slice_dict['global_root_heading']
        self.pose_end = representation.slice_dict['global_rot_data'].stop
        info = dict(mode='motion-goal', target_source='motion_goal', motion_goal=self.goal,
                    noise_seed=self.seed, walk_steps=self.walk_steps, arm_ik=self.arm_ik, active_sides=[], phases=self.phases,
                    root_goal_world=self.frame.world([self.root_end[0], 0., self.root_end[1]])[:2].tolist(),
                    arrival_time=self.start_time+NOT_YET, witness_joint_pos=None, hand_path='none')
        if 'hand' in self.goal:
            hand = self.goal['hand']
            i = SIDES.index(hand['side'])
            # The virtual hand-end point keeps its measured offset in the wrist link's frame.
            tip = np.asarray(sample['wrist_rot'])[i].T @ (self.targets[i, 1]-self.targets[i, 0])
            rotation = heading @ approach_rotation(hand['approach'])
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
                        hand_path='minimum-jerk world line and rotation from the wrist measured when the reach '
                                  'begins; the other hand is unconstrained')
        if 'lean' in self.goal:
            joints, root, rotation = leaned_pose(self.fk, self.initial_joints, self.initial_root, self.initial_rotation,
                                                 self.goal['lean']['pitch_deg'], self.foot_names, self.goal['lean']['by'])
            pose = self.features.codec.decode_joint_configuration_pose(joints[None],
                self.frame.points([root]) @ BASIS, root_rotation_matrices=np.array([heading.T @ rotation]),
                joint_names=G1_JOINT_NAMES)
            self.lean_features = self.features.features(pose['local_rot_mats'], pose['root_positions'])[0]
            info['lean_pose'] = {'joints': joints.tolist(), 'root_world': root.tolist(), 'root_rotation_world': rotation.tolist()}
        self.arrival = info['arrival_time']
        self.target_info.update(info)
        self.save_target()

    def save_target(self):
        (self.run/'target.json').write_text(json.dumps(self.target_info, indent=2))

    def measured_xz(self, obs):
        return self.frame.points(obs['root_pos'])[[0, 2]]

    def enter(self, phase, obs, **details):
        """Switch phase at a plan boundary and record when, where and why."""
        t = obs['time']-self.start_time
        self.phase = phase
        remaining = float(np.linalg.norm(self.root_end-self.measured_xz(obs)))
        self.phases.append(dict(phase=phase, time=obs['time'], elapsed_s=t, root_remaining_m=remaining, **details))
        if phase == 'walk':
            self.walk_begin = t
        elif phase == 'lean':
            self.stance_xz = self.measured_xz(obs)
        elif phase == 'reach':
            i = self.hand_index
            self.hand_begin = t
            self.hand_start = (np.asarray(obs['wrist_pos'])[i], np.asarray(obs['wrist_rot'])[i])
            self.stance_xz = self.measured_xz(obs)
            self.arrival = obs['time']+self.goal['hand']['duration_s']
            self.target_info['hand_start'] = {'time': obs['time'], 'wrist_world': self.hand_start[0].tolist(),
                                              'wrist_rotation_world': self.hand_start[1].tolist()}
        elif phase == 'hold':
            self.arrival = obs['time']
        self.target_info['arrival_time'] = self.arrival
        self.save_target()

    def advance(self, obs):
        """The phase of the plan about to be made."""
        t = obs['time']-self.start_time
        if self.phase == 'ready' and t >= READY_S-1e-9:
            self.enter('walk' if 'root' in self.goal else 'lean' if 'lean' in self.goal else 'reach', obs)
        if self.phase == 'walk':
            remaining = float(np.linalg.norm(self.root_end-self.measured_xz(obs)))
            late = t-self.walk_begin >= walk_allowance(self.goal)
            if remaining <= self.goal['root']['tolerance_m'] or late:
                self.enter('stop', obs, walk_timed_out=bool(late and remaining > self.goal['root']['tolerance_m']))
        elif self.phase == 'stop' and self.stop_sent:
            self.enter('reach' if 'hand' in self.goal else 'hold', obs)
        elif self.phase == 'lean' and self.lean_sent:
            self.enter('hold', obs)
        return self.phase

    def anchor(self, obs):
        """Move the reference's ground position onto the measured one; returns the world shift.

        The tracker never sees this position, so shifting it changes nothing it does. It keeps the
        logged reference where the body is, and makes each plan continue from there.
        """
        shift = np.r_[np.asarray(obs['root_pos'])[:2]-np.asarray(obs['reference_root_pos'])[:2], 0.]
        self.jumps.append((obs['time'], shift))
        return shift

    def reference_features(self):
        """Kimodo features of the consumed reference over the stored samples, as one continuous motion."""
        joints = np.asarray([x['reference_joints'] for x in self.history])
        roots = np.asarray([x['reference_root_pos'] for x in self.history], dtype=float)
        for when, shift in self.jumps:  # samples from before a shift were logged in the older position
            roots[[x['time'] <= when+1e-9 for x in self.history]] += shift
        roots = (roots-self.frame.origin) @ self.frame.heading
        rotations = np.stack([self.frame.heading.T @ rotation_wxyz(x['reference_root_quat']) for x in self.history])
        first_step = int(round((self.history[0]['time']-self.start_time)*50))
        return self.features.features(*self.features.motion(joints, roots, rotations, first_step))

    def speed_toward(self, target_xz):
        """Measured ground speed of the root toward a model-frame point, over the last 0.2 s."""
        if len(self.history) < 11:
            return 0.
        now, before = (self.measured_xz(self.history[k]) for k in (-1, -11))
        direction = np.asarray(target_xz)-now
        distance = np.linalg.norm(direction)
        return 0. if distance < 1e-6 else max(0., float((now-before) @ direction/distance)/.2)

    def hand_pose(self, elapsed):
        """World wrist position [N, 3] and rotation [N, 3, 3] of the hand path at ``elapsed`` seconds."""
        start_pos, start_rot = self.hand_start
        w = ease((np.asarray(elapsed, dtype=float)-self.hand_begin)/self.goal['hand']['duration_s'])
        position = start_pos + w[:, None]*(self.hand_goal-start_pos)
        turn = Slerp([0., 1.], Rotation.from_matrix(np.stack([start_rot, self.hand_goal_rotation])))
        return position, turn(w).as_matrix()

    def conditions(self, phase, obs, step, cut):
        """Sparse future conditions of one plan: (values, mask, root path [CANVAS, 2], hand keyframes, dense hand frames)."""
        representation = self.model.representation
        values = torch.zeros(CANVAS, representation.motion_rep_dim)
        mask = torch.zeros_like(values, dtype=torch.bool)
        here = self.measured_xz(obs)
        if phase in ('reach', 'lean'):
            path = np.tile(self.stance_xz, (CANVAS, 1))
        else:
            target = self.root_end
            if phase == 'stop' and self.phases[-1].get('walk_timed_out'):
                target = here  # stopped short: stand where the body is
            path = ground_path(here, target, self.speed_toward(target), self.goal['root']['speed_m_s'])
        for k in range(1, CANVAS):
            values[k, self.root_slot], values[k, self.root_slot+2] = float(path[k, 0]), float(path[k, 1])
            mask[k, self.root_slot] = mask[k, self.root_slot+2] = True
            values[k, self.heading_slice], mask[k, self.heading_slice] = self.standing_features[self.heading_slice], True
        if phase == 'stop':
            standing = self.standing_features.clone()
            standing[self.root_slot], standing[self.root_slot+2] = float(path[-1, 0]), float(path[-1, 1])
            values[STAND_FROM:], mask[STAND_FROM:] = standing, True
        keyframes, dense = [], []
        if phase == 'lean':
            leaned = self.lean_features.clone()
            # The leaned pelvis sits behind the stance's ground point by what the feet-fixed pitch needs.
            offset = leaned[[self.root_slot, self.root_slot+2]]-self.standing_features[[self.root_slot, self.root_slot+2]]
            leaned[self.root_slot], leaned[self.root_slot+2] = (torch.as_tensor(self.stance_xz, dtype=leaned.dtype)+offset)
            values[LEAN_FROM:], mask[LEAN_FROM:] = leaned, True
        if phase == 'reach':
            standing = self.standing_features.clone()
            standing[self.root_slot], standing[self.root_slot+2] = float(self.stance_xz[0]), float(self.stance_xz[1])
            condition_support(values, mask, standing, representation, 'root')
            arrival = round((self.hand_begin+self.goal['hand']['duration_s'])*30)-cut
            keyframes = [arrival] if 0 < arrival < CANVAS else [] if arrival > 0 else [15, 30, 49]
            dense = execution_frames(step, cut, REACH_STEPS)
            for k in sorted(set(keyframes) | set(dense)):
                position, rotation = self.hand_pose(np.array([(cut+k)/30]))
                wrist = self.frame.points(np.stack([position[0], position[0] + rotation[0] @ self.hand_tip]))
                v, m = hand_keyframe(representation, frames=CANVAS, frame=int(k), smooth_root_xz=path[k],
                                     hand_positions={self.hand_side: wrist},
                                     wrist_rotations={self.hand_side: self.frame.rotations(rotation[0])})
                values[m], mask[m] = v[m], True
        return values, mask, path, keyframes, dense

    def hold_chunk(self, obs):
        """Repeat the last consumed reference pose for 0.5 s; no model call."""
        rot = self.frame.heading.T @ rotation_wxyz(obs['reference_root_quat'])
        action = np.r_[0., 0., obs['reference_root_pos'][2], rot[:, :2].ravel(), obs['reference_joints'], 0., 0.]
        return {'action_chunk': np.tile(action, (25, 1)).tolist(), 'oracle_target': self.target_info}

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
        phase = self.advance(obs)
        if phase in ('ready', 'hold'):
            return self.hold_chunk(obs)
        sent = {'reach': REACH_STEPS, 'lean': LEAN_STEPS, 'stop': STOP_STEPS}.get(phase, self.walk_steps)
        # Every plan starts from where the body is. In a reach the conditions stay on the stance it
        # began with; without this the reference's root wandered 0.3 m from the body over a reach,
        # which the tracker never sees but which put the logged reference wrist 0.3 m off.
        reference_root = np.asarray(obs['reference_root_pos'], dtype=float)
        shift = self.anchor(obs)
        cut = round(t*30)
        features = self.reference_features()
        hist = features[:-1][-100:]
        hist_mask = torch.ones_like(hist, dtype=torch.bool)
        values, mask, path, keyframes, dense = self.conditions(phase, obs, step, cut)
        # Frame zero is now: its pose is known, its velocities and contacts are left to the model.
        values[0, :self.pose_end], mask[0, :self.pose_end] = features[-1, :self.pose_end], True
        generator = torch.Generator(device=self.model.device).manual_seed(_episode_noise_seed(self.seed, self.plan_index))
        begin = time.perf_counter()
        output = self.model.predict_future(str(request['task']).strip(), torch.zeros(1, 3, 1, 1), hist,
            history_feature_mask=hist_mask, text_feat=torch.zeros(1, 4096).to(self.model.device),
            future_condition=values, future_condition_mask=mask,
            diffusion_steps=self.steps, generator=generator)
        elapsed = time.perf_counter()-begin
        output = {k: v.cpu() if torch.is_tensor(v) else v for k, v in output.items()}
        if phase == 'stop':
            self.stop_sent = True
        if phase == 'lean':
            self.lean_sent = True

        # Kimodo's 30 Hz frames on the 50 Hz control grid: the steps executed from this plan.
        times = step*.6-cut+torch.arange(sent+1)*.6
        before = output['history_last_root_position'].reshape(1, 3)
        local, root = _interpolate_motion(torch.cat((output['local_rot_mats'][:1], output['local_rot_mats'])),
                                          torch.cat((before, output['root_positions'])), times+1)
        raw_local, raw_root = local[1:], root[1:]
        zero_hands = torch.zeros(sent, 2)

        def reference_pose(root_world):
            return self.features.codec.decode_joint_configuration_pose(np.array([obs['reference_joints']]),
                np.array([self.frame.points(root_world) @ BASIS]),
                root_rotation_matrices=np.array([self.frame.heading.T @ rotation_wxyz(obs['reference_root_quat'])]),
                joint_names=G1_JOINT_NAMES)
        boundary = reference_pose(reference_root+shift)       # the consumed reference, where the body is
        consumed_root = reference_pose(reference_root)['root_positions'][0]  # where the simulator still has it
        boundary_root = boundary['root_positions'][0]
        raw_actions = self.features.codec.encode(raw_local, raw_root, previous_root_position=consumed_root,
                                                 hand_binary=zero_hands).numpy()
        sent_local, sent_root = blend_reference(raw_local, raw_root, boundary['local_rot_mats'][0], boundary_root,
                                                np.arange(1, sent+1)/50, self.handoff_seconds)
        actions = self.features.codec.encode(sent_local, sent_root, previous_root_position=consumed_root,
                                             hand_binary=zero_hands).numpy()
        actions[:, 9:38] = bounded_hinge_trajectory(sent_local, self.features.codec, self.fk.joint_limits,
                                                    obs['reference_joints'], self.joint_speed)
        arm_ik_left = None
        if self.arm_ik and phase == 'reach':
            path_position, path_rotation = self.hand_pose(t+np.arange(1, sent+1)/50)
            near = np.linalg.norm(path_position-self.hand_goal, axis=1) <= ARM_IK_WITHIN_M
            rotation = self.frame.heading @ rot6d_row_to_matrix(torch.from_numpy(actions[:, 3:9])).numpy()
            actions[:, 9:38], arm_ik_left = refine_arm(self.fk, self.hand_side, actions[:, 9:38], obs['reference_joints'],
                self.frame.world(sent_root.numpy()), rotation, path_position, path_rotation, near, self.joint_speed)

        def world_pose(action, model_root, names=None):
            """World root rotation, and positions and rotations of the wrists (or ``names``), per action."""
            rotation = self.frame.heading @ rot6d_row_to_matrix(torch.from_numpy(action[:, 3:9])).numpy()
            bodies = [self.fk(a[9:38], self.frame.world(p), r, body_names=names)
                      for a, p, r in zip(action, model_root.numpy(), rotation)]
            return rotation, np.array([b[0] for b in bodies]), np.array([b[1] for b in bodies])
        raw_rotation, raw_wrist, _ = world_pose(raw_actions, raw_root)
        sent_rotation, sent_wrist, _ = world_pose(actions, sent_root)
        raw_feet = world_pose(raw_actions, raw_root, self.foot_names)[1]
        sent_feet = world_pose(actions, sent_root, self.foot_names)[1]
        rows = [{'plan': self.plan_index, 'action_index': i, 'time': obs['time']+(i+1)/50, 'phase': phase,
                 'raw_projected_wrist': raw_wrist[i].tolist(), 'sent_wrist': sent_wrist[i].tolist(),
                 'raw_joints': raw_actions[i, 9:38].tolist(), 'sent_joints': actions[i, 9:38].tolist(),
                 'raw_root_world': self.frame.world(raw_root[i].numpy()).tolist(),
                 'sent_root_world': self.frame.world(sent_root[i].numpy()).tolist(),
                 'raw_root_rotation_world': raw_rotation[i].tolist(),
                 'sent_root_rotation_world': sent_rotation[i].tolist(),
                 'raw_feet': raw_feet[i].tolist(), 'sent_feet': sent_feet[i].tolist()} for i in range(sent)]
        with (self.run/'commands.jsonl').open('a') as f:
            f.write(''.join(json.dumps(x)+'\n' for x in rows))

        # The whole 50-frame prediction, decoded the way analyze_prediction.py decodes it.
        all_actions = self.features.codec.encode(output['local_rot_mats'], output['root_positions'],
            previous_root_position=before[0], hand_binary=torch.zeros(CANVAS, 2)).numpy()
        _, future_wrist, future_wrist_rotation = world_pose(all_actions, output['root_positions'])
        future_root = self.frame.world(output['root_positions'].numpy())
        path_world = self.frame.world(np.insert(path, 1, 0., axis=1))[:, :2]
        record = {'plan': self.plan_index, 'time': obs['time'], 'cut': cut, 'phase': phase, 'sent_steps': sent,
                  'keyframes': keyframes, 'dense_hand_frames': dense, 'root_frames': list(range(1, CANVAS)),
                  'pose_frames': list(range({'stop': STAND_FROM, 'lean': LEAN_FROM}.get(phase, CANVAS), CANVAS)),
                  'hand_constraints': 'continuous', 'history_source': 'reference',
                  'history_condition_frames': int(hist_mask.any(-1).sum()),
                  'inference_s': elapsed, 'constraint_cfg': self.constraint_cfg, 'diffusion_steps': self.steps,
                  'variant': self.variant, 'handoff_seconds': self.handoff_seconds,
                  'joint_speed_rad_s': self.joint_speed, 'support_constraints': 'none', 'trajectory_projection': 'none',
                  'arm_ik_frames': 0 if arm_ik_left is None else int(np.isfinite(arm_ik_left).sum()),
                  'arm_ik_left_cm_max': None if arm_ik_left is None or not np.isfinite(arm_ik_left).any()
                                        else 100*float(np.nanmax(arm_ik_left)),
                  'anchor_shift_m': float(np.linalg.norm(shift)), 'anchor_shift_world': shift[:2].tolist(),
                  'root_remaining_m': float(np.linalg.norm(self.root_end-self.measured_xz(obs))),
                  # How far Kimodo's own output is from the path it was given, before any execution.
                  'root_path_error_m': np.linalg.norm(future_root[1:, :2]-path_world[1:], axis=1).max().item()}
        if phase == 'reach':
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
