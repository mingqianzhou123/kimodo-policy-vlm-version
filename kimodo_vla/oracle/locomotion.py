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
the waist joint or, with "by": "hips", by the pelvis; Kimodo generates the way there. A squat goal
(added 2026-11-26 for the second pilot batch's command test) lowers the pelvis with the trunk upright
and the feet unmoved, and goes the lean's way (the same phase, one plan to the end pose):

    {"name": "squat-15", "squat": {"drop_m": 0.15}}

A root goal may also turn on the spot: ``{"xy": [0, 0], "yaw_deg": 90, "yaw_rate_deg_s": 45}``
conditions the heading to turn toward 90 degrees left of where it faced when the walk began.

Ways of using Kimodo while walking, all off by default (the default is experiment two's setting:
no text, the root ground path and heading on every frame, up to 100 history frames, 0.5 s plans):

  text              a phrase key of the text cache; guidance is then Kimodo's separated CFG with
                    text weight ``text_cfg`` and constraint weight ``constraint_cfg``
  root_constraints  'dense' (every frame), 'sparse' (frames 15, 30, 49) or 'none'
  history_frames    how many consumed-reference frames condition each plan (1 = only the present)
  lead_segment      a recorded walking segment (``demo_walks``) played open loop before the walk,
                    so the walk's history is real walking; the goal is then relative to where the
                    body is and faces when the lead-in ends
  generator         'policy' (Kimodo-policy's canvas: up to 100 history frames + 50 generated, the
                    default) or 'segment': walk and stop plans are whole segments of
                    ``segment_seconds`` sampled the official Kimodo way (``model.segment_sampler``),
                    the present only a frame-0 keyframe, ``segment_steps`` DDIM steps
  generator='native'  every plan (walk, stop, reach, lean) is sampled the official Kimodo way and
                    continues the reference already sent the way official multi-prompt generation
                    continues a segment: its last ``transition_frames`` frames (ending now, already
                    moved to where the body was measured) constrained in full plus hand and foot
                    rotations, the rest of the canvas free but for the phase's own conditions; no
                    longer history. Walk and stop canvases run ``segment_seconds`` past now, reach
                    and lean canvases 49 frames. Saved plan features start at now. ``text='auto'`` gives the walk the phrase of its
                    direction (``WALK_TEXTS``) and the other phases no text; ``text_until_m`` drops
                    the phrase once the goal is that close (it otherwise marches on the spot).
  open_loop         plans start from the reference's own position and heading instead of the
                    measured ones (no anchoring): the reference carries out the command and the
                    tracker's deviation is left in, to measure it (native generator only)
  approach_gain     walk and stop paths aim at here + gain x (goal - here): the tracker covers only
                    part of each commanded displacement (about half for SONIC), so gain > 1 closes
                    the last centimetres; 1 (the default) aims at the goal.
  walk_clips        a gait-cycle library (``clip_legs``): the walk is looped recorded strides chosen
                    from the measured position, Kimodo is not called until the walk has stopped
                    (the reach, if any, is Kimodo's as before); there is no separate stop plan
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
from kimodo_vla.model.segment_sampler import decode, sample_segment, transition_condition
from .clip_legs import ClipWalker, load_library
from .demo_walks import Timeline, load_segment
from .geometry import BASIS, SIDES, angle_deg, rotation_wxyz
from .hand_path import execution_frames
from kimodo_vla.model.native_policy import ConstraintGuidance, SeparatedGuidance
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
WALK_TEXTS = {'forward': 'walk_forward', 'backward': 'walk_backward', 'left': 'sidestep_left',
              'right': 'sidestep_right', 'turn': 'turn_left_in_place'}  # text='auto' with the native generator
SPARSE_FRAMES = (15, 30, 49)  # root_constraints='sparse': one waypoint every 0.5 s
ROOT_CONSTRAINTS = ('dense', 'sparse', 'none')
ARM_JOINTS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')


def step_of(obs, start_time):
    """Control step of a sample, counted from the episode start."""
    return int(round((obs['time']-start_time)*50))


def ease(u):
    """Minimum-jerk progress: 0 at u<=0, 1 at u>=1, zero velocity and acceleration at both ends."""
    u = np.clip(np.asarray(u, dtype=float), 0., 1.)
    return np.clip(u**3 * (10 + u*(-15 + 6*u)), 0., 1.)  # rounding can leave 1 + 2e-16


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
    unknown = set(goal) - {'name', 'root', 'hand', 'lean', 'squat', 'note'}
    if unknown:
        raise ValueError(f'Unknown goal fields: {sorted(unknown)}')
    if not goal.get('name'):
        raise ValueError('A goal needs a name')
    out = {'name': str(goal['name'])}
    if 'root' in goal:
        root = dict(goal['root'])
        xy = np.asarray(root.pop('xy'), dtype=float)
        speed, tolerance = float(root.pop('speed_m_s', .35)), float(root.pop('tolerance_m', .05))
        yaw, rate = float(root.pop('yaw_deg', 0.)), float(root.pop('yaw_rate_deg_s', 45.))
        if root or xy.shape != (2,) or not np.isfinite(xy).all() or not .05 <= speed <= 1.5 or tolerance <= 0:
            raise ValueError('root needs xy [forward, left]; optional speed_m_s in [0.05, 1.5] and tolerance_m > 0')
        if not abs(yaw) <= 180 or not 5 <= rate <= 180 or (yaw and np.linalg.norm(xy) > 0):
            raise ValueError('A turn (yaw_deg up to 180, yaw_rate_deg_s in [5, 180]) is on the spot: xy must be [0, 0]')
        out['root'] = {'xy': xy.tolist(), 'speed_m_s': speed, 'tolerance_m': tolerance}
        if yaw:
            out['root'].update(yaw_deg=yaw, yaw_rate_deg_s=rate)
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
    if 'squat' in goal:
        squat = dict(goal['squat'])
        drop = float(squat.pop('drop_m'))
        if squat or len(out) > 1 or not 0 < drop <= .3:
            raise ValueError('squat needs drop_m in (0, 0.3] and cannot be combined with root, hand or lean')
        out['squat'] = {'drop_m': drop}
    if len(out) == 1:
        raise ValueError('A goal needs a root target, a hand target, both, a lean or a squat')
    return out


def anchored_goal(goal, facing_deg):
    """The goal with its walking target turned by ``facing_deg``: from the robot's own start frame into a
    frame in which the robot faces ``facing_deg`` to the left of x (the anchor frame of a mid-episode start)."""
    out = json.loads(json.dumps(goal))
    a = np.radians(facing_deg)
    x, y = out['root']['xy']
    out['root']['xy'] = [float(np.cos(a)*x-np.sin(a)*y), float(np.sin(a)*x+np.cos(a)*y)]
    return out


def walk_allowance(goal):
    """Seconds a walk may take before it is stopped short: the distance at 0.05 m/s, plus 5 s.

    The path asks for ``speed_m_s``, but the tracker turns the generated steps into far less
    ground than that (0.07 m/s measured on the first forward walk).
    """
    if goal['root'].get('yaw_deg'):
        return abs(goal['root']['yaw_deg'])/10.+5.   # turns: 10 degrees per second
    return float(np.linalg.norm(goal['root']['xy']))/.05+5.


def goal_direction(goal):
    """'forward', 'backward', 'left', 'right' (the larger component of the root goal) or 'turn'."""
    if goal['root'].get('yaw_deg'):
        return 'turn'
    forward, left = goal['root']['xy']
    if abs(forward) >= abs(left):
        return 'forward' if forward >= 0 else 'backward'
    return 'left' if left > 0 else 'right'


def open_loop_seconds(goal, hold=2.):
    """Episode length for ``open_loop``: the reference walks the command at its speed (or turns at its
    rate), with 2 s to spare, then the stop plan and ``hold`` seconds."""
    root = goal['root']
    walk = abs(root['yaw_deg'])/root['yaw_rate_deg_s'] if root.get('yaw_deg') else np.linalg.norm(root['xy'])/root['speed_m_s']
    return READY_S+.2+walk+2.+2*1.6+hold


def episode_seconds(goal, hold=2., lead_s=0.):
    """Longest an episode of this goal can need: every phase at its allowance, then ``hold`` seconds.

    ``lead_s``: the length of a recorded lead-in (blend included) played before the walk."""
    total = READY_S+.2  # the first plan is made at 0.6 s; keep the full hold after the last phase
    if lead_s:
        total += lead_s
    if 'root' in goal:
        total += walk_allowance(goal)+2*1.6  # the walk can overrun by one plan, then the stop plan
    if 'hand' in goal:
        total += goal['hand']['duration_s']
    if 'lean' in goal or 'squat' in goal:
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


def squatted_pose(fk, joints, root, rotation, drop_m, foot_names):
    """Joints, root position and rotation of the start pose with the pelvis ``drop_m`` lower, feet unmoved.

    Both legs bend the same way: for a knee bend k the hip and ankle pitch joints are solved so that
    the pelvis keeps its orientation and stays over the same ground point (along the heading), and
    k is found so that the pelvis drops by ``drop_m``. Arms, waist and the other joints keep their
    values. The root pose is solved from the left foot, as for the lean.
    """
    names = list(fk.joint_names)
    hips = [names.index(f'{side}_hip_pitch_joint') for side in SIDES]
    knees = [names.index(f'{side}_knee_joint') for side in SIDES]
    ankles = [names.index(f'{side}_ankle_pitch_joint') for side in SIDES]
    joints, root, rotation = np.array(joints, dtype=float), np.asarray(root, dtype=float), np.asarray(rotation)
    foot, foot_rotation = (x[0] for x in fk(joints, root, rotation, body_names=foot_names[:1]))
    forward = Rotation.from_euler('z', np.arctan2(rotation[1, 0], rotation[0, 0])).as_matrix()[:, 0]

    def pose(q):
        at, turned = (x[0] for x in fk(q, np.zeros(3), np.eye(3), body_names=foot_names[:1]))
        new_rotation = foot_rotation @ turned.T
        return foot-new_rotation @ at, new_rotation

    def bent(k):
        def residual(x):
            q = joints.copy()
            q[knees] += k
            q[hips] += x[0]
            q[ankles] += x[1]
            new_root, new_rotation = pose(q)
            turn = Rotation.from_matrix(new_rotation @ rotation.T).as_rotvec()
            return np.r_[10*turn, (new_root-root) @ forward]
        x = least_squares(residual, [-k/2, -k/2]).x
        q = joints.copy()
        q[knees] += k
        q[hips] += x[0]
        q[ankles] += x[1]
        return q, *pose(q)

    low, high = 0., 1.5
    for _ in range(40):   # pelvis height falls monotonically with the knee bend over this range
        k = (low+high)/2
        if root[2]-bent(k)[1][2] < drop_m:
            low = k
        else:
            high = k
    squatted, new_root, new_rotation = bent((low+high)/2)
    if abs(root[2]-new_root[2]-drop_m) > 1e-3:
        raise ValueError('Squat depth not reachable')
    if np.any(squatted < fk.joint_limits[:, 0]) or np.any(squatted > fk.joint_limits[:, 1]):
        raise ValueError('Squatted pose exceeds joint limits')
    return squatted, new_root, new_rotation


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
    # Ways of using Kimodo (module docstring); class defaults are experiment two's setting.
    text, text_cfg, text_feat = None, 2., None
    root_constraints, history_frames, lead = 'dense', 100, None
    clips, clip_walker = None, None
    generator_mode, segment_frames, segment_steps, transition_frames, walk_text = 'policy', CANVAS, 50, 5, None
    approach_gain, text_until_m, open_loop = 1., 0., False

    def __init__(self, *args, motion_goal, noise_seed=None, walk_steps=75, arm_ik=False, text=None, text_cfg=2.,
                 root_constraints='dense', history_frames=100, lead_segment=None, walk_clips=None, model=None,
                 generator='policy', segment_seconds=4., segment_steps=100, transition_frames=5, approach_gain=1.,
                 text_until_m=0., open_loop=False, anchor_heading=None, **kwargs):
        # Reach configuration two: over the last 25 cm of the wrist's path the reaching arm's joints
        # are solved for the path pose; legs and trunk stay as Kimodo generated them.
        self.arm_ik = bool(arm_ik)
        if generator not in ('policy', 'segment', 'native'):
            raise ValueError("generator must be 'policy', 'segment' or 'native'")
        self.generator_mode = generator
        if generator == 'native':
            self.transition_frames = int(transition_frames)
            if not 1 <= self.transition_frames <= 30:
                raise ValueError('transition_frames must be 1-30')
        if text == 'auto' and generator != 'native':
            raise ValueError("text='auto' needs the native generator")
        self.approach_gain, self.text_until_m, self.open_loop = float(approach_gain), float(text_until_m), bool(open_loop)
        if self.open_loop and (generator != 'native' or approach_gain != 1.):
            raise ValueError('open_loop is for the native generator without approach gain')
        if not 1. <= self.approach_gain <= 4. or self.text_until_m < 0:
            raise ValueError('approach_gain in [1, 4]; text_until_m >= 0')
        if generator in ('segment', 'native'):
            self.segment_frames, self.segment_steps = int(round(segment_seconds*30)), int(segment_steps)
            shortest = 1. if generator == 'native' else 2.
            if not shortest <= segment_seconds <= 10 or not 25 <= walk_steps <= int((self.segment_frames-1)/.6):
                raise ValueError(f'Segments of {shortest:g}-10 s; execute at least 25 control steps and no more than the segment')
        elif not 25 <= walk_steps <= 80:
            raise ValueError('A plan covers 1.63 s; execute between 25 and 80 control steps of a walking plan')
        self.goal = load_goal(motion_goal)
        # Off by default. A policy started mid-episode (kimodo_vla/oracle/carry_policy.py) passes the heading
        # Arena's action decoder is anchored at; its walking goal stays in the robot's own start frame.
        self.anchor_heading = None if anchor_heading is None else np.asarray(anchor_heading, dtype=float)
        self.walk_steps = int(walk_steps)
        self.noise_seed = noise_seed  # repeats of one goal differ only in Kimodo's sampling noise
        if root_constraints not in ROOT_CONSTRAINTS or not 1 <= int(history_frames) <= 100 or text_cfg < 0:
            raise ValueError('root_constraints must be dense, sparse or none; history_frames 1 to 100; text_cfg >= 0')
        self.root_constraints, self.history_frames = root_constraints, int(history_frames)
        self.lead = None if lead_segment is None else load_segment(lead_segment)
        if self.lead is not None and 'root' not in self.goal:
            raise ValueError('A recorded lead-in comes before a walk')
        self.clips = None if walk_clips is None else load_library(walk_clips)
        if self.clips is not None and ('root' not in self.goal or self.lead is not None):
            raise ValueError('Walking on recorded cycles needs a root goal and no lead-in')
        self.shared_model = model      # a loaded model to reuse (offline sweeps), else OraclePolicy loads one
        super().__init__(*args, **kwargs)
        self.text, self.text_cfg = text, float(text_cfg)
        guided = isinstance(self.model.denoiser, (ConstraintGuidance, SeparatedGuidance))
        inner = self.model.denoiser.model if guided else self.model.denoiser
        if text == 'auto':
            self.walk_text = WALK_TEXTS[goal_direction(self.goal)] if 'root' in self.goal else None
            if self.walk_text is not None and self.walk_text not in self.texts:
                raise ValueError(f'No text embedding {self.walk_text!r} in the text cache')
            self.text_feat = None if self.walk_text is None else self.texts[self.walk_text].float()
        elif text is None:
            self.text_feat = None
            if self.shared_model is not None:   # a reused model may still carry another policy's guidance
                self.model.denoiser = ConstraintGuidance(inner, self.constraint_cfg)
        else:
            if text not in self.texts:
                raise ValueError(f'No text embedding {text!r} in the text cache')
            self.text_feat = self.texts[text].float()
            self.model.denoiser = SeparatedGuidance(inner, self.text_cfg, self.constraint_cfg)
        if (self.mode != 'hold' or self.variant != 'bounded' or self.support_constraints != 'none'
                or self.trajectory_projection != 'none'):
            raise ValueError('Moving goals use the bounded handoff with no standing support and no IK repair')

    def reset(self, seed=None):
        self.phase, self.phases, self.jumps = 'ready', [], []
        self.hand_start = self.hand_begin = self.stance_xz = self.walk_begin = None
        self.stop_sent = self.lean_sent = False
        self.lead_timeline = self.lead_begin = self.lead_end_step = None
        self.clip_walker = None
        self.walk_yaw = self.last_yaw = self.turned = 0.
        status = super().reset(seed)
        if self.noise_seed is not None:
            self.seed = int(self.noise_seed)
        return status

    def initialize(self, sample):
        if self.anchor_heading is not None and self.goal.get('root') and not self.goal['root'].get('yaw_deg'):
            facing = self.anchor_heading.T @ rotation_wxyz(sample['root_quat'])
            self.goal = anchored_goal(self.goal, float(np.degrees(np.arctan2(facing[1, 0], facing[0, 0]))))
        super().initialize(sample, heading=self.anchor_heading)  # measured start frame, standing features, both wrists held
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
        if 'lean' in self.goal or 'squat' in self.goal:
            if 'lean' in self.goal:
                joints, root, rotation = leaned_pose(self.fk, self.initial_joints, self.initial_root, self.initial_rotation,
                                                     self.goal['lean']['pitch_deg'], self.foot_names, self.goal['lean']['by'])
            else:
                joints, root, rotation = squatted_pose(self.fk, self.initial_joints, self.initial_root, self.initial_rotation,
                                                       self.goal['squat']['drop_m'], self.foot_names)
            pose = self.features.codec.decode_joint_configuration_pose(joints[None],
                self.frame.points([root]) @ BASIS, root_rotation_matrices=np.array([heading.T @ rotation]),
                joint_names=G1_JOINT_NAMES)
            self.lean_features = self.features.features(pose['local_rot_mats'], pose['root_positions'])[0]
            info['lean_pose'] = {'joints': joints.tolist(), 'root_world': root.tolist(), 'root_rotation_world': rotation.tolist()}
        self.arrival = info['arrival_time']
        self.target_info.update(info)
        self.save_target()

    def measured_yaw(self, obs):
        """Heading of the measured root in the start frame, degrees, left positive (open loop: the reference's)."""
        rotation = self.frame.heading.T @ rotation_wxyz(obs['reference_root_quat' if self.open_loop else 'root_quat'])
        return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))

    def yaw_remaining(self, obs):
        """Degrees still to turn (0 for goals without a turn)."""
        if not self.goal.get('root', {}).get('yaw_deg'):
            return 0.
        # Turned so far, unwrapped from sample to sample: a 180 degree goal must not read as -180.
        now = self.measured_yaw(obs)
        self.turned += (now-self.last_yaw+180.) % 360.-180.
        self.last_yaw = now
        return self.goal['root']['yaw_deg']-self.turned

    def turned_standing(self, standing, heading):
        """Standing features rotated about the vertical so their heading entries equal ``heading`` (cos, sin)."""
        rep = self.model.representation
        now = standing[self.heading_slice]
        delta = float(torch.atan2(heading[1], heading[0])-torch.atan2(now[1], now[0]))
        for angle in (delta, -delta):
            turned = rep.rotate(standing[None, None].cpu(), torch.tensor([angle]))[0, 0].to(standing)
            if torch.allclose(turned[self.heading_slice], torch.as_tensor(heading).to(standing), atol=1e-4):
                return turned
        raise RuntimeError('Could not turn the standing pose to the requested heading')

    def heading_features(self, yaw_deg):
        """Kimodo heading features [N, 2] of the standing pose turned ``yaw_deg`` (start frame, left positive)."""
        yaw = np.radians(np.atleast_1d(np.asarray(yaw_deg, dtype=float)))
        tilt = self.frame.heading.T @ self.initial_rotation
        tilt = Rotation.from_euler('z', -np.arctan2(tilt[1, 0], tilt[0, 0])).as_matrix() @ tilt
        rotations = np.stack([Rotation.from_euler('z', y).as_matrix() @ tilt for y in yaw])
        pose = self.features.codec.decode_joint_configuration_pose(np.tile(self.initial_joints, (len(yaw), 1)),
            np.tile(self.frame.points([self.initial_root]) @ BASIS, (len(yaw), 1)),
            root_rotation_matrices=rotations, joint_names=G1_JOINT_NAMES)
        return self.features.features(pose['local_rot_mats'], pose['root_positions'])[:, self.heading_slice]

    def lead_chunk(self, obs):
        """The next commands of the recorded lead-in, ending exactly where its window begins."""
        if self.lead_timeline is None:
            self.lead_timeline = Timeline(self.lead, obs, settle_s=0.)
            self.lead_yaw = self.measured_yaw(obs)
            self.lead_begin = step_of(obs, self.start_time)
            self.lead_end_step = self.lead_begin+self.lead_timeline.window[0]
            self.target_info['lead'] = {'segment': self.lead['name'], 'begin_time': obs['time'],
                                        'end_time': self.start_time+self.lead_end_step/50,
                                        'window_begin': int(self.lead['window'][0])}
            self.save_target()
        index = step_of(obs, self.start_time)-self.lead_begin
        size = min(25, self.lead_timeline.window[0]-index)
        actions = self.lead_timeline.actions[index:index+size].copy()
        # The timeline's rotations are relative to the heading measured when the lead-in began; Arena
        # wants them relative to the episode's first heading.
        began = Rotation.from_euler('z', np.radians(self.lead_yaw)).as_matrix()
        relative = np.einsum('ij,tjk->tik', began, rot6d_row_to_matrix(torch.from_numpy(actions[:, 3:9])).numpy())
        actions[:, 3:9] = relative[:, :, :2].reshape(len(actions), 6)
        self.log_open_loop(actions, relative, obs, 'lead', {'lead_index': int(index)})
        return {'action_chunk': actions.tolist(), 'oracle_target': self.target_info}

    def clip_chunk(self, obs):
        """The next commands of the walk on recorded gait cycles (``clip_legs``)."""
        here = (np.asarray(obs['root_pos'], dtype=float)-self.frame.origin) @ self.frame.heading
        if self.clip_walker is None:
            rot = self.frame.heading.T @ rotation_wxyz(obs['reference_root_quat'])
            stance = np.r_[0., 0., obs['reference_root_pos'][2], rot[:, :2].ravel(), obs['reference_joints'], 0., 0.]
            goal = self.frame.heading.T @ (np.r_[self.frame.world([self.root_end[0], 0., self.root_end[1]])[:2], 0.]
                                           - self.frame.origin)
            turn = self.goal['root'].get('yaw_deg', 0.)
            self.clip_walker = ClipWalker(self.clips, goal[:2], self.walk_yaw+turn if turn else 0., stance,
                                          variant=(self.seed or 1000)-1000, tolerance_m=self.goal['root']['tolerance_m'])
        actions = self.clip_walker.chunk(here[:2], self.measured_yaw(obs), obs['time'])
        relative = rot6d_row_to_matrix(torch.from_numpy(actions[:, 3:9])).numpy()
        self.log_open_loop(actions, relative, obs, 'walk', {'clip': self.clip_walker.current,
                                                            'expected_advance': dict(self.clip_walker.expected)})
        return {'action_chunk': actions.tolist(), 'oracle_target': self.target_info}

    def log_open_loop(self, actions, relative, obs, phase, extra):
        """Log commands that did not come from Kimodo in the walk plans' format, root integrated from the body."""
        rotation = self.frame.heading @ relative
        root, rows = np.array(obs['root_pos'], dtype=float), []
        for i, (a, r) in enumerate(zip(actions, rotation)):
            step = r @ np.array([a[0], a[1], 0.])
            root = np.array([root[0]+step[0], root[1]+step[1], a[2]])
            wrist = self.fk(a[9:38], root, r)[0]
            feet = self.fk(a[9:38], root, r, body_names=self.foot_names)[0]
            rows.append({'plan': self.plan_index, 'action_index': i, 'time': obs['time']+(i+1)/50, 'phase': phase,
                         'raw_projected_wrist': wrist.tolist(), 'sent_wrist': wrist.tolist(),
                         'raw_joints': a[9:38].tolist(), 'sent_joints': a[9:38].tolist(),
                         'raw_root_world': root.tolist(), 'sent_root_world': root.tolist(),
                         'raw_root_rotation_world': r.tolist(), 'sent_root_rotation_world': r.tolist(),
                         'raw_feet': feet.tolist(), 'sent_feet': feet.tolist()})
        with (self.run/'commands.jsonl').open('a') as f:
            f.write(''.join(json.dumps(x)+'\n' for x in rows))
        with (self.run/'plans.jsonl').open('a') as f:
            f.write(json.dumps({'plan': self.plan_index, 'time': obs['time'], 'phase': phase, 'sent_steps': len(actions),
                                'model_called': False, 'inference_s': 0., 'root_path_error_m': 0., **extra})+'\n')
        self.plan_index += 1

    def save_target(self):
        (self.run/'target.json').write_text(json.dumps(self.target_info, indent=2))

    def walk_text_on(self, phase, obs):
        """The walk phrase conditions this plan: walking, and (with ``text_until_m``) still that far from the goal."""
        if phase != 'walk' or self.text_feat is None:
            return False
        return float(np.linalg.norm(self.root_end-self.measured_xz(obs))) > self.text_until_m or bool(
            self.goal['root'].get('yaw_deg'))

    def measured_xz(self, obs):
        """Ground point the plans start from: the measured root (open loop: the reference's own root)."""
        return self.frame.points(obs['reference_root_pos' if self.open_loop else 'root_pos'])[[0, 2]]

    def enter(self, phase, obs, **details):
        """Switch phase at a plan boundary and record when, where and why."""
        t = obs['time']-self.start_time
        self.phase = phase
        remaining = float(np.linalg.norm(self.root_end-self.measured_xz(obs)))
        self.phases.append(dict(phase=phase, time=obs['time'], elapsed_s=t, root_remaining_m=remaining, **details))
        if phase == 'walk':
            self.walk_begin = t
            self.walk_yaw = self.last_yaw = self.measured_yaw(obs)
            self.turned = 0.
            if self.lead is not None:
                # The goal is relative to where the body is and faces when the recorded lead-in ends.
                here = np.asarray(obs['root_pos'], dtype=float)
                turn = Rotation.from_euler('z', np.radians(self.walk_yaw)).as_matrix()
                forward, left = self.goal['root']['xy']
                self.root_end = self.frame.points(here+self.frame.heading @ turn @ np.array([forward, left, 0.]))[[0, 2]]
                self.target_info['root_goal_world'] = self.frame.world([self.root_end[0], 0., self.root_end[1]])[:2].tolist()
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
            if self.lead is not None:
                self.enter('lead', obs)
            else:
                self.enter('walk' if 'root' in self.goal else 'lean' if 'lean' in self.goal or 'squat' in self.goal else 'reach', obs)
        if self.phase == 'lead' and self.lead_end_step is not None and step_of(obs, self.start_time) >= self.lead_end_step:
            self.enter('walk', obs)
        if self.phase == 'walk' and self.clips is not None:
            if self.clip_walker is not None and self.clip_walker.done:
                self.target_info['clip_walk'] = self.clip_walker.log
                self.enter('reach' if 'hand' in self.goal else 'hold', obs)
            return self.phase
        if self.phase == 'walk':
            remaining = float(np.linalg.norm(self.root_end-self.measured_xz(obs)))
            yaw_left = abs(self.yaw_remaining(obs))
            late = t-self.walk_begin >= walk_allowance(self.goal)
            done = remaining <= self.goal['root']['tolerance_m'] and yaw_left <= 5.
            if done or late:
                self.enter('stop', obs, walk_timed_out=bool(late and not done))
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
        if self.open_loop:
            shift[:] = 0.   # open loop: the reference keeps its own position; nothing is measured back in
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

    def conditions(self, phase, obs, step, cut, canvas=CANVAS):
        """Sparse future conditions of one plan: (values, mask, root path [canvas, 2], hand keyframes, dense hand frames)."""
        representation = self.model.representation
        values = torch.zeros(canvas, representation.motion_rep_dim)
        mask = torch.zeros_like(values, dtype=torch.bool)
        here = self.measured_xz(obs)
        if phase in ('reach', 'lean'):
            path = np.tile(self.stance_xz, (canvas, 1))
        else:
            target = self.root_end
            if phase == 'stop' and self.phases[-1].get('walk_timed_out'):
                target = here  # stopped short: stand where the body is
            elif self.approach_gain != 1.:
                # The tracker covers only part of a commanded displacement; aim past the goal by
                # (gain - 1) times what is left so that the body itself ends nearer it.
                target = here+self.approach_gain*(target-here)
            path = ground_path(here, target, self.speed_toward(target), self.goal['root']['speed_m_s'], frames=canvas)
        heading = None   # experiment two: the start heading on every frame
        if phase == 'walk' and (self.lead is not None or self.goal['root'].get('yaw_deg')):
            now, left = self.measured_yaw(obs), self.yaw_remaining(obs)
            turned = np.sign(left)*np.minimum(abs(left), self.goal['root'].get('yaw_rate_deg_s', 0.)*np.arange(canvas)/30)
            heading = self.heading_features(now+turned)
        elif phase == 'stop' and (self.lead is not None or self.goal['root'].get('yaw_deg')):
            heading = self.heading_features(np.full(canvas, self.measured_yaw(obs)))
        frames = range(1, canvas)
        if phase == 'walk' and self.root_constraints != 'dense':
            frames = SPARSE_FRAMES if self.root_constraints == 'sparse' else ()
        for k in frames:
            values[k, self.root_slot], values[k, self.root_slot+2] = float(path[k, 0]), float(path[k, 1])
            mask[k, self.root_slot] = mask[k, self.root_slot+2] = True
            values[k, self.heading_slice] = self.standing_features[self.heading_slice] if heading is None else heading[k]
            mask[k, self.heading_slice] = True
        if phase == 'stop':
            standing = self.standing_features.clone()
            if heading is not None and self.generator_mode == 'native':
                # The standing pose turned to the heading reached; without this the pose (joint
                # positions and rotations, not only the heading entries) faces the start and the stop
                # plan turns the body back. Only for the native generator: the other paths keep theirs.
                standing = self.turned_standing(standing, heading[-1])
            standing[self.root_slot], standing[self.root_slot+2] = float(path[-1, 0]), float(path[-1, 1])
            if heading is not None:
                standing[self.heading_slice] = heading[-1]
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
        if phase == 'lead':
            return self.lead_chunk(obs)
        if phase == 'walk' and self.clips is not None:
            return self.clip_chunk(obs)
        sent = {'reach': REACH_STEPS, 'lean': LEAN_STEPS, 'stop': STOP_STEPS}.get(phase, self.walk_steps)
        # Every plan starts from where the body is. In a reach the conditions stay on the stance it
        # began with; without this the reference's root wandered 0.3 m from the body over a reach,
        # which the tracker never sees but which put the logged reference wrist 0.3 m off.
        reference_root = np.asarray(obs['reference_root_pos'], dtype=float)
        shift = self.anchor(obs)
        cut = round(t*30)
        features = self.reference_features()
        hist = features[:-1][-self.history_frames:]
        hist_mask = torch.ones_like(hist, dtype=torch.bool)
        native = self.generator_mode == 'native'
        segment = self.generator_mode in ('segment', 'native') and phase in ('walk', 'stop')
        # A native walk or stop canvas is now plus segment_seconds after it (5 s reads a 5 s prediction).
        canvas = self.segment_frames+native if segment else CANVAS
        values, mask, path, keyframes, dense = self.conditions(phase, obs, step, cut, canvas)
        if not native:
            # Frame zero is now: its pose is known, its velocities and contacts are left to the model.
            values[0, :self.pose_end], mask[0, :self.pose_end] = features[-1, :self.pose_end], True
        generator = torch.Generator(device=self.model.device).manual_seed(_episode_noise_seed(self.seed, self.plan_index))
        begin = time.perf_counter()
        if native:
            # Official multi-prompt continuation of the reference sent so far (already where the body
            # is): its last frames, ending now, in full; the plan's own conditions from now on.
            n = min(self.transition_frames, len(features))
            v, m, heading, origin = transition_condition(self.model.representation, features, canvas-1, transition=n,
                                                         values=values[1:], mask=mask[1:])
            feats, _ = sample_segment(self.model, v, m, first_heading=heading,
                text_feat=self.text_feat if self.walk_text_on(phase, obs) else None, text_weight=self.text_cfg,
                constraint_weight=self.constraint_cfg, steps=self.segment_steps, generator=generator)
            feats = feats.cpu().clone()
            feats[:, self.root_slot] += origin[0]
            feats[:, self.root_slot+2] += origin[1]
            decoded = decode(self.model.representation, feats[n-1:])   # frame 0 is now
            output = {'local_rot_mats': decoded['local_rot_mats'], 'root_positions': decoded['root_positions'],
                      'history_last_root_position': decoded['root_positions'][0].clone(), 'motion_features': feats[n-1:]}
            hist_mask = torch.zeros(0, dtype=torch.bool)
        elif segment:
            # The official way: the whole segment from noise, the present only as the frame-0 keyframe.
            cos_sin = features[-1, self.heading_slice]
            feats, decoded = sample_segment(self.model, values, mask, first_heading=float(torch.atan2(cos_sin[1], cos_sin[0])),
                text_feat=self.text_feat, text_weight=self.text_cfg, constraint_weight=self.constraint_cfg,
                steps=self.segment_steps, generator=generator)
            output = {'local_rot_mats': decoded['local_rot_mats'], 'root_positions': decoded['root_positions'],
                      'history_last_root_position': decoded['root_positions'][0].clone(), 'motion_features': feats}
        else:
            output = self.model.predict_future(str(request['task']).strip(), torch.zeros(1, 3, 1, 1), hist,
                history_feature_mask=hist_mask,
                text_feat=(torch.zeros(1, 4096) if self.text_feat is None else self.text_feat).to(self.model.device),
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
            previous_root_position=before[0], hand_binary=torch.zeros(len(output['local_rot_mats']), 2)).numpy()
        _, future_wrist, future_wrist_rotation = world_pose(all_actions, output['root_positions'])
        future_root = self.frame.world(output['root_positions'].numpy())
        path_world = self.frame.world(np.insert(path, 1, 0., axis=1))[:, :2]
        record = {'plan': self.plan_index, 'time': obs['time'], 'cut': cut, 'phase': phase, 'sent_steps': sent,
                  'keyframes': keyframes, 'dense_hand_frames': dense, 'root_frames': list(range(1, canvas)),
                  'pose_frames': list(range({'stop': STAND_FROM, 'lean': LEAN_FROM}.get(phase, canvas), canvas)),
                  'generator': self.generator_mode if segment or native else 'policy', 'canvas_frames': canvas,
                  'hand_constraints': 'continuous', 'history_source': 'reference',
                  'history_condition_frames': n if native else int(hist_mask.any(-1).sum()),
                  'transition_frames': n if native else None,
                  'text': ((self.walk_text or self.text) if self.walk_text_on(phase, obs) else None) if native else self.text,
                  'approach_gain': self.approach_gain,
                  'text_cfg': self.text_cfg if self.text else None,
                  'root_constraints': self.root_constraints if phase == 'walk' else 'dense',
                  'inference_s': elapsed, 'constraint_cfg': self.constraint_cfg,
                  'diffusion_steps': self.segment_steps if segment or native else self.steps,
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
