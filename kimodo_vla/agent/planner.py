"""From one body command to the 40-d reference the tracker executes.

``Planner`` is the seam: ``start`` takes one validated command and the measured history, and
returns what the command asks for in world terms (the commanded target, logged for link 3);
``chunk`` returns the next rows of 40-d reference, or None once the command is complete. The
loop (``loop.py``) never looks inside. A kinematic planner would be another subclass.

``KimodoPlanner`` is built from the existing Kimodo building blocks, one per command type:

  walk, turn       the official-way Kimodo walk validated in experiment two
                   (``oracle.grasp_walk``: native generator, 1 s replans of 5 s segments from the
                   measured position, approach gain 2, one stop plan into the standing pose);
                   a turn is the same walker with an on-the-spot yaw goal
  pelvis_height,   one Kimodo plan into the pose ``posture.posture_pose`` solves (the walker's
  torso_pitch      lean phase with that pose as its end); new
  hand_pose        the standing IK-hold backend of the box grasp (``oracle.policy`` with continuous
                   hand conditions and IK projection, the wrist path of ``oracle.grasp_policy``):
                   the named wrists move along a minimum-jerk path to their targets, the others
                   hold their measured pose
  hand             the hand bits change; the body holds the reference it is on

Every chunk carries the hand state in columns 38-39 (>= 0.5 closes, as the simulator reads it).
While either hand is closed, walks and turns keep the arm and waist joints, and posture moves
the arm joints, at the reference they had when the command began, so whatever is held stays
where it is relative to the torso; the setting ``hold_arms_when_closed`` switches this off.
"""
from pathlib import Path

import numpy as np

from kimodo_vla.core.types import G1_JOINT_NAMES, yaw_matrix
from kimodo_vla.oracle.geometry import BASIS, SIDES, rotation_wxyz
from kimodo_vla.oracle.grasp_policy import WristPath
from kimodo_vla.oracle.grasp_walk import DEFAULTS as WALK_DEFAULTS, NativeApproach, walk_goal_xy
from kimodo_vla.oracle.locomotion import load_goal
from kimodo_vla.oracle.policy import OraclePolicy
from .commands import wrist_rotation
from .posture import posture_pose
from .state import ARM_JOINTS, WAIST_JOINTS, body_state, yaw_deg

SETTINGS = {
    'diffusion_steps': 50, 'constraint_cfg': 2., 'joint_speed': 4., 'handoff_seconds': .2,
    'walk': {},                     # overrides of oracle.grasp_walk.DEFAULTS
    'turn_rate_deg_s': 45.,
    'turn_gain': 2.,                # turns aim this many times the yaw still to go (the walk's approach gain, for yaw)
    'hand_settle_s': .5,            # a hand_pose command ends this long after its path does
    'hand_toggle_s': .6,            # an open / close command holds the body this long
    'wrist_feedback': False,        # off: the measured wrist error is not fed back (replacing link 3 is experiment 4's job)
    'wrist_feedback_gain': .7, 'wrist_feedback_max_m': .12,
    'wrist_feedback_s': 1.,         # with feedback on, a hand_pose command runs this much longer, so the correction acts
    'hold_arms_when_closed': True,
    'reach_settle_s': [.3, 2.],     # before a hand_pose: hold at least / at most this long until the body is still
    'settled_root_speed_mps': .02, 'settled_foot_speed_mps': .01,   # "still", as the first batch's grasp loop
}


def hold_rows(obs, heading, count, hands):
    """``count`` rows repeating the reference the tracker is on: no root motion, same posture."""
    rotation = heading.T @ rotation_wxyz(obs['reference_root_quat'])
    row = np.r_[0., 0., obs['reference_root_pos'][2], rotation[:, :2].ravel(), obs['reference_joints'], hands]
    return np.tile(row, (count, 1))


class Planner:
    """Interface. ``start(command, context)`` -> plan record with ``target``; ``chunk(request)`` -> rows or None."""
    name = 'abstract'

    def start(self, command, context):
        raise NotImplementedError

    def chunk(self, request):
        raise NotImplementedError


class HandReach(OraclePolicy):
    """The box grasp's standing reach, re-armed for every hand_pose command."""

    def arm(self, obs, prior, heading, goals, duration, settle_s, feedback=None):
        self.reset(self.seed)
        self.history = list(prior)[-180:]
        self.initialize(obs, heading=heading)
        wrists, rotations = np.asarray(obs['wrist_pos'], dtype=float), np.asarray(obs['wrist_rot'], dtype=float)
        # The virtual hand-end point keeps its measured offset in the wrist link's frame.
        self.end_local = np.einsum('sji,sj->si', rotations, self.targets[:, 1]-self.targets[:, 0])
        positions, turned = wrists.copy(), rotations.copy()
        for i, side in enumerate(SIDES):
            if side in goals:
                positions[i], turned[i] = goals[side]
        self.path = WristPath(obs['time'], wrists, rotations)
        self.path_end = self.path.to(positions, turned, duration)
        self.done_time = self.path_end+settle_s
        self.bias, self.feedback = np.zeros((2, 3)), feedback
        return positions, turned

    def goal(self, time):
        positions, rotations = self.path(time)
        return positions+self.bias, rotations

    def path_goals(self, elapsed):
        positions, rotations = zip(*(self.goal(self.start_time+e) for e in np.atleast_1d(elapsed)))
        positions, rotations = np.array(positions), np.array(rotations)
        ends = positions+np.einsum('tsij,sj->tsi', rotations, self.end_local)
        return self.frame.points(np.stack([positions, ends], axis=2)), self.frame.rotations(rotations)

    def aim(self, now):
        self.arrival = min(self.path.piece_end(now), now+1.5)
        wrists, rotations = self.goal(self.arrival)
        self.targets = np.stack([wrists, wrists+np.einsum('sij,sj->si', rotations, self.end_local)], axis=1)
        self.target_rot_world, self.target_rot_model = rotations, self.frame.rotations(rotations)
        self.target_info.update(mode='hand-pose', targets_world=self.targets.tolist(),
                                wrist_rotation_world=rotations.tolist(), arrival_time=self.arrival)

    def correct(self, obs):
        if not self.feedback or obs['time'] < self.path_end-1e-6:
            return
        wanted, _ = self.path(obs['time'])
        bias = self.bias+self.feedback['gain']*(wanted-np.asarray(obs['wrist_pos']))
        size = np.linalg.norm(bias, axis=1, keepdims=True)
        self.bias = bias*np.minimum(1., self.feedback['max_m']/np.maximum(size, 1e-9))

    def step(self, request):
        obs = request['observation']['oracle_history'][-1]
        if obs['time'] >= self.done_time-1e-6:
            return None
        self.correct(obs)
        self.aim(obs['time'])
        return np.asarray(self.__call__(request)['action_chunk'])


class TurnWalk(NativeApproach):
    """The walker, whose heading conditions aim ``yaw_gain`` times the yaw still to go during a turn.

    The tracker stays several degrees behind the reference heading, and Kimodo's reference stops a
    few degrees short of the conditioned one, so a turn aimed exactly at its goal stalls outside
    the 5 degree tolerance until the time allowance ends (2026-11-26 smoke runs). The walk's
    validated setting answers the same shortfall in position with approach gain 2; this is that
    gain for yaw. When the turn is done is still judged on the yaw actually turned.
    """
    yaw_gain = 1.

    def conditions(self, phase, obs, step, cut, canvas=None):
        kwargs = {} if canvas is None else {'canvas': canvas}
        if phase != 'walk' or self.yaw_gain == 1. or not self.goal.get('root', {}).get('yaw_deg'):
            return super().conditions(phase, obs, step, cut, **kwargs)
        remaining = NativeApproach.yaw_remaining
        self.yaw_remaining = lambda o: self.yaw_gain*remaining(self, o)
        try:
            return super().conditions(phase, obs, step, cut, **kwargs)
        finally:
            del self.yaw_remaining


class PostureMove(NativeApproach):
    """The walker's lean phase, ending in the pose ``posture_pose`` solves instead of a plain lean."""
    posture = {}

    def initialize(self, sample):
        super().initialize(sample)
        joints, root, rotation, info = posture_pose(self.fk, self.initial_joints, self.initial_root, self.initial_rotation,
            self.posture.get('height_m'), self.posture.get('pitch_deg'), self.posture.get('behind_feet_m'))
        pose = self.features.codec.decode_joint_configuration_pose(joints[None], self.frame.points([root]) @ BASIS,
            root_rotation_matrices=np.array([self.frame.heading.T @ rotation]), joint_names=G1_JOINT_NAMES)
        self.lean_features = self.features.features(pose['local_rot_mats'], pose['root_positions'])[0]
        self.posture_solution = {'joints': joints.tolist(), 'root_world': root.tolist(),
                                 'root_rotation_world': rotation.tolist(), **info}
        self.target_info['lean_pose'] = self.posture_solution
        self.save_target()


def make_native(host, cls, text_cache, settings, goal):
    """A Locomotion-type executor sharing ``host``'s loaded model (as ``grasp_walk.make_walker``)."""
    s = {**WALK_DEFAULTS, **{k: v for k, v in (settings or {}).items() if k != '_about'}}
    denoiser = host.model.denoiser
    try:
        executor = cls(None, text_cache, host.run, 'hold', model=host.model, generator='native', motion_goal=goal,
            walk_steps=s['walk_steps'], segment_seconds=s['segment_seconds'], segment_steps=s['segment_steps'],
            transition_frames=s['transition_frames'], approach_gain=s['approach_gain'],
            constraint_cfg=host.constraint_cfg, variant='bounded', handoff_seconds=host.handoff_seconds,
            joint_speed=host.joint_speed)
    finally:
        host.model.denoiser = denoiser
    executor.native_settings = s
    return executor


def prime(executor, *, run, goal, heading, history, seed):
    """Point a Locomotion-type executor at ``goal`` and hand it the recent measured history."""
    executor.goal = load_goal(goal)
    executor.run = Path(run)
    executor.run.mkdir(parents=True, exist_ok=True)
    executor.fixed_heading = np.asarray(heading, dtype=float)
    executor.noise_seed = seed
    executor.reset(seed)
    executor.primed = list(history)[-executor.native_settings['prime_samples']:]


class KimodoPlanner(Planner):
    name = 'kimodo'

    def __init__(self, checkpoint, text_cache, run, settings=None, device='cuda'):
        self.settings = {**SETTINGS, **(settings or {})}
        s = self.settings
        self.run = Path(run)
        self.reacher = HandReach(checkpoint, text_cache, self.run, 'hold', device=device, diffusion_steps=s['diffusion_steps'],
            constraint_cfg=s['constraint_cfg'], variant='bounded', handoff_seconds=s['handoff_seconds'],
            joint_speed=s['joint_speed'], support_constraints='root', hand_constraints='continuous',
            trajectory_projection='ik-hold')
        self.fk = self.reacher.fk
        self.walker = make_native(self.reacher, TurnWalk, text_cache, s['walk'],
                                  {'name': 'command-walk', 'root': {'xy': [.1, 0.]}})
        self.poser = make_native(self.reacher, PostureMove, text_cache, s['walk'],
                                 {'name': 'command-posture', 'lean': {'pitch_deg': 1.}})
        self.active = self.settle = None

    # ---- the interface -----------------------------------------------------------------
    def start(self, command, context):
        """``context``: history (samples, the current one last), hands [l, r], heading (3x3), folder, seed."""
        history, heading = context['history'], np.asarray(context['heading'], dtype=float)
        obs = history[-1]
        self.hands = np.asarray(context['hands'], dtype=float)
        self.heading, self.folder, self.seed = heading, Path(context['folder']), int(context['seed'])
        self.held = np.asarray(obs['reference_joints'], dtype=float)
        self.first = True
        kind = command['type']
        now = body_state(obs, self.fk)
        target, plan = {}, {}
        if kind in ('walk', 'turn'):
            if kind == 'walk':
                yaw = np.radians(now['yaw_deg'])
                goal_xy = np.asarray(obs['root_pos'][:2])+yaw_matrix(yaw)[:2, :2] @ [command['forward_m'], command['left_m']]
                primed_from = history[-self.walker.native_settings['prime_samples']:][0]['root_pos']
                goal = {'name': 'command-walk', 'root': {'xy': walk_goal_xy(primed_from, goal_xy, heading).tolist(),
                        'speed_m_s': self.walker.native_settings['speed_m_s'],
                        'tolerance_m': self.walker.native_settings['tolerance_m']}}
                target = {'root_xy': goal_xy.tolist(), 'yaw_deg': now['yaw_deg']}
            else:
                goal = {'name': 'command-turn', 'root': {'xy': [0., 0.], 'yaw_deg': command['left_deg'],
                        'yaw_rate_deg_s': self.settings['turn_rate_deg_s'], 'tolerance_m': self.walker.native_settings['tolerance_m']}}
                target = {'root_xy': obs['root_pos'][:2], 'yaw_deg': (now['yaw_deg']+command['left_deg']+180.) % 360.-180.}
            self.walker.yaw_gain = float(self.settings['turn_gain']) if kind == 'turn' else 1.
            prime(self.walker, run=self.folder, goal=goal, heading=heading, history=history, seed=self.seed)
            self.active, plan = ('native', self.walker, ARM_JOINTS+WAIST_JOINTS), {'walker_goal': goal}
        elif kind in ('pelvis_height', 'torso_pitch'):
            posture = ({'height_m': command['height_m'], 'behind_feet_m': command.get('behind_feet_m')}
                       if kind == 'pelvis_height' else {'pitch_deg': command['pitch_deg']})
            self.poser.posture = posture
            prime(self.poser, run=self.folder, goal={'name': 'command-posture', 'lean': {'pitch_deg': 1.}},
                  heading=heading, history=history, seed=self.seed)
            self.active = ('native', self.poser, ARM_JOINTS)
            target = {'pelvis_height_m': command['height_m']} if kind == 'pelvis_height' else {'torso_pitch_deg': command['pitch_deg']}
            if kind == 'pelvis_height' and command.get('behind_feet_m') is not None:
                target['behind_feet_m'] = command['behind_feet_m']
            plan = {'posture': posture}
        elif kind == 'hand_pose':
            # The first batch's grasp loop lets the body settle after a walk before the hands move.
            low, high = self.settings['reach_settle_s']
            self.settle = (obs['time']+low, obs['time']+high)
            goals = {}
            for side, t in command['hands'].items():
                rotation = (wrist_rotation(side, t['palm'], t['fingers']) if 'palm' in t
                            else np.asarray(now['wrists'][side]['rotation']))
                goals[side] = (np.asarray(t['position'], dtype=float), rotation)
            feedback = ({'gain': self.settings['wrist_feedback_gain'], 'max_m': self.settings['wrist_feedback_max_m']}
                        if self.settings['wrist_feedback'] else None)
            settle = self.settings['hand_settle_s']+(self.settings['wrist_feedback_s'] if feedback else 0.)
            self.reacher.run = self.folder
            self.folder.mkdir(parents=True, exist_ok=True)
            self.recent = list(history)[:len(history)-context.get('new_samples', 1)]
            self.active = ('reach', self.reacher, None)
            self.pending = (command, history, heading, goals, settle, feedback)
            target = {'wrists': {side: {'position': goals[side][0].tolist(), 'rotation': goals[side][1].tolist()}
                                 for side in goals}}
            plan = {'settle_s': self.settings['reach_settle_s'], 'duration_s': command['duration_s'], 'feedback': feedback}
        elif kind == 'hand':
            for i, side in enumerate(SIDES):
                if side in command:
                    self.hands[i] = 1. if command[side] == 'close' else 0.
            self.active = ('hold', obs['time']+self.settings['hand_toggle_s'], None)
            target = {'hands_closed': [bool(h) for h in self.hands]}
        else:
            raise ValueError(f'The planner does not execute {kind!r}')
        return {'target': target, 'plan': plan, 'hands_closed': [bool(h) for h in self.hands]}

    def chunk(self, request):
        obs = request['observation']['oracle_history'][-1]
        mode, executor, keep = self.active
        if mode == 'hold':
            if obs['time'] >= executor-1e-6:
                return None
            count = int(np.clip(round((executor-obs['time'])*50), 1, 25))
            return hold_rows(obs, self.heading, count, self.hands)
        if mode == 'reach':
            if self.settle is not None:
                new = request['observation']['oracle_history']
                self.recent = (self.recent+list(new))[-400:]
                if obs['time'] < self.settle[0]-1e-6 or (obs['time'] < self.settle[1]-1e-6 and not self.still()):
                    return hold_rows(obs, self.heading, 5, self.hands)
                command, _, heading, goals, settle, feedback = self.pending
                self.settle = None
                executor.arm(obs, self.recent[:-len(new)], heading, goals, command['duration_s'], settle, feedback)
            rows = executor.step(request)
        else:
            if self.first:
                request = {**request, 'observation': {**request['observation'], 'oracle_history': executor.primed}}
            rows = np.asarray(executor(request)['action_chunk'], dtype=float)
            if executor.phase == 'hold':      # the stop (or posture) plan has been executed whole
                return None
        self.first = False
        if rows is None:
            return None
        rows = np.array(rows, dtype=float)
        if keep is not None and self.settings['hold_arms_when_closed'] and self.hands.any():
            rows[:, [9+j for j in keep]] = self.held[keep]
        rows[:, 38:40] = self.hands
        return rows

    def still(self):
        """Pelvis and both ankles nearly at rest over the last 0.3 s (the first batch's ``settled``)."""
        h = self.recent[-15:]
        if len(h) < 15:
            return False
        dt = h[-1]['time']-h[0]['time']
        root = np.linalg.norm(np.subtract(h[-1]['root_pos'], h[0]['root_pos']))/dt
        feet = np.linalg.norm(np.subtract(h[-1]['foot_pos'], h[0]['foot_pos']), axis=1).max()/dt
        return bool(root < self.settings['settled_root_speed_mps'] and feet < self.settings['settled_foot_speed_mps'])

    def hold(self, obs, count):
        return hold_rows(obs, self.heading, count, self.hands)


def end_states(sample, fk):
    """Reference and measured body state at one sample: the planner's reference end and the measured end."""
    return {'reference': body_state(sample, fk, 'reference'), 'measured': body_state(sample, fk, 'measured')}


__all__ = ['Planner', 'KimodoPlanner', 'SETTINGS', 'hold_rows', 'end_states', 'yaw_deg']
