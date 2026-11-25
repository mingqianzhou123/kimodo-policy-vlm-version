"""Blocking one-handed hammer grasp on the frozen Kimodo + SONIC backend (DoubleDesk).

The same blocking round as the box grasp (``grasp_policy.BoxGraspPolicy``): latch the hammer
pose once, wait a fixed delay, bring the right hand over the handle, let the simulator-side
gate close the right hand once the wrist has arrived, lift, hold; on failure release and start
another round from wherever things are. The robot never walks: it starts in front of the table
and every round reaches from where it stands. The left arm keeps the pose it had at the anchor.

Where the hand takes the handle is fixed by the handle's pose alone (``grasp_pose``): the hand
is centred on a point of the handle axis, comes from above, palm down and pitched forward,
with the finger-curl plane across the handle, so that the closing fingers sweep its far side
towards the thumb on its near side.
"""
import numpy as np

from kimodo_vla.core.types import yaw_of
from .geometry import rotation_wxyz
from .grasp_policy import BoxGraspPolicy, WristPath

UP = np.array([0., 0., 1.])


def handle_frame(root_pos, root_quat, cfg):
    """Grip point on the handle axis [3] and the horizontal unit handle axis [3], head-ward.

    ``root_pos`` and ``root_quat`` are the hammer's rigid-body pose (its asset origin sits at
    the centre of its bounding box; the handle runs along its local +y towards the head).
    """
    rotation = rotation_wxyz(root_quat)
    point = np.asarray(root_pos, dtype=float) + rotation @ np.asarray(cfg['grip_point_local_m'], dtype=float)
    axis = rotation[:, 1].copy()
    axis[2] = 0.
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        raise ValueError('The handle stands upright')
    return point, axis / norm


def grasp_rotation(axis, pitch_down_deg):
    """Right-wrist rotation (columns: finger direction, palm normal, index side) over a handle.

    The fingers point away from the robot across the handle, tipped ``pitch_down_deg`` below
    horizontal; the palm faces down and back; the finger-curl axis (wrist z) runs along the
    handle towards its head. Dex3 fingers curl about wrist z towards wrist +y.
    """
    axis = np.asarray(axis, dtype=float)
    across = np.array([axis[1], -axis[0], 0.])   # the handle axis turned 90 degrees clockwise
    pitch = np.radians(pitch_down_deg)
    fingers = np.cos(pitch) * across - np.sin(pitch) * UP
    palm = -np.sin(pitch) * across - np.cos(pitch) * UP
    return np.stack([fingers, palm, axis], axis=1)


def grasp_pose(root_pos, root_quat, heading, cfg):
    """Right-wrist waypoints of one attempt in world coordinates.

    ``grip``: the handle point ``grip_point_local_m`` sits at ``grip_in_hand_m`` of the wrist
    frame (between the open fingers and the open thumb), raised by ``grip_height_m``.
    ``pre``: the same pose ``approach_up_m`` higher, from where the hand comes straight down.
    ``lift``: the grip pose ``lift_m`` higher. ``heading`` is the robot's 3x3 heading, used
    only to check that the hand approaches from the robot's side of the handle.
    """
    point, axis = handle_frame(root_pos, root_quat, cfg)
    forward = np.asarray(heading, dtype=float)[:, 0]
    across = np.array([axis[1], -axis[0], 0.])
    if across @ forward < 0:
        # The head points to the robot's right: the right hand would come from the far side.
        raise ValueError('Handle turned more than 90 degrees from the robot\'s left')
    rotation = grasp_rotation(axis, cfg['wrist_pitch_down_deg'])
    in_hand = np.array([cfg['grip_in_hand_m'][0], cfg['grip_in_hand_m'][1], 0.])
    grip = point - rotation @ in_hand + UP * cfg['grip_height_m']
    pre = grip + UP * cfg['approach_up_m'] - across * cfg['approach_back_m']
    return {'grip': grip, 'pre': pre, 'lift': grip + UP * cfg['lift_m'], 'rotation': rotation,
            'grip_point': point, 'handle_axis': axis, 'handle_yaw_deg': handle_yaw_deg(axis, heading)}


def handle_yaw_deg(axis, heading):
    """Handle direction relative to the robot's left, degrees; positive turns the head towards the robot."""
    heading = np.asarray(heading, dtype=float)
    return float(np.degrees(np.arctan2(-(np.asarray(axis) @ heading[:, 0]), np.asarray(axis) @ heading[:, 1])))


def graspable(com, quat, cfg, edge_distance):
    """The hammer still lies flat on the table, within reach of the table edge."""
    up = rotation_wxyz(quat)[2, 2]
    low, high = cfg['graspable_height_m']
    return bool(low <= com[2] <= high and abs(up) > np.cos(np.radians(cfg['graspable_tilt_deg']))
                and edge_distance <= cfg['graspable_depth_m'])


class HammerGraspPolicy(BoxGraspPolicy):
    """Right-handed blocking grasp; stand-still only. Phases as the box: wait, raise, advance
    (over the handle, then down to it), grip, lift, hold, release; the squeeze phase is unused."""

    def __init__(self, *args, config, latch_delay_steps=75, stand_still=True, **kwargs):
        if not stand_still:
            raise ValueError('The hammer grasp never walks')
        super().__init__(*args, config=config, latch_delay_steps=latch_delay_steps, stand_still=True, **kwargs)

    # ---- table geometry in the robot's frame ---------------------------------------------
    def edge_distance(self, point):
        """How far a world point lies behind the table's near edge, along the robot's heading."""
        c = self.cfg
        forward = self.episode_heading[:2, 0]
        return float((np.asarray(point[:2]) - np.asarray(c['table_edge_point_m'])) @ forward)

    def begin_round(self, sample):
        index = 0 if self.round is None else self.round.index + 1
        from .grasp_policy import Round
        self.round = Round(index, sample['time'], list(sample['box_com']), list(sample['box_quat']),
                           phase_time=sample['time'])
        self.round.notes['latched_root_pos'] = list(sample['box_pos'])
        self.round.graspable = graspable(sample['box_com'], sample['box_quat'], self.cfg,
                                         self.edge_distance(sample['box_com']))
        self.log('latch', graspable=self.round.graspable)

    def anchor(self, obs):
        """Start the standing backend where the robot stands and lay out this attempt."""
        self.initialize(obs, heading=self.episode_heading)
        c, r = self.cfg, self.round
        r.notes['anchor_reference_offset_m'] = float(np.linalg.norm(np.subtract(obs['reference_root_pos'], obs['root_pos'])))
        wrists, rotations = np.asarray(obs['wrist_pos']), np.asarray(obs['wrist_rot'])
        self.end_local = np.einsum('sji,sj->si', rotations, self.targets[:, 1] - self.targets[:, 0])
        self.rest = (wrists.copy(), rotations.copy())
        self.bias = np.zeros((2, 3))
        try:
            w = grasp_pose(r.notes['latched_root_pos'], r.box_quat, self.episode_heading, c)
        except ValueError as exc:
            r.notes['ungraspable'] = str(exc)
            r.graspable = False
            return False
        # Left: where it is. Right: up and over the table edge, above the handle, down onto it.
        via = w['pre'].copy()
        forward = self.episode_heading[:, 0]
        root = np.asarray(obs['root_pos'], dtype=float)
        ahead = (via - root) @ forward
        via += forward * (min(ahead, c['via_forward_m']) - ahead) + UP * c['via_raise_m']
        both = lambda right: np.stack([wrists[0], right])
        turn = lambda right: np.stack([rotations[0], right])
        self.way = {'via': both(via), 'pre': both(w['pre']), 'grip': both(w['grip']), 'lift': both(w['lift']),
                    'rotation': turn(w['rotation'])}
        self.path = WristPath(obs['time'], wrists, rotations)
        r.notes['raise_end'] = self.path.to(self.way['via'], self.way['rotation'], c['raise_s'])
        r.notes['over_end'] = self.path.to(self.way['pre'], self.way['rotation'], c['advance_s'])
        r.notes['advance_end'] = self.path.to(self.way['grip'], self.way['rotation'], c['descend_s'])
        r.event = {'id': f'round-{r.index}', 'positions': self.way['grip'].tolist(),
                   'rotations': self.way['rotation'].tolist(), 'deadline': r.notes['advance_end'] + c['gate_timeout_s']}
        r.notes.update(anchor_root=list(obs['root_pos']), grip_point=w['grip_point'].tolist(),
                       handle_axis=w['handle_axis'].tolist(), handle_yaw_deg=w['handle_yaw_deg'],
                       anchor_edge_clearance_m=-self.edge_distance(obs['root_pos']),
                       grip_depth_m=self.edge_distance(w['grip_point']),
                       wrist_targets={k: self.way[k][1].tolist() for k in ('via', 'pre', 'grip', 'lift')})
        return True

    def correct(self, obs):
        """Outer loop on the right wrist once it should be at the handle (see the box policy)."""
        c, r = self.cfg, self.round
        if r.phase != 'advance' or obs['time'] < r.notes['advance_end'] - 1e-6:
            return
        wanted, _ = self.path(obs['time'])
        bias = self.bias[1] + c['reach_feedback_gain'] * (wanted[1] - np.asarray(obs['wrist_pos'][1]))
        self.bias[1] = bias * min(1., c['reach_feedback_max_m'] / max(np.linalg.norm(bias), 1e-9))
        r.notes['wrist_bias'] = self.bias[1].tolist()

    def advance(self, samples):
        obs = samples[-1]
        now, c, r = obs['time'], self.cfg, self.round
        closed = bool(obs['gripper_command'][1])
        if r.phase == 'wait' and now >= r.latch_time + self.latch_delay/50 - 1e-6:
            r.notes['stand_still'] = True
            if not r.graspable or not self.anchor(obs):
                return self.enter('idle', now)
            return self.enter('raise', now)
        if r.phase == 'idle' and now >= r.phase_time + c['idle_s']:
            return self.begin_round(obs)
        if r.phase == 'raise' and now >= r.notes['raise_end'] - 1e-6:
            return self.enter('advance', now)
        if r.phase == 'advance':
            if closed:
                r.notes['lift_start'] = max(now + c['grip_s'], r.notes['advance_end'])
                # The closed hand holds the commanded grip pose (the measured correction stays).
                r.notes['lift_end'] = self.path.to(self.way['lift'], self.way['rotation'], c['lift_s'],
                                                   start=r.notes['lift_start'])
                return self.enter('grip', now)
            if now >= r.event['deadline']:
                return self.release(now, 'gate_timeout')
        if r.phase == 'grip' and now >= r.notes['lift_start'] - 1e-6:
            return self.enter('lift', now)
        if r.phase == 'lift' and now >= r.notes['lift_end'] - 1e-6:
            return self.enter('hold', now)
        if r.phase == 'hold' and now >= r.phase_time + c['hold_s']:
            return self.release(now, 'no_success_after_hold')
        if r.phase == 'release' and now >= r.notes['release_end'] - 1e-6:
            return self.begin_round(obs)

    def release(self, now, reason):
        """Open, go up and back over the edge, then to the arm's starting pose."""
        c, w, r = self.cfg, self.way, self.round
        here, _ = self.path(now)
        up = here.copy()
        up[1] += UP * c['release_up_m']
        self.path.to(up, w['rotation'], c['open_s'], start=now)
        self.path.to(w['via'], w['rotation'], c['retreat_s'])
        r.notes.update(release_reason=reason, release_end=self.path.to(*self.rest, c['raise_s']))
        self.enter('release', now)
