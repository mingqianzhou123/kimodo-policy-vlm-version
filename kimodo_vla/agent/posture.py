"""Pelvis height and torso pitch with the feet where they stand (new; nothing else uses it yet).

``posture_pose`` returns the joints and root pose of a standing pose whose pelvis is at a given
height above the floor and whose torso is pitched forward by a given angle, with the left foot
exactly where it was (position and orientation) and the right foot following by symmetry. Four
joint groups move, the same amount on both sides: hip pitch, knee, ankle pitch, and the waist
pitch. The root pose is solved from the left foot, as ``locomotion.leaned_pose`` does. Three
conditions fix them: the pelvis height, the torso pitch, and balance, taken as a mass proxy (half
on the pelvis, half on a point 0.15 m up the torso) staying as far ahead of the ankles as it was
in the starting pose, or, with ``behind_feet_m``, the pelvis that far behind the midpoint of the ankles
(a sit: the seat, not the feet, will carry the weight). The planner then has Kimodo generate the way to this pose (``planner``).

Measurement helpers: ``pelvis_height`` and ``torso_pitch_deg`` read the same quantities off any
pose, so a command's target, the planner's reference end and the measured end compare one to one.
"""
import numpy as np
from scipy.optimize import least_squares

from kimodo_vla.oracle.geometry import SIDES

FEET = [f'{side}_ankle_roll_link' for side in SIDES]
UPPER_OFFSET_M = .15   # the upper-body mass point, this far up the torso from the torso link origin
PELVIS_SHARE = .5      # share of the mass proxy on the pelvis


def heading_of(rotation):
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def torso_pitch_deg(fk, joints, root, rotation):
    """Forward tilt of the torso link's up axis from vertical, degrees, in the root's heading frame."""
    _, torso_rotation = (x[0] for x in fk(joints, root, rotation, body_names=['torso_link']))
    up = heading_of(rotation).T @ torso_rotation[:, 2]
    return float(np.degrees(np.arctan2(up[0], up[2])))


def pelvis_height(root):
    return float(np.asarray(root)[2])


def mass_proxy(fk, joints, root, rotation):
    torso, torso_rotation = (x[0] for x in fk(joints, root, rotation, body_names=['torso_link']))
    upper = torso + torso_rotation[:, 2]*UPPER_OFFSET_M
    return PELVIS_SHARE*np.asarray(root) + (1-PELVIS_SHARE)*upper


def posture_pose(fk, joints, root, rotation, height_m=None, pitch_deg=None, behind_feet_m=None):
    """(joints, root, rotation, info) of the pose with the pelvis at ``height_m`` and the torso at
    ``pitch_deg``; either left as None keeps its present value. Raises ValueError if the solve leaves
    more than 1 cm or 1 degree, or the pose is outside the joint limits."""
    names = list(fk.joint_names)
    groups = [[names.index(f'{side}_hip_pitch_joint') for side in SIDES],
              [names.index(f'{side}_knee_joint') for side in SIDES],
              [names.index(f'{side}_ankle_pitch_joint') for side in SIDES],
              [names.index('waist_pitch_joint')]]
    joints, root, rotation = np.array(joints, dtype=float), np.asarray(root, dtype=float), np.asarray(rotation, dtype=float)
    heading = heading_of(rotation)
    foot, foot_rotation = (x[0] for x in fk(joints, root, rotation, body_names=FEET[:1]))
    feet_mid = np.mean(fk(joints, root, rotation, body_names=FEET)[0], axis=0)
    height_m = pelvis_height(root) if height_m is None else float(height_m)
    pitch_deg = torso_pitch_deg(fk, joints, root, rotation) if pitch_deg is None else float(pitch_deg)
    ahead = (heading.T @ (mass_proxy(fk, joints, root, rotation)-feet_mid))[0]

    def apply(delta):
        q = joints.copy()
        for group, d in zip(groups, delta):
            q[group] += d
        return q

    def place(q):
        """Root pose that puts the left foot of joints ``q`` back where it stood."""
        at, turned = (x[0] for x in fk(q, np.zeros(3), np.eye(3), body_names=FEET[:1]))
        new_rotation = foot_rotation @ turned.T
        return foot-new_rotation @ at, new_rotation

    def residual(delta):
        q = apply(delta)
        r, R = place(q)
        mid = np.mean(fk(q, r, R, body_names=FEET)[0], axis=0)
        if behind_feet_m is None:
            balance = (heading.T @ (mass_proxy(fk, q, r, R)-mid))[0]-ahead
        else:
            balance = (heading.T @ (r-mid))[0]+float(behind_feet_m)
        return np.r_[(r[2]-height_m)*10., np.radians(torso_pitch_deg(fk, q, r, R)-pitch_deg)*3., balance*10., .01*np.asarray(delta)]

    limits = fk.joint_limits
    low = [max(limits[g, 0]-joints[g]).item() + 1e-3 for g in groups]
    high = [min(limits[g, 1]-joints[g]).item() - 1e-3 for g in groups]
    low, high = np.minimum(low, 0.), np.maximum(high, 0.)
    solution = least_squares(residual, np.zeros(4), bounds=(low, high), xtol=1e-10, ftol=1e-10)
    q = apply(solution.x)
    r, R = place(q)
    right_before = fk(joints, root, rotation, body_names=FEET[1:])[0][0]
    right_after = fk(q, r, R, body_names=FEET[1:])[0][0]
    info = {'height_error_m': float(r[2]-height_m), 'pitch_error_deg': torso_pitch_deg(fk, q, r, R)-pitch_deg,
            'right_foot_shift_m': float(np.linalg.norm(right_after-right_before)), 'joint_deltas_rad': solution.x.tolist(),
            'pelvis_shift_xy_m': (r-root)[:2].tolist(),
            'pelvis_behind_feet_m': float(-(heading.T @ (r-np.mean(fk(q, r, R, body_names=FEET)[0], axis=0)))[0])}
    off = 0. if behind_feet_m is None else info['pelvis_behind_feet_m']-float(behind_feet_m)
    if abs(info['height_error_m']) > .01 or abs(info['pitch_error_deg']) > 1. or abs(off) > .01:
        raise ValueError(f'Posture out of reach: {info}')
    if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
        raise ValueError('Posture exceeds joint limits')
    return q, r, R, info
