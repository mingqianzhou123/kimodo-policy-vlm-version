"""The body state the command set talks about, read off one simulator sample, measured or reference.

A sample is one row of the simulator's measured history (``agentic/arena/oracle_bridge.sample``):
measured root, joints and wrists, and the reference the tracker was consuming at that step
(``reference_root_pos``, ``reference_root_quat``, ``reference_joints``). ``body_state`` gives the
same quantities for either, so a command's target, the planner's reference end state and the
measured end state compare one to one (link 3 of the five-link judge).
"""
import numpy as np

from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.oracle.geometry import SIDES, rotation_wxyz
from .commands import palm_and_fingers
from .posture import torso_pitch_deg

WRISTS = [f'{side}_wrist_yaw_link' for side in SIDES]


def yaw_deg(rotation):
    rotation = np.asarray(rotation)
    return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))


def body_state(sample, fk, which='measured'):
    """Root position, heading, pelvis height, torso pitch and both wrists, as plain lists.

    ``which``: 'measured' (simulator truth) or 'reference' (the pose the tracker was asked for).
    """
    if which == 'measured':
        root, quat, joints = sample['root_pos'], sample['root_quat'], sample['joints']
    elif which == 'reference':
        root, quat, joints = sample['reference_root_pos'], sample['reference_root_quat'], sample['reference_joints']
    else:
        raise ValueError(which)
    root, joints = np.asarray(root, dtype=float), np.asarray(joints, dtype=float)
    rotation = rotation_wxyz(quat)
    if which == 'measured' and 'wrist_pos' in sample:
        positions, rotations = np.asarray(sample['wrist_pos']), np.asarray(sample['wrist_rot'])
    else:
        positions, rotations = fk(joints, root, rotation, body_names=WRISTS)
    wrists = {}
    for i, side in enumerate(SIDES):
        palm, fingers = palm_and_fingers(side, rotations[i])
        wrists[side] = {'position': positions[i].tolist(), 'rotation': np.asarray(rotations[i]).tolist(),
                        'palm': palm.tolist(), 'fingers': fingers.tolist()}
    return {'root': root.tolist(), 'yaw_deg': yaw_deg(rotation), 'pelvis_height_m': float(root[2]),
            'torso_pitch_deg': torso_pitch_deg(fk, joints, root, rotation), 'wrists': wrists}


def hands_closed(sample):
    """[left, right] gripper command as last applied by the simulator."""
    return [bool(x) for x in sample.get('gripper_command', [0, 0])]


def joint_index(*fragments):
    return [i for i, n in enumerate(G1_JOINT_NAMES) if any(f in n for f in fragments)]


ARM_JOINTS = joint_index('shoulder', 'elbow', 'wrist')
WAIST_JOINTS = joint_index('waist')
