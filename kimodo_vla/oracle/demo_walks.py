"""Walking references that do not come from Kimodo: segments of the recorded Arena demonstrations.

The HumanoidArena SONIC demonstrations (``kimodo_vla.benchmarks.arena.demos``) store, at 50 Hz,
the whole-body reference SONIC followed when they were recorded. Their walking is retargeted
human motion, so it is the "known good" reference for testing the tracker on its own.

A segment is one stretch of an episode: a lead-in (whatever the demonstration did before) and
then a window of at least 2 s that walks one way: forward, backward, sideways left or right, or
turns on the spot. Windows are found from the reference's own root path, measured in the body's
heading frame at the window's start.

``Timeline`` turns a segment into the commands an episode sends: a 1 s minimum-jerk blend from
the measured standing pose into the segment's first frame, the segment, then its last frame held.
``DemoReference`` serves that timeline to the simulator, with no model loaded.

``stepping`` measures how a 50 Hz foot trajectory walks (contact slide, both feet moving, step
length, cadence); the same numbers are computed for demonstration windows, Kimodo rollouts and
the robot's measured feet.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from kimodo_vla.core.types import G1_JOINT_NAMES, yaw_matrix, yaw_of
from .geometry import XmlFK, rotation_wxyz

RATE = 50.
DIRECTIONS = ('forward', 'backward', 'left', 'right', 'turn', 'in_place')   # in_place: HITCH-planned body or hand motions (experiment2/heft)
BLEND_S = 1.       # standing pose -> first segment frame
SETTLE_S = 1.      # last frame held after the segment
FOOT_NAMES = ('left_ankle_roll_link', 'right_ankle_roll_link')
CONTACT_M = .025   # an ankle within this of its lowest height is on the ground
MOVING_M_S = .10   # both_moving: both ankles faster than this over the ground


def rot6d_rows(actions):
    """Root rotations [T, 3, 3] from the action layout's first two columns (indices 3:9)."""
    a = np.asarray(actions, dtype=float)[:, 3:9].reshape(-1, 3, 2)
    x = a[..., 0]/np.linalg.norm(a[..., 0], axis=-1, keepdims=True)
    y = a[..., 1]-(x*a[..., 1]).sum(-1, keepdims=True)*x
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], axis=-1)


def integrate_root(actions, rotations=None):
    """World root positions [T, 3] from body-frame xy steps, as Arena's decoder integrates them."""
    a = np.asarray(actions, dtype=float)
    r = rot6d_rows(a) if rotations is None else rotations
    out = np.zeros((len(a), 3))
    out[:, 2] = a[:, 2]
    for t in range(1, len(a)):
        local = np.array([a[t, 0], a[t, 1], 0.])
        if abs(r[t, 2, 2]) > 1e-6:
            local[2] = (a[t, 2]-a[t-1, 2]-r[t, 2, 0]*local[0]-r[t, 2, 1]*local[1])/r[t, 2, 2]
        out[t, :2] = out[t-1, :2]+(r[t] @ local)[:2]
    return out


def heading_frame_motion(root, rotations, begin, end):
    """(forward, left) displacement in the heading frame at ``begin``, and the yaw change in degrees."""
    yaw = np.unwrap(yaw_of(rotations))
    c, s = np.cos(yaw[begin]), np.sin(yaw[begin])
    d = root[end, :2]-root[begin, :2]
    return np.array([c*d[0]+s*d[1], -s*d[0]+c*d[1]]), float(np.degrees(yaw[end]-yaw[begin]))


def classify(displacement, yaw_deg, seconds):
    """Direction of one window, or None if it is mixed, standing or curved."""
    fwd, left = displacement
    distance = float(np.hypot(fwd, left))
    if abs(yaw_deg) >= 30*seconds and distance < .1*seconds:
        return 'turn'
    if abs(yaw_deg) > 10*seconds or distance < .1*seconds:
        return None
    if fwd > 0 and abs(left) < .25*fwd:
        return 'forward'
    if fwd < 0 and abs(left) < .25*-fwd:
        return 'backward'
    if abs(fwd) < .25*abs(left):
        return 'left' if left > 0 else 'right'
    return None


def find_windows(actions, window_s=2., hop_s=.2, min_height=.6, lead_s=3.4, max_s=4.):
    """Directional windows of one episode: dicts with direction, begin, end (frames), merged and capped.

    Consecutive 2 s windows of the same direction are merged into one, at most ``max_s`` long.
    A window needs ``lead_s`` of the episode before it (Kimodo's 100-frame history) and the root
    above ``min_height`` from there to its end (no sitting or crouching).
    """
    a = np.asarray(actions, dtype=float)
    rotations = rot6d_rows(a)
    root = integrate_root(a, rotations)
    w, hop, lead = int(window_s*RATE), int(hop_s*RATE), int(lead_s*RATE)
    labels = []
    for i in range(lead, len(a)-w, hop):
        d, yaw = heading_frame_motion(root, rotations, i, i+w)
        labels.append((i, classify(d, yaw, window_s)))
    out, k = [], 0
    while k < len(labels):
        begin, label = labels[k]
        j = k
        while j+1 < len(labels) and labels[j+1][1] == label and labels[j+1][0]+w-begin <= max_s*RATE:
            j += 1
        end = labels[j][0]+w
        if label and a[begin-lead:end+1, 2].min() >= min_height:
            out.append({'direction': label, 'begin': int(begin), 'end': int(end)})
        k = j+1
    return out


def window_summary(actions, begin, end):
    """Commanded motion of one window: displacement, yaw change, speed, purity."""
    a = np.asarray(actions, dtype=float)
    rotations = rot6d_rows(a)
    root = integrate_root(a, rotations)
    d, yaw = heading_frame_motion(root, rotations, begin, end)
    seconds = (end-begin)/RATE
    distance = float(np.hypot(*d))
    feet = feet_world(g1_fk(), a[begin:end+1, 9:38], root[begin:end+1], rotations[begin:end+1])
    legs = leg_displacement(feet, root[begin:end+1])
    c, s = np.cos(yaw_of(rotations[begin])), np.sin(yaw_of(rotations[begin]))
    legs = np.array([c*legs[0]+s*legs[1], -s*legs[0]+c*legs[1]])
    return {'displacement_m': d.tolist(), 'distance_m': distance, 'yaw_deg': yaw, 'seconds': seconds,
            'leg_displacement_m': legs.tolist(),
            'speed_m_s': distance/seconds, 'yaw_rate_deg_s': yaw/seconds,
            'min_height_m': float(a[begin:end+1, 2].min())}


def feet_world(fk, joints, root, rotations):
    """Ankle-roll link origins [T, 2, 3] for joint rows, root positions and root rotations."""
    return np.array([fk(q, p, r, body_names=list(FOOT_NAMES))[0] for q, p, r in zip(joints, root, rotations)])


def reference_feet(actions, fk=None):
    """Feet [T, 2, 3] of a reference in the action layout (its own integrated root path)."""
    a = np.asarray(actions, dtype=float)
    fk = fk or g1_fk()
    rotations = rot6d_rows(a)
    return feet_world(fk, a[:, 9:38], integrate_root(a, rotations), rotations)


def g1_fk():
    return XmlFK(Path(__file__).resolve().parents[1]/'skeleton/assets/g1skel34/xml/g1.xml', G1_JOINT_NAMES)


def stepping(feet, rate=RATE, contact_m=CONTACT_M, moving=MOVING_M_S, span_s=.1):
    """How a foot trajectory [T, 2, 3] walks.

    contact          ankle within ``contact_m`` of its own 5th-percentile height
    (all speeds are ground displacement over ``span_s`` divided by it)
    slide_m_s        mean ground speed of a foot in the planted part of a stance (0 for a clean foot)
    slide_p90_m_s    its 90th percentile
    both_moving      share of frames on which both ankles move faster than ``moving`` m/s
    steps            swings (contact lost for at least 0.1 s, then regained)
    step_length_m    median ground distance one foot moves in a swing (stance to next stance)
    cadence_hz       steps per second
    foot_travel_m    ground distance of both ankles
    """
    feet = np.asarray(feet, dtype=float)
    # Speeds over 0.1 s, not per frame: the recorded demonstrations repeat a frame now and then
    # and make up for it with a double step on the next, which per-frame speeds read as sliding.
    k = max(1, int(round(span_s*rate)))
    speed = (rate/k)*np.linalg.norm(feet[k:, :, :2]-feet[:-k, :, :2], axis=-1)   # [T-k, 2]
    height = feet[..., 2]-np.percentile(feet[..., 2], 5, axis=0)
    contact = height < contact_m
    # The planted part of a stance: in contact for 3 frames (0.06 s) either side, so the first and
    # last moments of a swing, when the ankle is low but already moving, do not count as sliding.
    planted = contact.copy()
    for shift in range(1, 4):
        planted[shift:] &= contact[:-shift]
        planted[:-shift] &= contact[shift:]
    over = np.ones((len(feet)-k, 2), dtype=bool)
    for shift in range(k+1):
        over &= planted[shift:len(feet)-k+shift]
    slide = speed[over]
    lengths, swings = [], 0
    for side in range(2):
        c = contact[:, side]
        t = 0
        while t < len(c):
            if c[t]:
                t += 1
                continue
            lift = t
            while t < len(c) and not c[t]:
                t += 1
            if t < len(c) and lift > 0 and t-lift >= int(.1*rate):
                swings += 1
                before = lift
                while before > 0 and c[before-1]:
                    before -= 1
                after = t
                while after < len(c) and c[after]:
                    after += 1
                # between the median positions of the stances either side of the swing
                lengths.append(float(np.linalg.norm(np.median(feet[t:after, side, :2], axis=0)
                                                    - np.median(feet[before:lift, side, :2], axis=0))))
    seconds = len(feet)/rate
    return {'slide_m_s': float(slide.mean()) if slide.size else None,
            'slide_p90_m_s': float(np.percentile(slide, 90)) if slide.size else None,
            'both_moving': float((speed > moving).all(axis=1).mean()),
            'steps': swings, 'step_length_m': float(np.median(lengths)) if lengths else None,
            'cadence_hz': swings/seconds, 'foot_travel_m': float(speed.sum()/rate),
            'contact_share': float(contact.mean()), 'seconds': seconds}


def leg_displacement(feet, root, contact_m=CONTACT_M):
    """Ground displacement [2] the legs alone would produce: the root's, minus what planted feet skate.

    Each frame the root's ground step is reduced by the mean ground step of the feet in contact
    (a planted foot that slides is ground the legs do not push the body over). Frames without a
    foot in contact keep the root's step. For a clean reference this equals the root displacement.
    """
    feet, root = np.asarray(feet, dtype=float), np.asarray(root, dtype=float)
    contact = feet[..., 2]-np.percentile(feet[..., 2], 5, axis=0) < contact_m
    step_root = np.diff(root[:, :2], axis=0)
    step_feet = np.diff(feet[..., :2], axis=0)
    both = contact[1:] & contact[:-1]
    out = np.zeros(2)
    for t in range(len(step_root)):
        out += step_root[t]-(step_feet[t][both[t]].mean(axis=0) if both[t].any() else 0.)
    return out


class Timeline:
    """Commands of one replay episode, in the action layout, on the 50 Hz control grid.

    ``segment`` holds ``actions`` [T, 40] (rotation relative to the segment's first heading) and
    ``window`` [begin, end] frames. Built from the first measured sample: blend, segment, hold.
    """
    def __init__(self, segment, sample, blend_s=BLEND_S, settle_s=SETTLE_S):
        a = np.array(segment['actions'], dtype=float)
        first = rot6d_rows(a[:1])[0]
        turn = yaw_matrix(float(yaw_of(first))).T      # the segment's first heading becomes "ahead"
        rotations = np.einsum('ij,tjk->tik', turn, rot6d_rows(a))
        a[:, 3:9] = rotations[:, :, :2].reshape(len(a), 6)
        a[:, 38:40] = 0.                                    # hands open, as in every other run
        measured = rotation_wxyz(sample['root_quat'])
        start_rot = yaw_matrix(float(yaw_of(measured))).T @ measured
        n = int(round(blend_s*RATE))
        u = np.arange(1, n+1)/n
        w = u**3*(10+u*(-15+6*u))
        joints = np.array(sample['joints'])[None]+w[:, None]*(a[0, 9:38]-np.array(sample['joints'])[None])
        slerp = Slerp([0., 1.], Rotation.from_matrix(np.stack([start_rot, rotations[0]])))
        blend_rot = slerp(w).as_matrix()
        height = sample['root_pos'][2]+w*(a[0, 2]-sample['root_pos'][2])
        blend = np.zeros((n, 40))
        blend[:, 2], blend[:, 3:9], blend[:, 9:38] = height, blend_rot[:, :, :2].reshape(n, 6), joints
        first_step = a[:1].copy()
        first_step[0, :2] = 0.                              # the blend's last frame is already there
        hold = np.tile(a[-1:], (int(round(settle_s*RATE)), 1))
        hold[:, :2] = 0.
        self.actions = np.concatenate([blend, first_step, a[1:], hold])
        self.segment_start = n                              # timeline index of segment frame 0
        begin, end = segment['window']
        self.window = (n+int(begin), n+int(end))
        self.length = len(self.actions)

    def chunk(self, index, size=25):
        """``size`` commands from timeline ``index``, the last one repeated past the end."""
        rows = self.actions[min(index, self.length-1):index+size]
        if len(rows) < size:
            pad = np.tile(self.actions[-1:], (size-len(rows), 1))
            pad[:, :2] = 0.
            rows = np.concatenate([rows, pad])
        return rows


# Joint order of the LAFAN1 G1 retargeting (huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset):
# 30 FPS rows of root x y z, quaternion x y z w, then these 29 joints.
LAFAN1_G1_JOINTS = tuple(f'{side}_{j}_joint' for side in ('left', 'right') for j in
                         ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')) + \
    ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint') + \
    tuple(f'{side}_{j}_joint' for side in ('left', 'right') for j in
          ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw'))


def lafan1_actions(rows, fps=30.):
    """LAFAN1 G1 rows [T, 36] -> actions [T', 40] at 50 Hz (root steps in the body frame, as Arena stores them)."""
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 36:
        raise ValueError('LAFAN1 G1 rows have 36 columns: root xyz, quaternion xyzw, 29 joints')
    t = np.arange(len(rows))/fps
    grid = np.arange(0., t[-1]+1e-9, 1/RATE)
    position = np.stack([np.interp(grid, t, rows[:, i]) for i in range(3)], axis=1)
    rotation = Slerp(t, Rotation.from_quat(rows[:, 3:7]))(grid).as_matrix()
    order = [LAFAN1_G1_JOINTS.index(n) for n in G1_JOINT_NAMES]
    joints = np.stack([np.interp(grid, t, rows[:, 7+i]) for i in order], axis=1)
    out = np.zeros((len(grid), 40))
    step = np.zeros_like(position)
    step[1:] = position[1:]-position[:-1]
    out[:, :2] = np.einsum('tji,tj->ti', rotation, step)[:, :2]
    out[:, 2] = position[:, 2]
    out[:, 3:9] = rotation[:, :, :2].reshape(len(grid), 6)
    out[:, 9:38] = joints
    return out


def load_segment(path):
    """A reference file: JSON with ``actions`` [T, 40], ``window`` [begin, end], ``direction``, ``source``."""
    segment = json.loads(Path(path).read_text())
    a = np.asarray(segment['actions'], dtype=float)
    begin, end = segment['window']
    if a.ndim != 2 or a.shape[1] != 40 or not np.isfinite(a).all() or not 0 <= begin < end < len(a):
        raise ValueError(f'{path}: expected actions [T, 40] and a window inside them')
    if segment.get('direction') not in DIRECTIONS:
        raise ValueError(f'{path}: direction must be one of {DIRECTIONS}')
    return segment


class DemoReference:
    """Serves one segment's timeline to the simulator; no model, no feedback."""
    def __init__(self, run, segment_path):
        self.run, self.path = Path(run), Path(segment_path)
        self.segment = load_segment(segment_path)
        self.reset()

    def reset(self, seed=None):
        self.timeline, self.start, self.plan = None, None, 0
        return {'status': 'ok'}

    def __call__(self, request):
        samples = request['observation']['oracle_history']
        if not samples:
            raise ValueError('Measured history required for the start pose and time')
        sample = samples[-1]
        if self.timeline is None:
            first = samples[0]
            self.timeline, self.start = Timeline(self.segment, first), first['time']
            heading = yaw_matrix(float(yaw_of(rotation_wxyz(first['root_quat']))))
            target = {'source': 'demo', 'segment': self.path.name, 'direction': self.segment['direction'],
                      'origin': first['root_pos'], 'heading': heading.tolist(), 'start_time': first['time'],
                      'segment_start_time': first['time']+self.timeline.segment_start/RATE,
                      'window_time': [first['time']+(i+1)/RATE for i in self.timeline.window],  # after-samples of those steps
                      'timeline_steps': self.timeline.length, 'arrival_time': first['time']+self.timeline.length/RATE,
                      'targets_world': [[first['wrist_pos'][0]], [first['wrist_pos'][1]]],
                      'wrist_rotation_world': first['wrist_rot']}
            self.target = target
            (self.run/'target.json').write_text(json.dumps(target, indent=2))
        index = int(round((sample['time']-self.start)*RATE))
        actions = self.timeline.chunk(index)
        with (self.run/'plans.jsonl').open('a') as f:
            f.write(json.dumps({'plan': self.plan, 'time': sample['time'], 'source': 'demo',
                                'timeline_index': index, 'commands': len(actions)})+'\n')
        with (self.run/'commands.jsonl').open('a') as f:
            f.write(''.join(json.dumps({'plan': self.plan, 'timeline_index': index+i, 'action': r.tolist()})+'\n'
                            for i, r in enumerate(actions)))
        self.plan += 1
        return {'action_chunk': actions.tolist(), 'oracle_target': self.target}


def score_run(steps, target, segment):
    """Score one replay run from its execution rows against the segment's window.

    Completion is the measured root displacement over the window, in the measured heading frame
    at the window's start, projected on the reference's own displacement over the window in its
    heading frame there, divided by that displacement's length. Turns compare yaw changes.
    Falls are checked from the start to 0.5 s after the window.
    """
    after = [s['after'] for s in steps]
    time = np.array([x['time'] for x in after])
    root = np.array([x['root_pos'] for x in after])
    rotations = np.array([rotation_wxyz(x['root_quat']) for x in after])
    t0, t1 = target['window_time']
    i0, i1 = int(np.argmin(abs(time-t0))), int(np.argmin(abs(time-t1)))
    if abs(time[i0]-t0) > .011 or abs(time[i1]-t1) > .011:
        return {'complete': False, 'steps': len(steps)}
    tilt = np.degrees(np.arccos(np.clip(rotations[:, 2, 2], -1, 1)))
    upto = time <= t1+.5
    fell = bool((root[upto, 2] < .40).any() or (tilt[upto] > 60).any())
    a = np.asarray(segment['actions'], dtype=float)
    b, e = segment['window']
    ref = window_summary(a, b, e)
    d_act, yaw_act = heading_frame_motion(root, rotations, i0, i1)
    out = {'complete': True, 'fell': fell, 'direction': segment['direction'], 'reference': ref,
           'actual_displacement_m': d_act.tolist(), 'actual_yaw_deg': yaw_act}
    if segment['direction'] == 'turn':
        out['completion'] = yaw_act/ref['yaw_deg']
        out['drift_cm'] = 100*float(np.hypot(*d_act))
        out['heading_drift_deg'] = yaw_act-ref['yaw_deg']
    else:
        along = np.asarray(ref['displacement_m'])/ref['distance_m']
        out['completion'] = float(d_act @ along)/ref['distance_m']
        # against what the reference's legs command along its direction (its feet may skate)
        legs = float(np.asarray(ref['leg_displacement_m']) @ along)
        out['leg_distance_m'] = legs
        out['completion_of_legs'] = float(d_act @ along)/legs if legs > .05 else None
        out['lateral_cm'] = 100*float(abs(along[0]*d_act[1]-along[1]*d_act[0]))
        out['heading_drift_deg'] = yaw_act-ref['yaw_deg']
    feet = np.array([x['foot_pos'] for x in after])[i0:i1+1]
    out['gait_actual'] = stepping(feet)
    return out
