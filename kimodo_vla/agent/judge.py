"""Which of the five links failed first, and how far each target was from the reachable boundary.

Reads one episode's logs only (``episode_log.load_episode``) and a thresholds dict (experiment 3's
``exp3_thresholds.json``; every number the rules use is a named entry there). The links follow
section 4 of the six-experiment plan:

  link  reach class (P&PBox)                                 leg class (SitSofa)
  1     stance / posture: a body collision, instability, or  body relative to the seat not in the
        the target outside the reachable region              sittable range, instability
  2     the commanded wrist pose is not in the object's      the commanded pelvis target is not
        graspable region, or the palm faces the wrong way    over the seat
  3     the measured wrist is off the commanded pose         the measured pelvis is off the
                                                             commanded target
  4     the wrists arrived, but closing did not lift it      the pelvis arrived, but no stable support
  5     lifted, then dropped (or the robot fell) before      seated, then lost the seat or fell
        success

A grasp attempt is a hand_pose command followed by closing the hands; a sit attempt is a
pelvis_height command low enough to sit (or with ``behind_feet_m``). For a failed episode the
judge walks the attempts in time and returns the first link whose check fails (an attempt that
lifted the box, or seated the hips, keeps its link 1-3 checks in the record as superseded: the
body evidently got there)
(``attribution='earliest'``, the plan's rule), or only the last attempt's
(``attribution='last_attempt'``, an option for the team). Every check it made is returned with
its value and threshold, so a reader can see why. A failed episode with no attempt, or one where
every check passed, is 'unclassified' with the reason; the judge does not force it into a link.

For P&PBox the placement on the shelf is judged with the same links (2: the box would not end
over the target layer, 3: the wrists did not arrive), marked ``phase='place'``.
"""
import numpy as np

from kimodo_vla.oracle.geometry import angle_deg, rotation_wxyz
from kimodo_vla.oracle.grasp_policy import stance

LINKS = {1: 'stance or posture', 2: 'key-part target wrong', 3: 'target right, body did not arrive',
         4: 'arrived, contact failed', 5: 'lost after contact'}


def value(thresholds, name):
    entry = thresholds[name]
    return entry['value'] if isinstance(entry, dict) else entry


def rectangle_margin(point, low, high):
    """Signed distance of a 2-d point to an axis-aligned rectangle: positive inside (to the nearest
    edge), negative outside (to the rectangle)."""
    p, low, high = (np.asarray(x, dtype=float) for x in (point, low, high))
    inside = np.minimum(p-low, high-p)
    if np.all(inside >= 0):
        return float(inside.min())
    return -float(np.linalg.norm(np.maximum(0., np.maximum(low-p, p-high))))


def box_reach_region(grasp_config, samples=41):
    """The first batch's validated box region (``stand.region``, both yaws pooled, stance rule moved
    back by ``stance_extra_m``) as the box centre's offset from the pelvis in the robot's frame:
    (forward range, left range)."""
    policy, stand, edge = grasp_config['policy'], grasp_config['stand'], grasp_config['scene']['table_edge_x_m']
    region, extra = stand['region'], stand['stance_extra_m']
    forward = []
    for depth in np.linspace(*region['depth_m'], samples):
        box = np.array([edge+depth, 0.])
        forward.append(box[0]-(stance(box, policy)[0]-extra))
    return [float(min(forward)), float(max(forward))], [float(x) for x in region['lateral_m']]


def to_frame(point, origin, yaw_deg):
    """World ground point in a frame at ``origin`` turned by ``yaw_deg``: (forward, left)."""
    yaw = np.radians(yaw_deg)
    d = np.asarray(point, dtype=float)[:2]-np.asarray(origin, dtype=float)[:2]
    return np.array([np.cos(yaw)*d[0]+np.sin(yaw)*d[1], -np.sin(yaw)*d[0]+np.cos(yaw)*d[1]])


_ARM_REACH = {}


def arm_reach_m(fk, samples=3000, seed=0):
    """Longest shoulder-to-wrist distance over random arm configurations within the joint limits."""
    if 'm' not in _ARM_REACH:
        names = list(fk.joint_names)
        arm = [names.index(n) for n in names if n.startswith('left_') and any(k in n for k in ('shoulder', 'elbow', 'wrist'))]
        rng = np.random.default_rng(seed)
        low, high = fk.joint_limits[arm, 0], fk.joint_limits[arm, 1]
        best = 0.
        for _ in range(samples):
            q = np.zeros(len(names))
            q[arm] = rng.uniform(low, high)
            positions, _ = fk(q, np.zeros(3), np.eye(3), body_names=['left_shoulder_pitch_link', 'left_wrist_yaw_link'])
            best = max(best, float(np.linalg.norm(positions[1]-positions[0])))
        _ARM_REACH['m'] = best
    return _ARM_REACH['m']


# ---- reading the logs -------------------------------------------------------------------
def steps_between(steps, start, end=None):
    return [s for s in steps if s['time'] >= start-1e-9 and (end is None or s['time'] <= end+1e-9)]


def object_track(steps, name):
    return [(s['time'], next((o for o in s.get('objects', []) if o['name'] == name), None)) for s in steps]


def scene_object(episode, name):
    scene = episode.get('scene') or {}
    return next((o for o in scene.get('objects', []) if o['name'] == name), None)


def sustained_contacts(steps, t, exclude_hands=True):
    """(time, body, force) of the first robot-body contact of at least ``collision_force_n`` that lasts
    ``collision_min_s`` (feet and ankles never count; hands and wrists only with ``exclude_hands`` off)."""
    since = {}
    for s in steps:
        pressing = {n: f for n, f in s.get('other_contacts_n', {}).items() if f >= value(t, 'collision_force_n')
                    and not (exclude_hands and ('_hand_' in n or 'wrist' in n))}
        since = {n: since.get(n, s['time']) for n in pressing}
        for n, first in since.items():
            if s['time']-first >= value(t, 'collision_min_s')-1e-9:
                return first, n, pressing[n]
    return None


def approach_problem(steps, t, until=None):
    """The first collision, fall or excessive tilt before ``until`` (the first attempt), or None."""
    window = steps_between(steps, 0., until)
    found = []
    hit = sustained_contacts(window, t, exclude_hands=False)
    if hit:
        found.append((hit[0], f'body collision before any attempt ({hit[1]}, {hit[2]:.0f} N)'))
    fell = fall_time(window)
    if fell is not None:
        found.append((fell, 'fell before any attempt'))
    tilted = next((s['time'] for s in window if s.get('tilt_deg', 0.) > value(t, 'stable_tilt_max_deg')), None)
    if tilted is not None:
        found.append((tilted, 'unstable before any attempt'))
    return min(found) if found else None


def success_of(episode):
    if episode.get('episode') is not None:
        return bool(episode['episode'].get('success'))
    return any(s.get('arena_success') for s in episode['steps'])


def fall_time(steps, start=0., end=None):
    return next((s['time'] for s in steps_between(steps, start, end) if s.get('fallen')), None)


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, link, name, passed, value=None, threshold=None, time=None, phase='grasp', **extra):
        self.rows.append({'link': link, 'check': name, 'passed': bool(passed), 'value': value, 'threshold': threshold,
                          'time': time, 'phase': phase, **extra})
        return passed

    def first_failure(self, rows=None):
        return next((r for r in (rows if rows is not None else self.rows) if not r['passed']), None)


# ---- reach class ------------------------------------------------------------------------
def grasp_attempts(commands):
    """(reach command, close command, index of the next opening) for every closing of the hands
    that follows a hand_pose command."""
    attempts, reach = [], None
    for i, c in enumerate(commands):
        if c['type'] == 'hand_pose':
            reach = c
        elif c['type'] == 'hand' and 'close' in (c['command'].get('left'), c['command'].get('right')) and reach is not None:
            opening = next((c2 for c2 in commands[i+1:] if c2['type'] == 'hand'
                            and 'open' in (c2['command'].get('left'), c2['command'].get('right'))), None)
            attempts.append({'reach': reach, 'close': c, 'open': opening, 'after': commands[i+1:]})
            reach = None
    return attempts


def wrist_in_box_frame(position, box_position, yaw_deg):
    f, l = to_frame(position, box_position, yaw_deg)
    return np.array([f, l, float(position[2]-box_position[2])])


def judge_box(episode, t, fk=None, grasp_config=None):
    checks, margins = Checks(), {}
    steps, commands = episode['steps'], episode['commands']
    box0 = scene_object(episode, 'box')
    rest_z = box0['position'][2] if box0 else None
    half = (box0 or {}).get('size', [value(t, 'box_size_m')]*3)[0]/2 if box0 else value(t, 'box_size_m')/2
    attempts = grasp_attempts(commands)
    early_fall = fall_time(steps, 0., attempts[0]['close']['time'] if attempts else None)
    results = []
    for k, a in enumerate(attempts):
        rows_before = len(checks.rows)
        reach, close = a['reach'], a['close']
        t0, t1 = reach['time'], (close.get('end') or {}).get('time', close['time'])
        pelvis, heading = reach['measured']['root'], reach['measured']['yaw_deg']
        box = next((o for _, o in object_track(steps_between(steps, t0), 'box')[:1] if o), box0)
        # link 1: collisions, stability, reach
        window = steps_between(steps, t0, t1)
        hit = sustained_contacts(window, t)
        checks.add(1, 'body collision during the reach', hit is None, value=hit, threshold=value(t, 'collision_force_n'),
                   time=hit[0] if hit else None, attempt=k)
        tilt = max((s.get('tilt_deg', 0.) for s in window), default=0.)
        fell = early_fall if k == 0 else fall_time(steps, t0, t1)
        checks.add(1, 'stable during the reach', fell is None and tilt <= value(t, 'stable_tilt_max_deg'),
                   value={'max_tilt_deg': tilt, 'fell_at': fell}, threshold=value(t, 'stable_tilt_max_deg'), time=t0, attempt=k)
        if box is not None:
            offset = to_frame(box['position'], pelvis, heading)
            if grasp_config is not None:
                forward, lateral = box_reach_region(grasp_config)
                low = np.array([forward[0], lateral[0]])-value(t, 'reach_region_pad_m')
                high = np.array([forward[1], lateral[1]])+value(t, 'reach_region_pad_m')
                margins['box_region_m'] = rectangle_margin(offset, low, high)
                checks.add(1, 'box inside the validated reach region', margins['box_region_m'] >= value(t, 'reach_margin_min_m'),
                           value=margins['box_region_m'], threshold=value(t, 'reach_margin_min_m'), time=t0, attempt=k,
                           box_offset_forward_left_m=offset.tolist())
        if fk is not None and 'shoulders' in reach['measured']:
            reach_max = arm_reach_m(fk)
            for side, target in reach['target'].get('wrists', {}).items():
                d = float(np.linalg.norm(np.subtract(target['position'], reach['measured']['shoulders'][side])))
                margins[f'{side}_arm_m'] = reach_max-d
                checks.add(1, f'{side} wrist target within arm reach', reach_max-d >= value(t, 'reach_margin_min_m'),
                           value=reach_max-d, threshold=value(t, 'reach_margin_min_m'), time=t0, attempt=k)
        # link 2: the commanded wrist poses against the graspable region of the box
        targets = reach['target'].get('wrists', {})
        checks.add(2, 'both wrists commanded', set(targets) == {'left', 'right'}, value=sorted(targets), time=t0, attempt=k)
        if box is not None:
            yaw = box.get('yaw_deg', 0.)
            # The face the hands go to is the one nearest the robot's side; box yaw relative to the robot.
            relative = np.radians(((yaw-heading)+45.) % 90.-45.)
            extent = half*(abs(np.cos(relative))+abs(np.sin(relative)))
            for side, target in targets.items():
                p = wrist_in_box_frame(target['position'], box['position'], heading)
                sign = 1. if side == 'left' else -1.
                gap = sign*p[1]-extent
                lo, hi = value(t, 'grasp_face_gap_m')
                checks.add(2, f'{side} wrist beside its face', lo <= gap <= hi, value=gap, threshold=[lo, hi], time=t0, attempt=k)
                lo, hi = value(t, 'grasp_forward_m')
                checks.add(2, f'{side} wrist along the face', lo <= p[0] <= hi, value=float(p[0]), threshold=[lo, hi], time=t0, attempt=k)
                lo, hi = value(t, 'grasp_height_m')
                checks.add(2, f'{side} wrist height', lo <= p[2] <= hi, value=float(p[2]), threshold=[lo, hi], time=t0, attempt=k)
                rotation = np.asarray(target['rotation'])
                palm = -rotation[:, 1] if side == 'left' else rotation[:, 1]
                inward = np.r_[-sign*np.array([-np.sin(np.radians(heading)), np.cos(np.radians(heading))]), 0.]
                cos = float(np.clip(palm @ inward/np.linalg.norm(palm), -1, 1))
                checks.add(2, f'{side} palm faces the box', np.degrees(np.arccos(cos)) <= value(t, 'palm_angle_max_deg'),
                           value=float(np.degrees(np.arccos(cos))), threshold=value(t, 'palm_angle_max_deg'), time=t0, attempt=k)
        # link 3: arrival of the commanded wrists
        end = reach.get('end') or {}
        error = end.get('measured_error', {})
        for side in targets:
            checks.add(3, f'{side} wrist arrived', error.get(f'{side}_wrist_m', np.inf) <= value(t, 'wrist_arrival_m'),
                       value=error.get(f'{side}_wrist_m'), threshold=value(t, 'wrist_arrival_m'), time=end.get('time'), attempt=k)
            checks.add(3, f'{side} wrist orientation arrived', error.get(f'{side}_wrist_deg', np.inf) <= value(t, 'wrist_arrival_deg'),
                       value=error.get(f'{side}_wrist_deg'), threshold=value(t, 'wrist_arrival_deg'), time=end.get('time'), attempt=k)
        # link 4: closing lifted the box
        stop = a['open']['time'] if a['open'] else None
        track = [(time, o) for time, o in object_track(steps_between(steps, close['time'], stop), 'box') if o]
        lift = [(time, o['position'][2]-rest_z) for time, o in track] if rest_z is not None else []
        lifted_at, run = None, 0.
        for (time, h), (time2, _) in zip(lift, lift[1:]+[(None, None)]):
            run = run+((time2 or time)-time) if h >= value(t, 'lifted_height_m') else 0.
            if run >= value(t, 'lifted_hold_s') and lifted_at is None:
                lifted_at = time
        checks.add(4, 'box lifted after closing', lifted_at is not None, value=max((h for _, h in lift), default=None),
                   threshold=value(t, 'lifted_height_m'), time=lifted_at, attempt=k)
        # link 5: lost after lifting
        if lifted_at is not None:
            # Lost = the box moved away from the hands (its offset from the wrists' midpoint changed) while
            # they were still closed, or the robot fell. The box going down with the hands is a placement.
            later = [s for s in steps_between(steps, lifted_at, stop) if s.get('wrist_pos') and (stop is None or s['time'] < stop)]
            offset = lambda s: np.subtract(next(o['position'] for o in s['objects'] if o['name'] == 'box'), np.mean(s['wrist_pos'], axis=0))
            dropped, drift = None, 0.
            if later:
                held = offset(later[0])
                for s in later:
                    d = float(np.linalg.norm(offset(s)-held))
                    drift = max(drift, d)
                    if d > value(t, 'dropped_drift_m') and not s.get('arena_success') and dropped is None:
                        dropped = s['time']
            fell = fall_time(steps, lifted_at, stop)
            checks.add(5, 'kept the box until the hands opened', dropped is None and fell is None,
                       value={'dropped_at': dropped, 'fell_at': fell, 'max_drift_m': drift}, threshold=value(t, 'dropped_drift_m'),
                       time=dropped or fell, attempt=k)
            if a['open'] is not None:
                place = next((c for c in reversed(commands) if c['type'] == 'hand_pose' and c['time'] < a['open']['time']), None)
                judge_place(checks, place, reach, box, episode, t, k)
        results.append(checks.rows[rows_before:])
    return checks, margins, results, early_fall, bool(attempts)


def judge_place(checks, place, grasp, box, episode, t, k):
    """Placement with the same links: where the commanded wrists would put the box, and whether they got there."""
    layer = scene_object(episode, 'shelf_target_layer')
    if place is None or layer is None or place is grasp:
        return
    wrists = place['target'].get('wrists', {})
    if set(wrists) != {'left', 'right'}:
        checks.add(2, 'place: both wrists commanded', False, value=sorted(wrists), time=place['time'], phase='place', attempt=k)
        return
    held = np.mean([place['measured']['wrists'][s]['position'] for s in wrists], axis=0)
    box_now = next((o['position'] for s in steps_between(episode['steps'], place['time'])[:1] for o in s.get('objects', [])
                    if o['name'] == 'box'), None)
    if box_now is None:
        return
    target_mid = np.mean([wrists[s]['position'] for s in wrists], axis=0)
    centre = np.asarray(box_now)+(target_mid-held)
    half = value(t, 'box_size_m')/2
    low, high = np.asarray(layer['min'][:2])+half, np.asarray(layer['max'][:2])-half
    margin = rectangle_margin(centre[:2], low, high)
    checks.add(2, 'place: box would end over the target layer', margin >= 0., value=margin, threshold=0.,
               time=place['time'], phase='place', attempt=k)
    bottom = centre[2]-half-layer['max'][2]
    lo, hi = value(t, 'place_bottom_gap_m')
    checks.add(2, 'place: box bottom at the layer', lo <= bottom <= hi, value=float(bottom), threshold=[lo, hi],
               time=place['time'], phase='place', attempt=k)
    error = (place.get('end') or {}).get('measured_error', {})
    for side in wrists:
        checks.add(3, f'place: {side} wrist arrived', error.get(f'{side}_wrist_m', np.inf) <= value(t, 'wrist_arrival_m'),
                   value=error.get(f'{side}_wrist_m'), threshold=value(t, 'wrist_arrival_m'), time=(place.get('end') or {}).get('time'),
                   phase='place', attempt=k)


# ---- leg class --------------------------------------------------------------------------
def sit_attempts(commands, t):
    return [c for c in commands if c['type'] == 'pelvis_height' and (c['command'].get('behind_feet_m') is not None
                                                                     or c['command']['height_m'] <= value(t, 'sit_height_max_m'))]


def judge_sofa(episode, t):
    checks, margins = Checks(), {}
    steps = episode['steps']
    seat = scene_object(episode, 'sofa_seat')
    attempts = sit_attempts(episode['commands'], t)
    early_fall = fall_time(steps, 0., attempts[0]['time'] if attempts else None)
    results = []
    for k, s in enumerate(attempts):
        rows_before = len(checks.rows)
        t0 = s['time']
        pelvis, heading = np.asarray(s['measured']['root']), s['measured']['yaw_deg']
        back = -np.array([np.cos(np.radians(heading)), np.sin(np.radians(heading))])
        if seat is None:
            break
        low, high = np.asarray(seat['min']), np.asarray(seat['max'])
        centre = (low+high)/2
        # link 1: the robot's back to the seat, close enough, standing
        to_seat = centre[:2]-pelvis[:2]
        facing = float(np.degrees(np.arccos(np.clip(back @ to_seat/max(np.linalg.norm(to_seat), 1e-9), -1, 1))))
        checks.add(1, 'back turned to the seat', facing <= value(t, 'sit_facing_max_deg'), value=facing,
                   threshold=value(t, 'sit_facing_max_deg'), time=t0, phase='sit', attempt=k)
        # Distance along the back direction from the pelvis to where it would enter the seat box.
        ray = [d for d in np.linspace(0., 1.5, 301) if np.all((pelvis[:2]+d*back >= low[:2]) & (pelvis[:2]+d*back <= high[:2]))]
        entry = ray[0] if ray else None
        lo, hi = value(t, 'sit_entry_distance_m')
        checks.add(1, 'seat edge within reach behind the pelvis', entry is not None and lo <= entry <= hi, value=entry,
                   threshold=[lo, hi], time=t0, phase='sit', attempt=k)
        fell = early_fall if k == 0 else fall_time(steps, steps_between(steps, 0., t0)[-1]['time'] if steps else 0., t0)
        checks.add(1, 'stable before sitting', fell is None, value=fell, time=t0, phase='sit', attempt=k)
        # link 2: the commanded pelvis target over the seat
        behind = s['command'].get('behind_feet_m') or 0.
        target = np.r_[pelvis[:2]+behind*back, s['command']['height_m']]
        margins['pelvis_target_seat_m'] = rectangle_margin(target[:2], low[:2], high[:2])
        checks.add(2, 'pelvis target over the seat', margins['pelvis_target_seat_m'] >= value(t, 'seat_margin_min_m'),
                   value=margins['pelvis_target_seat_m'], threshold=value(t, 'seat_margin_min_m'), time=t0, phase='sit', attempt=k)
        lo, hi = value(t, 'sit_height_above_seat_m')
        checks.add(2, 'pelvis target height above the seat top', lo <= target[2]-high[2] <= hi, value=float(target[2]-high[2]),
                   threshold=[lo, hi], time=t0, phase='sit', attempt=k)
        # link 3: arrival
        end = s.get('end') or {}
        measured = end.get('measured') or {}
        if measured:
            xy = float(np.linalg.norm(np.asarray(measured['root'][:2])-target[:2]))
            dz = abs(measured['root'][2]-target[2])
            checks.add(3, 'pelvis arrived (ground)', xy <= value(t, 'pelvis_arrival_m'), value=xy, threshold=value(t, 'pelvis_arrival_m'),
                       time=end['time'], phase='sit', attempt=k)
            checks.add(3, 'pelvis arrived (height)', dz <= value(t, 'pelvis_arrival_m'), value=dz, threshold=value(t, 'pelvis_arrival_m'),
                       time=end['time'], phase='sit', attempt=k)
        # link 4: support formed
        after = steps_between(steps, end.get('time', t0), end.get('time', t0)+value(t, 'support_window_s'))
        held = [x.get('hip_contact_n', 0.) for x in after]
        supported = bool(held) and np.median(held) >= value(t, 'support_force_n')
        checks.add(4, 'seat carries the hips', supported, value=float(np.median(held)) if held else None,
                   threshold=value(t, 'support_force_n'), time=end.get('time'), phase='sit', attempt=k)
        # link 5: kept the seat
        if supported:
            later = steps_between(steps, end['time'])
            fell = fall_time(steps, end['time'])
            lost = next((x['time'] for x in later if x.get('hip_contact_n', 0.) < value(t, 'support_force_n')/2
                         and x['time'] > end['time']+value(t, 'support_window_s')), None)
            checks.add(5, 'stayed seated', fell is None and (lost is None or success_of(episode)), value={'fell_at': fell, 'lost_at': lost},
                       time=fell or lost, phase='sit', attempt=k)
        results.append(checks.rows[rows_before:])
    return checks, margins, results, early_fall, bool(attempts)


def judge(episode, thresholds, fk=None, grasp_config=None, attribution='earliest'):
    """Verdict for one episode: success, the failed link (1-5) or 'unclassified', and every check."""
    if attribution not in ('earliest', 'last_attempt'):
        raise ValueError(attribution)
    task = (episode.get('meta') or {}).get('task') or (episode.get('scene') or {}).get('task')
    if task == 'pp_box':
        checks, margins, attempts, early_fall, tried = judge_box(episode, thresholds, fk, grasp_config)
    elif task == 'sit_sofa':
        checks, margins, attempts, early_fall, tried = judge_sofa(episode, thresholds)
    else:
        raise ValueError(f'Unknown task {task!r}')
    success = success_of(episode)
    verdict = {'run': episode.get('run'), 'task': task, 'success': success, 'link': None, 'link_name': None,
               'phase': None, 'reason': None, 'attempts': len(attempts), 'checks': checks.rows, 'margins': margins,
               'end_reason': (episode.get('summary') or {}).get('end_reason'),
               'arena_failure_reason': (episode.get('episode') or {}).get('failure_reason'), 'attribution': attribution}
    if success:
        return verdict
    failed = None
    if attempts:
        pool = attempts if attribution == 'earliest' else attempts[-1:]
        for rows in pool:
            # An attempt that lifted the box (or seated the hips) shows its body got there: its link 1-3 checks
            # stay in the record but do not count as the failure (marked superseded).
            if any(r['link'] == 4 and r['passed'] for r in rows):
                for r in rows:
                    if r['link'] in (1, 2, 3) and not r['passed'] and r.get('phase') != 'place':
                        r['superseded'] = True
            failed = checks.first_failure([r for r in rows if not r.get('superseded')])
            if failed:
                break
    if failed is None and not tried:
        # No attempt at all: a collision or instability on the way is a stance / posture failure (link 1);
        # a clean approach that never attempted stays unclassified.
        problem = approach_problem(episode['steps'], thresholds)
        if problem is not None:
            failed = {'link': 1, 'check': problem[1], 'phase': 'approach', 'time': problem[0]}
    if failed is None:
        verdict.update(link='unclassified', reason=('no attempt' if not tried else 'every check passed')
                       + (f" (episode ended: {verdict['arena_failure_reason']})" if verdict['arena_failure_reason'] else ''))
        return verdict
    verdict.update(link=failed['link'], link_name=LINKS[failed['link']], phase=failed.get('phase'),
                   reason=failed['check'], failed_check=failed)
    return verdict
