"""Experiment 1 follow-up, step 3: did the table push the robot back? Hands against the table, from logs.

For each first attempt of the static box groups, every control step from the anchor to P: the
fingers, thumb and palm of both hands (Dex3 finger chain of HITCH's ``dex3_1.xml`` in the
wrist-yaw frame, driven by the logged finger angles) against the table's collision box
(``grasp-scene.json``, the PackingTable collider). Clearance is the signed distance of the
closest hand point to that box: negative means the point is inside the table, i.e. the hand
would have to pass through it.

Fixed before looking: back-off onset is the first step the pelvis is >= ONSET_M behind the
anchor along the heading (successes stay below ~4 cm); "at the table" means a hand point within
TOUCH_M of the table box. Read: hands at the table before or at back-off onset and inside it
around onset -> consistent with the table pushing back; hands clearly above or short of the
table at onset -> against it.

The same is computed for the open-floor replays (``replay_references.py``) against the table
carried into the replay frame (no table was there): how far the hands went where the table
would have been.

Usage (dev box):
  python table_clearance.py logs --reach-out DIR --stage-out DIR --out DIR
  python table_clearance.py replays --replay-out DIR --out DIR [--date YYYYMMDD] [--tag-prefix p2e1replay]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage_split import key, quat_matrix, read_jsonl, round_targets

ONSET_M = .05
TOUCH_M = .01
EDGE_M = .03    # a first touch whose closest point is less than this past the front edge counts as "at the edge"
# Dex3-1 on the G1 wrist-yaw link (HITCH assets/g1/hands/dex3_1.xml); sign s = +1 left, -1 right.
FINGER_BASE = {'index': (.1192, .0046, .0285), 'middle': (.1192, .0046, -.0285)}
PHALANX_M, TIP_M = .0458, .03          # proximal link length; distal tip length (mesh, estimated)


def rot(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def hand_points(side, angles):
    """Palm, finger and thumb points [k, 3] in the wrist-yaw frame for one hand's finger angles (dict, radians)."""
    s = 1. if side == 'left' else -1.
    get = lambda name: float(angles.get(f'{side}_hand_{name}_joint', 0.))
    points = [np.array([.0415, s * .003, 0.])]
    for finger, (x, y, z) in FINGER_BASE.items():
        base = np.array([x, s * y, z])
        r0 = rot('z', get(f'{finger}_0'))
        mid = base + r0 @ [PHALANX_M, 0, 0]
        r1 = r0 @ rot('z', get(f'{finger}_1'))
        points += [base, mid, mid + r1 @ [TIP_M, 0, 0]]
    t0 = np.array([.067, s * .003, 0.])
    r = rot('y', get('thumb_0')) @ rot('z', get('thumb_1'))
    t1 = t0 + r @ [-.0025, -s * .0193, 0]
    r = r @ rot('z', get('thumb_2'))
    t2 = t1 + r @ [0, -s * .0458, 0]
    points += [t1, t2, t2 + r @ [0, -s * TIP_M, 0]]
    return np.array(points)


def world_points(after):
    out = []
    for h, side in enumerate(('left', 'right')):
        local = hand_points(side, after.get('hand_joint_pos') or {})
        out.append(np.asarray(after['wrist_pos'][h]) + local @ np.asarray(after['wrist_rot'][h]).T)
    return np.concatenate(out)


def box_distance(points, lo, hi):
    """Signed distance of each point to an axis-aligned box: negative inside."""
    p = np.asarray(points, float)
    outside = np.maximum(np.maximum(lo - p, p - hi), 0.)
    out = np.linalg.norm(outside, axis=-1)
    inside = np.all((p > lo) & (p < hi), axis=-1)
    depth = np.min(np.minimum(p - lo, hi - p), axis=-1)
    return np.where(inside, -depth, out)


def table_box(run):
    scene = json.loads((run / 'grasp-scene.json').read_text())
    table = next(c for c in scene['colliders_near_box'] if 'PackingTable' in c['path'])
    return np.asarray(table['min'], float), np.asarray(table['max'], float)


def trace(rows, start, end, lo, hi, origin=None, heading=None, to_world=None):
    """Per step: time, clearance (signed, closest hand point), back-off along the heading, and whether
    the closest point is in front of the edge / above the top."""
    out = []
    for r in rows:
        t = r['after']['time']
        if t < start - 1e-6 or t > end + 1e-6:
            continue
        pts = world_points(r['after'])
        if to_world is not None:
            pts = to_world(pts)
        d = box_distance(pts, lo, hi)
        i = int(np.argmin(d))
        back = -float((np.asarray(r['after']['root_pos'][:2]) - origin) @ heading)
        out.append({'t': t, 'clearance': float(d[i]), 'backoff': back, 'z_above_top': float(pts[i, 2] - hi[2]),
                    'x_beyond_edge': float(pts[i, 0] - lo[0])})
    return out


def advance_contact(tr, from_t):
    """Closest approach once the hands should be raised (from ``from_t``), and where: added after the first
    table showed that the hanging hands at the anchor already sit next to the table's front face."""
    later = [x for x in tr if x['t'] >= from_t - 1e-6]
    if not later:
        return {}
    w = min(later, key=lambda x: x['clearance'])
    return {'advance_min_clearance_m': w['clearance'], 'advance_z_above_top_m': w['z_above_top'],
            'advance_x_beyond_edge_m': w['x_beyond_edge'],
            'advance_edge_contact': bool(w['clearance'] <= TOUCH_M and w['x_beyond_edge'] < EDGE_M),
            'advance_contact_s': w['t']}


def summarize_trace(tr, start):
    onset = next((x for x in tr if x['backoff'] >= ONSET_M), None)
    touch = next((x for x in tr if x['clearance'] <= TOUCH_M), None)
    worst = min(tr, key=lambda x: x['clearance']) if tr else None
    near = [x for x in tr if onset and abs(x['t'] - onset['t']) <= .1]
    out = {'min_clearance_m': worst['clearance'] if worst else None,
           'touch_s': None if touch is None else touch['t'] - start, 'onset_s': None if onset is None else onset['t'] - start,
           'max_backoff_m': max((x['backoff'] for x in tr), default=None)}
    if touch is not None:   # where the hand first met the table: at the front edge, or over the top
        out.update(z_above_top_at_touch_m=touch['z_above_top'], x_beyond_edge_at_touch_m=touch['x_beyond_edge'],
                   touch_at_edge=bool(touch['x_beyond_edge'] < EDGE_M))
    if onset is not None:
        out.update(clearance_at_onset_m=min(x['clearance'] for x in near),
                   z_above_top_at_onset_m=min(near, key=lambda x: x['clearance'])['z_above_top'],
                   x_beyond_edge_at_onset_m=min(near, key=lambda x: x['clearance'])['x_beyond_edge'])
        out['order'] = ('no_touch' if touch is None else 'touch_first' if touch['t'] <= onset['t'] else 'onset_first')
    else:
        out['order'] = 'no_onset'
    return out


def episode(runs, record, reach):
    run = runs / record['run']
    rounds = read_jsonl(run / 'rounds.jsonl')
    anchor = round_targets(rounds)[0][0]
    rows = read_jsonl(run / 'execution.jsonl')
    a = next(r for r in rows if key(r['after']['time']) == key(anchor))['after']
    R = quat_matrix(a['root_quat'])
    heading = np.array([R[0, 0], R[1, 0]]) / np.linalg.norm([R[0, 0], R[1, 0]])
    lo, hi = table_box(run)
    tr = trace(rows, anchor, record['time_P'], lo, hi, np.asarray(a['root_pos'][:2]), heading)
    raise_end = round_targets(rounds)[0][2]['raise_end']
    out = {'run': record['run'], 'cell': record['cell'], 'group': reach['group'], 'first_success': record['first_success'],
           'mode': record['mode'], **summarize_trace(tr, anchor), **advance_contact(tr, raise_end)}
    if out.get('advance_contact_s') is not None:
        out['advance_contact_s'] -= anchor
    return out


def replay(runs, replay_dir, m, tracker, date, prefix):
    """The replay's hands against the original table, carried into the replay frame by the anchors' reference roots."""
    original = runs / m['run']
    rounds = read_jsonl(original / 'rounds.jsonl')
    anchor = round_targets(rounds)[0][0]
    orig_rows = {key(r['after']['time']): r for r in read_jsonl(original / 'execution.jsonl')}
    a = orig_rows[key(anchor)]
    run = runs / f"{prefix}-{tracker}-{m['tag']}-{date}"
    target = json.loads((run / 'target.json').read_text())
    rows = [r for r in read_jsonl(run / 'execution.jsonl') if 'reference_root_world' in r]
    t0, t1 = target['window_time']
    b = min(rows, key=lambda r: abs(r['after']['time'] - t0))
    yaw = lambda q: np.arctan2(quat_matrix(q)[1, 0], quat_matrix(q)[0, 0])
    d = yaw(a['reference_root_quat']) - yaw(b['reference_root_quat'])
    turn = np.array([[np.cos(d), -np.sin(d), 0], [np.sin(d), np.cos(d), 0], [0, 0, 1]])
    pa, pb = np.asarray(a['reference_root_world']), np.asarray(b['reference_root_world'])
    to_world = lambda pts: (turn @ (pts - pb).T).T + pa      # replay frame -> original frame
    lo, hi = table_box(original)
    Rb = quat_matrix(b['after']['root_quat'])
    heading = np.array([Rb[0, 0], Rb[1, 0]]) / np.linalg.norm([Rb[0, 0], Rb[1, 0]])
    tr = trace(rows, t0, t1, lo, hi, np.asarray(b['after']['root_pos'][:2]), heading, to_world)
    raise_end = round_targets(rounds)[0][2]['raise_end'] - anchor + t0
    out = {'tag': m['tag'], 'tracker': tracker, 'run': run.name, 'outcome': m['outcome'], **summarize_trace(tr, t0),
           **advance_contact(tr, raise_end)}
    if out.get('advance_contact_s') is not None:
        out['advance_contact_s'] -= t0
    return out


def table(records, key_fields=('group', 'outcome')):
    med = lambda rows, name, scale=100.: ('—' if not [r for r in rows if r.get(name) is not None] else
                                          f"{np.median([r[name] for r in rows if r.get(name) is not None]) * scale:.1f}")
    by = defaultdict(list)
    for r in records:
        outcome = 'success' if r['first_success'] else ('failure, never closed' if r['mode'] == 'never_closed' else 'failure, other')
        by[(r['group'], outcome)].append(r)
        by[(r['group'] + ' · ' + r['cell'], outcome)].append(r)
    lines = ['| group | outcome | n | min hand–table clearance (cm) | hands inside the table box (n) | back-off >= 5 cm (n) | '
             'clearance at onset (cm) | closest point above the top / beyond the edge at onset (cm) | at the table before onset / after / never | '
             'first touch above the top / beyond the edge (cm) | first touch at the front edge (n) | '
             'after the raise: closest clearance (cm) | its point above the top / beyond the edge (cm) | at the front edge, touching (n) |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for (group, outcome), rows in sorted(by.items()):
        orders = {o: sum(r['order'] == o for r in rows) for o in ('touch_first', 'onset_first', 'no_touch')}
        lines.append(f"| {group} | {outcome} | {len(rows)} | {med(rows, 'min_clearance_m')} | "
                     f"{sum(1 for r in rows if (r['min_clearance_m'] or 1) < 0)} | {sum(r['onset_s'] is not None for r in rows)} | "
                     f"{med(rows, 'clearance_at_onset_m')} | {med(rows, 'z_above_top_at_onset_m')} / {med(rows, 'x_beyond_edge_at_onset_m')} | "
                     f"{orders['touch_first']} / {orders['onset_first']} / {orders['no_touch']} | "
                     f"{med(rows, 'z_above_top_at_touch_m')} / {med(rows, 'x_beyond_edge_at_touch_m')} | "
                     f"{sum(1 for r in rows if r.get('touch_at_edge'))} | {med(rows, 'advance_min_clearance_m')} | "
                     f"{med(rows, 'advance_z_above_top_m')} / {med(rows, 'advance_x_beyond_edge_m')} | "
                     f"{sum(1 for r in rows if r.get('advance_edge_contact'))} |")
    return '\n'.join(lines)


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('action', choices=('logs', 'replays'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reach-out', type=Path)
    p.add_argument('--stage-out', type=Path)
    p.add_argument('--replay-out', type=Path)
    p.add_argument('--date', default='20261127')
    p.add_argument('--prefix', default='p2e1replay')
    p.add_argument('--workers', type=int, default=32)
    a = p.parse_args(argv)
    from paths import RUNS
    a.out.mkdir(parents=True, exist_ok=True)
    if a.action == 'logs':
        reach = {r['run']: r for r in read_jsonl(a.reach_out / 'reach.jsonl') if 'skip' not in r and r['group'].startswith('box')}
        records = [r for r in read_jsonl(a.stage_out / 'episodes.jsonl') if r.get('run') in reach]
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            out = pool.starmap(episode, [(RUNS, r, reach[r['run']]) for r in records])
        with (a.out / 'clearance.jsonl').open('w') as f:
            f.write(''.join(json.dumps(r) + '\n' for r in out))
        (a.out / 'clearance.md').write_text(table(out) + '\n')
        print(table(out))
    else:
        rows = []
        for m in json.loads((a.replay_out / 'manifest.json').read_text()):
            for tracker in ('sonic', 'heft'):
                run = RUNS / f"{a.prefix}-{tracker}-{m['tag']}-{a.date}"
                if (run / 'target.json').exists():
                    rows.append(replay(RUNS, a.replay_out, m, tracker, a.date, a.prefix))
        (a.out / f'clearance-{a.prefix}.json').write_text(json.dumps(rows, indent=1))
        for r in rows:
            print(r['tag'], r['tracker'], r['outcome'], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                                                       if k in ('min_clearance_m', 'onset_s', 'max_backoff_m', 'advance_min_clearance_m',
                                                                'advance_z_above_top_m', 'advance_x_beyond_edge_m', 'advance_edge_contact')})


if __name__ == '__main__':
    main()
