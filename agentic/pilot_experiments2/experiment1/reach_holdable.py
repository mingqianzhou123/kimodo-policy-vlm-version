"""Experiment 1 follow-up: why the hands never arrived in the static grasp failures.

For every first attempt in the static box (standing, walking) and static hammer groups, three
questions, answered from the logs with the XML model (no simulation):

  reach      Could the arm get to the task target from the stance the robot actually had? The
             root stays where it stood, the legs as measured at the anchor, the waist anywhere
             in its joint limits (that is all the IK may move with the root and both ankles
             pinned); the target is in reach when the shoulder can come within one arm length.
  holdable   Is the reference statically holdable? Its centre of mass (model masses, reference
             joints and root) projected on the ground against the convex hull of both feet's
             contact capsules. A pose whose centre of mass lies outside cannot be held standing
             by any tracker.
  followed   What SONIC did: when the pelvis began to back off (>= BACKOFF_M along the heading,
             measured from the anchor) against when the reference's centre of mass first left
             the support.

Each episode gets one class, in this order: ``out_of_reach``, ``not_holdable`` (at the moment
the planned path reaches the task target, before any closed-loop correction), else
``holdable``. A failure in the last class is "the reference could be held, SONIC did not follow".

Usage (dev box, policy Python):
  python reach_holdable.py run --stage-out DIR --out DIR [--workers N]
  python reach_holdable.py table --out DIR
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage_split import key, quat_matrix, read_jsonl, round_targets, target_at

GROUPS = (('box', 'static-stand'), ('box', 'static-walk'), ('hammer', 'static-stand'))
BACKOFF_M = .02                       # pelvis movement against the heading that counts as backing off
EXTRA_HAND_KG = .5                    # sensitivity: the simulated Dex3 hands outweigh the model's rubber hands
WAIST = ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint')
WAIST_GRID = (np.linspace(-.8, .8, 9), np.linspace(-.52, .52, 5), np.linspace(-.52, .52, 9))
FOOT_POINTS = np.array([[x, y, -.025] for x in (-.054, .132) for y in (-.027, .027)])   # capsule ends, g1.xml
SIDES = ('left', 'right')


# ---- geometry ---------------------------------------------------------------------------------------
def hull(points):
    """Convex hull of 2-D points, counter-clockwise (monotone chain)."""
    pts = sorted(map(tuple, np.asarray(points, float)))
    if len(pts) < 3:
        return np.asarray(pts)
    cross = lambda o, a, b: (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def support_margin(point, polygon):
    """Signed distance from a 2-D point to a convex counter-clockwise polygon: positive inside."""
    p, poly = np.asarray(point, float), np.asarray(polygon, float)
    distances, inside = [], True
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        edge = b - a
        if (edge[0]*(p - a)[1] - edge[1]*(p - a)[0]) < 0:
            inside = False
        t = np.clip(np.dot(p - a, edge) / np.dot(edge, edge), 0., 1.)
        distances.append(np.linalg.norm(p - (a + t*edge)))
    return float(min(distances)) * (1. if inside else -1.)


class Body:
    """Whole-body quantities of one pose: centre of mass, foot support, shoulders."""
    def __init__(self, fk, xml, extra_hand_kg=0.):
        self.fk = fk
        tree = ET.parse(xml).getroot()
        inertial = {b.get('name'): b.find('inertial') for b in tree.iter('body')}
        self.names = [n for n in fk.body_names if inertial.get(n) is not None]
        self.mass = np.array([float(inertial[n].get('mass')) for n in self.names])
        self.local_com = np.array([np.fromstring(inertial[n].get('pos'), sep=' ') for n in self.names])
        for side in SIDES:   # a heavier hand: point mass at the palm site, 8 cm out along the wrist's x
            self.names.append(f'{side}_wrist_yaw_link')
            self.mass = np.r_[self.mass, extra_hand_kg]
            self.local_com = np.vstack([self.local_com, [.08, 0, 0]])
        self.joint_index = {n: i for i, n in enumerate(fk.joint_names)}

    def com(self, q, root, rotation):
        pos, rot = self.fk(q, root, rotation, body_names=self.names)
        points = pos + np.einsum('bij,bj->bi', rot, self.local_com)
        return (self.mass[:, None] * points).sum(0) / self.mass.sum()

    def support(self, q, root, rotation):
        pos, rot = self.fk(q, root, rotation, body_names=[f'{s}_ankle_roll_link' for s in SIDES])
        corners = np.concatenate([p + FOOT_POINTS @ r.T for p, r in zip(pos, rot)])
        return hull(corners[:, :2])

    def com_margin(self, q, root, rotation):
        return support_margin(self.com(q, root, rotation)[:2], self.support(q, root, rotation))

    def shoulders(self, q, root, rotation):
        return self.fk(q, root, rotation, body_names=[f'{s}_shoulder_pitch_link' for s in SIDES])[0]


def arm_length(fk, side, elbows=121, others=30, seed=0):
    """Longest shoulder-pitch-origin to wrist-yaw-origin distance over the arm's joint limits
    (an elbow sweep, with random shoulder and wrist angles at each elbow angle)."""
    rng = np.random.default_rng(seed)
    arm = [i for i, n in enumerate(fk.joint_names) if n.startswith(f'{side}_') and
           any(k in n for k in ('shoulder', 'elbow', 'wrist'))]
    elbow = next(i for i in arm if 'elbow' in fk.joint_names[i])
    names = [f'{side}_shoulder_pitch_link', f'{side}_wrist_yaw_link']
    best = 0.
    for angle in np.linspace(*fk.joint_limits[elbow], elbows):
        for _ in range(others):
            q = np.zeros(len(fk.joint_names))
            q[arm] = rng.uniform(fk.joint_limits[arm, 0], fk.joint_limits[arm, 1])
            q[elbow] = angle
            pos, _ = fk(q, np.zeros(3), np.eye(3), body_names=names)
            best = max(best, float(np.linalg.norm(pos[1] - pos[0])))
    return best


def reach_margin(body, lengths, q, root, rotation, targets, hands):
    """Arm length minus the closest shoulder-to-target distance over the waist's range; per hand, worst hand."""
    q = np.asarray(q, float).copy()
    ids = [body.joint_index[n] for n in WAIST]
    best = {h: np.inf for h in hands}
    for yaw in WAIST_GRID[0]:
        for roll in WAIST_GRID[1]:
            for pitch in WAIST_GRID[2]:
                q[ids] = (yaw, roll, pitch)
                shoulders = body.shoulders(q, root, rotation)
                for h in hands:
                    best[h] = min(best[h], float(np.linalg.norm(shoulders[h] - targets[h])))
    return min(lengths[h] - best[h] for h in hands)


# ---- one episode ------------------------------------------------------------------------------------
def classify(reach, holdable_margin):
    if reach < 0:
        return 'out_of_reach'
    return 'not_holdable' if holdable_margin < 0 else 'holdable'


def episode(run, record, body, heavy, lengths):
    run = Path(run)
    rounds = read_jsonl(run / 'rounds.jsonl')
    steps = read_jsonl(run / 'grasp-steps.jsonl')
    targets = round_targets(rounds)
    execution = {key(r['after']['time']): r for r in read_jsonl(run / 'execution.jsonl')}
    anchor_time = targets[0][0]
    anchor = execution.get(key(anchor_time))
    kind, hands = record['kind'], ((0, 1) if record['kind'] == 'box' else (1,))
    P_time = record['time_P']
    wrists, notes = target_at(targets, P_time)
    task = np.asarray(wrists['pre'] if kind == 'box' else [wrists['grip']] * 2, float)
    a = anchor['after']
    a_rot = quat_matrix(a['root_quat'])
    out = {'run': run.name, 'cell': record['cell'], 'group': f"{record['family']} · {record['condition']}",
           'first_success': record['first_success'], 'mode': record['mode'],
           'reach_margin_m': reach_margin(body, lengths, a['joints'], a['root_pos'], a_rot, task, hands)}
    upright = body.shoulders(np.asarray(a['joints'], float), np.asarray(a['root_pos'], float), a_rot)
    out['upright_reach_margin_m'] = min(lengths[h] - float(np.linalg.norm(upright[h] - task[h])) for h in hands)

    def reference(t):
        row = execution.get(key(t))
        if row is None or 'reference_joints' not in row:
            return None
        return np.asarray(row['reference_joints'], float), np.asarray(row['reference_root_world'], float), \
            quat_matrix(row['reference_root_quat'])

    # The reference when the planned path reaches the task target (no correction yet), else at P.
    arrive = min(float(notes['advance_end']), P_time)
    ref = reference(arrive) or reference(P_time)
    out['ref_com_margin_arrive_m'] = body.com_margin(*ref) if ref else float('nan')
    out['ref_com_margin_arrive_heavy_m'] = heavy.com_margin(*ref) if ref else float('nan')
    ref_P = reference(P_time)
    out['ref_com_margin_P_m'] = body.com_margin(*ref_P) if ref_P else float('nan')

    # Over the reach (anchor .. P): when the reference first left the support, when the pelvis backed off.
    yaw = np.arctan2(a_rot[1, 0], a_rot[0, 0])
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    start = np.asarray(a['root_pos'][:2], float)
    com_out = backoff_at = None
    worst_ref, worst_actual, max_backoff = np.inf, np.inf, 0.
    for s in steps:
        t = s['time']
        if s.get('round') != 0 or t < anchor_time - 1e-6 or t > P_time + 1e-6 or key(t) % 2:   # every other step
            continue
        row = execution.get(key(t))
        if row is None:
            continue
        r = reference(t)
        if r is not None:
            m = body.com_margin(*r)
            worst_ref = min(worst_ref, m)
            if m < 0 and com_out is None:
                com_out = t
        after = row['after']
        worst_actual = min(worst_actual, body.com_margin(np.asarray(after['joints'], float),
                                                         np.asarray(after['root_pos'], float), quat_matrix(after['root_quat'])))
        back = -float((np.asarray(after['root_pos'][:2]) - start) @ forward)
        max_backoff = max(max_backoff, back)
        if back >= BACKOFF_M and backoff_at is None:
            backoff_at = t
    out.update(ref_com_margin_worst_m=float(worst_ref), actual_com_margin_worst_m=float(worst_actual),
               max_backoff_m=max_backoff, backoff_s=None if backoff_at is None else backoff_at - anchor_time,
               ref_com_out_s=None if com_out is None else com_out - anchor_time)
    if backoff_at is None:
        out['order'] = 'no_backoff'
    elif com_out is None:
        out['order'] = 'backoff_reference_inside'
    else:
        out['order'] = 'backoff_first' if backoff_at < com_out else 'reference_left_first'
    out['class'] = classify(out['reach_margin_m'], out['ref_com_margin_arrive_m'])
    out['class_heavy_hands'] = classify(out['reach_margin_m'], out['ref_com_margin_arrive_heavy_m'])
    for name in ('P.torso_rot.1', 'P.torso_rot.total', 'P.root_pos.2', 'P.wrist_pos.correction'):
        out[name] = record.get(name)
    return out


# ---- the summary --------------------------------------------------------------------------------------
def counts(rows, field='class'):
    return {c: sum(r[field] == c for r in rows) for c in ('out_of_reach', 'not_holdable', 'holdable')}


def table(records):
    med = lambda rows, name, scale=1.: ('—' if not rows else
                                        f"{np.nanmedian([np.nan if r.get(name) is None else r[name] for r in rows]) * scale:.1f}")
    lines = ['| group · cell | outcome | n | out of reach / not holdable / holdable | same, hands +0.5 kg | '
             'reach margin (cm) | ref. CoM margin at arrival (cm) | worst ref. CoM margin (cm) | max pelvis back-off (cm) | '
             'trunk lean asked / done (deg) | back-off before ref. CoM leaves / after / ref. never leaves / no back-off |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    by = defaultdict(list)
    for r in records:
        if 'skip' in r:
            continue
        outcome = 'success' if r['first_success'] else ('failure, never closed' if r['mode'] == 'never_closed' else 'failure, other')
        by[(r['group'], r['cell'], outcome)].append(r)
        by[(r['group'], '(all cells)', outcome)].append(r)
    for (group, cell, outcome), rows in sorted(by.items()):
        c, h = counts(rows), counts(rows, 'class_heavy_hands')
        order = {o: sum(r['order'] == o for r in rows) for o in
                 ('backoff_first', 'reference_left_first', 'backoff_reference_inside', 'no_backoff')}
        lines.append(f"| {group} · {cell} | {outcome} | {len(rows)} | {c['out_of_reach']} / {c['not_holdable']} / {c['holdable']} | "
                     f"{h['out_of_reach']} / {h['not_holdable']} / {h['holdable']} | {med(rows, 'reach_margin_m', 100)} | "
                     f"{med(rows, 'ref_com_margin_arrive_m', 100)} | {med(rows, 'ref_com_margin_worst_m', 100)} | "
                     f"{med(rows, 'max_backoff_m', 100)} | {med(rows, 'P.torso_rot.1')} / {med(rows, 'P.torso_rot.total')} | "
                     f"{order['backoff_first']} / {order['reference_left_first']} / {order['backoff_reference_inside']} / {order['no_backoff']} |")
    return '\n'.join(lines)


_STATE = None


def _work(args):
    global _STATE
    run, record = args
    if _STATE is None:
        from paths import G1_XML
        from kimodo_vla.core.types import G1_JOINT_NAMES
        from kimodo_vla.oracle.geometry import XmlFK
        fk = XmlFK(G1_XML, G1_JOINT_NAMES)
        _STATE = (Body(fk, G1_XML), Body(fk, G1_XML, EXTRA_HAND_KG), [arm_length(fk, s) for s in SIDES])
    try:
        return episode(run, record, *_STATE)
    except Exception as exc:
        return {'run': Path(run).name, 'skip': f'{type(exc).__name__}: {exc}'}


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('command', choices=('run', 'table'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--stage-out', type=Path, help='output folder of stage_split.py (episodes.jsonl)')
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == 'run':
        from paths import RUNS
        root = args.runs or RUNS
        records = [r for r in read_jsonl(args.stage_out / 'episodes.jsonl')
                   if 'skip' not in r and (r['family'], r['condition']) in GROUPS]
        from multiprocessing import Pool
        with Pool(args.workers) as pool, (args.out / 'reach.jsonl').open('w') as f:
            for out in pool.imap_unordered(_work, [(str(root / r['run']), r) for r in records], chunksize=2):
                f.write(json.dumps(out) + '\n')
        print(f'{len(records)} episodes -> {args.out / "reach.jsonl"}')
    else:
        print(table(read_jsonl(args.out / 'reach.jsonl')))


if __name__ == '__main__':
    main()
