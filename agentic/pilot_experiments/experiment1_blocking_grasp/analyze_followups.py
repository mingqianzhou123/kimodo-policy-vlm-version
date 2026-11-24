"""Three follow-up analyses of the blocking grasp runs, from logs already on disk (cutoffs fixed
beforehand in stance-dynamic-20261125.md).

    python agentic/pilot_experiments/experiment1_blocking_grasp/analyze_followups.py proprio 'exp1-*-2026112[45]-*'
    python agentic/pilot_experiments/experiment1_blocking_grasp/analyze_followups.py retry 'exp1-move010-20261125-d1-*'
    python agentic/pilot_experiments/experiment1_blocking_grasp/analyze_followups.py age 'exp1-*-20261125-d1-*'

proprio  can the robot's own signals tell a hold from a failed one right after the hands close
retry    how often a failed first round is recovered, and whether the failed round wrecked the scene
age      how old the latched target is when the hands close, where that time goes, and success by it
"""
import argparse
from collections import Counter, defaultdict
import fnmatch
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))  # agentic/scripts: paths.py
from paths import AGENTIC, RUNS
from kimodo_vla.core.types import G1_JOINT_NAMES

ARM = [i for i, n in enumerate(G1_JOINT_NAMES) if any(k in n for k in ('shoulder', 'elbow', 'wrist'))]
LEFT = [i for i in ARM if G1_JOINT_NAMES[i].startswith('left')]
RIGHT = [i for i in ARM if G1_JOINT_NAMES[i].startswith('right')]
HELD = {'success': 1, 'touched_not_held': 0, 'slipped': 0, 'knocked_over': 0}
CONFIG = json.loads((AGENTIC/'configs/grasp-loop.json').read_text())


def runs(pattern):
    """Finished grasp runs matching a name pattern, with their result."""
    for run in sorted(RUNS.iterdir()):
        if run.is_dir() and fnmatch.fnmatch(run.name, pattern):
            try:
                result = json.loads((run/'grasp-result.json').read_text())
            except (OSError, ValueError):
                continue
            if result.get('termination'):
                yield run, result


def round_times(run):
    """Per round: the phase start times and the planned piece ends the policy logged."""
    out = defaultdict(dict)
    for line in (run/'rounds.jsonl').read_text().splitlines():
        row = json.loads(line)
        r = out[row['round']]
        r.setdefault('latch_time', row['latch_time'])
        r.setdefault('box_quat', row['box_quat'])
        if row['kind'] == 'phase':
            r.setdefault(row['phase'], row['time'])
        for k in ('raise_end', 'advance_end', 'lift_start', 'lift_end', 'squeeze_start'):
            if k in row:
                r[k] = row[k]
    return out


# ---- one: proprioceptive hold check ----------------------------------------------------------

def features(sample, previous):
    """What the robot itself can read at one step: arm tracking residuals, arm joint speeds,
    wrist speeds, and how symmetric the two arms are."""
    joints, reference = np.asarray(sample['joints']), np.asarray(sample['reference_joints'])
    residual = joints - reference
    speed = (joints - np.asarray(previous['joints'])) / max(sample['time'] - previous['time'], 1e-6)
    wrist = np.linalg.norm(np.asarray(sample['wrist_velocity']), axis=1)
    left, right = np.abs(residual[LEFT]).sum(), np.abs(residual[RIGHT]).sum()
    named = {f'res_{G1_JOINT_NAMES[i]}': residual[i] for i in ARM}
    named.update(res_left_sum=left, res_right_sum=right, res_asymmetry=left - right,
                 speed_left=np.linalg.norm(speed[LEFT]), speed_right=np.linalg.norm(speed[RIGHT]),
                 wrist_speed_left=wrist[0], wrist_speed_right=wrist[1])
    return named


def proprio_rows(pattern):
    """One row per closed round and moment: features at close + 0.2 s, mid-lift and end of hold."""
    rows = []
    for run, result in runs(pattern):
        times = round_times(run)
        steps = [json.loads(l)['after'] for l in (run/'execution.jsonl').read_text().splitlines()]
        clock = np.array([s['time'] for s in steps])
        for a in result['attempts']:
            if a['label'] not in HELD or 'closed' not in a['events']:
                continue
            t = times.get(a['attempt'], {})
            close = a['events']['closed']
            end = a['end_time'] if a['end_time'] is not None else clock[-1]
            moments = {'close+0.2s': close + .2}
            if 'lift_start' in t and 'lift_end' in t:
                moments['mid_lift'] = (t['lift_start'] + t['lift_end']) / 2
                moments['hold_end'] = min(t['lift_end'] + CONFIG['policy']['hold_s'], end)
            for name, when in moments.items():
                if when > end + 1e-6:
                    continue                      # the round (or the episode) was over by then
                k = int(np.searchsorted(clock, when - 1e-6))
                if k <= 0 or k >= len(steps):
                    continue
                rows.append({'run': run.name, 'seed': result['seed'], 'round': a['attempt'], 'moment': name,
                             'label': a['label'], 'held': HELD[a['label']], **features(steps[k], steps[k - 1])})
    return rows


def auc(score, y):
    """Area under the ROC curve by ranks (ties share their mean rank)."""
    score, y = np.asarray(score, float), np.asarray(y, int)
    pos, neg = y.sum(), len(y) - y.sum()
    if not pos or not neg:
        return None
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def logistic(x, y, l2=1., steps=3000, rate=.1):
    """Plain L2 logistic regression on standardized features; returns a scoring function."""
    mean, std = x.mean(0), x.std(0) + 1e-9
    z = (x - mean) / std
    w, b = np.zeros(z.shape[1]), 0.
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(z @ w + b)))
        w -= rate * (z.T @ (p - y) / len(y) + l2 * w / len(y))
        b -= rate * float((p - y).mean())
    return lambda q: ((q - mean) / std) @ w + b, dict(zip(range(len(w)), w))


def proprio(pattern):
    rows = proprio_rows(pattern)
    names = [k for k in rows[0] if k not in ('run', 'seed', 'round', 'moment', 'label', 'held')] if rows else []
    report = {}
    for moment in ('close+0.2s', 'mid_lift', 'hold_end'):
        sub = [r for r in rows if r['moment'] == moment]
        train = [r for r in sub if r['seed'] % 2 == 0]
        test = [r for r in sub if r['seed'] % 2 == 1]
        x = lambda rs: np.array([[r[n] for n in names] for r in rs], float)
        y = lambda rs: np.array([r['held'] for r in rs], int)
        entry = {'rounds': len(sub), 'train': len(train), 'test': len(test),
                 'test_held': int(y(test).sum()) if test else 0,
                 'labels': dict(Counter(r['label'] for r in sub))}
        if train and test and 0 < y(train).sum() < len(train):
            # single signals: direction chosen on the training rounds, AUC reported on the test rounds
            singles = []
            for i, n in enumerate(names):
                sign = 1 if (auc(x(train)[:, i], y(train)) or .5) >= .5 else -1
                singles.append((n, auc(sign * x(test)[:, i], y(test))))
            singles = [s for s in singles if s[1] is not None]
            entry['best_single_signals_test_auc'] = sorted(singles, key=lambda s: -s[1])[:5]
            score, weights = logistic(x(train), y(train).astype(float))
            entry['logistic_test_auc'] = auc(score(x(test)), y(test))
            predicted = score(x(test)) > 0
            entry['logistic_test_accuracy'] = float((predicted == y(test).astype(bool)).mean())
            entry['always_held_test_accuracy'] = float(y(test).mean())
            entry['largest_weights'] = sorted(((names[i], round(float(v), 2)) for i, v in weights.items()),
                                              key=lambda t: -abs(t[1]))[:6]
        report[moment] = entry
    a = report.get('close+0.2s', {}).get('logistic_test_auc')
    report['verdict_close+0.2s'] = None if a is None else 'feasible' if a >= .9 else 'not feasible' if a <= .7 else 'uncertain'
    return report


# ---- two: cost of trial and error ------------------------------------------------------------

def up_z(q):
    _, x, y, _ = q
    return 1 - 2 * (x * x + y * y)


def yaw_deg(q):
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def damage(before_pos, before_quat, after_pos, after_quat, layout):
    """How much one failed round changed the scene, and whether that counts as wrecked."""
    moved = float(np.hypot(after_pos[0] - before_pos[0], after_pos[1] - before_pos[1]))
    turned = abs((yaw_deg(after_quat) - yaw_deg(before_quat) + 180) % 360 - 180)
    tipped = up_z(after_quat) < math.cos(math.radians(30))
    outside = None
    if layout and CONFIG['stand'].get('region'):
        region = CONFIG['stand']['region']
        depth = after_pos[0] - CONFIG['scene']['table_edge_x_m']
        lateral = after_pos[1] - layout['robot_xy'][1]
        outside = not (region['depth_m'][0] <= depth <= region['depth_m'][1]
                       and region['lateral_m'][0] <= lateral <= region['lateral_m'][1])
    wrecked = moved > .05 or turned > 20 or tipped or bool(outside)
    return {'moved_cm': moved * 100, 'turned_deg': turned, 'tipped': tipped, 'outside': outside, 'wrecked': wrecked}


def retry(pattern):
    episodes = []
    for run, result in runs(pattern):
        attempts = result['attempts']
        if not attempts or attempts[0]['label'] == 'success':
            continue
        times = round_times(run)
        first = {'run': run.name, 'first_label': attempts[0]['label'], 'recovered': result['success'],
                 'rounds': len(attempts)}
        if len(attempts) > 1:
            before_pos, before_quat = attempts[0]['latched_box_pos'], times[0]['box_quat']
            release = result.get('drive_released_at')
            if result.get('layout', {}) and (result['layout'] or {}).get('speed_mps'):
                # a driven box moves by itself until a hand first touches it; only what happens after counts
                if release is None or release > attempts[1]['start_time']:
                    before_pos = attempts[1]['latched_box_pos']
                elif release > attempts[0]['start_time']:
                    before_pos = result['drive_release_box']
            first.update(damage(before_pos, before_quat, attempts[1]['latched_box_pos'], times[1]['box_quat'],
                                result.get('layout')))
        episodes.append(first)
    split = lambda key: {str(k): {'episodes': len(v), 'recovered': sum(e['recovered'] for e in v)}
                         for k, v in sorted(group(episodes, key).items(), key=lambda kv: str(kv[0]))}
    return {'first_round_failures': len(episodes), 'recovered': sum(e['recovered'] for e in episodes),
            'by_scene': split(lambda e: 'single round' if 'wrecked' not in e else
                              'wrecked' if e['wrecked'] else 'intact'),
            'by_first_label': split(lambda e: e['first_label']),
            'by_first_label_and_scene': split(lambda e: (e['first_label'], 'single' if 'wrecked' not in e else
                                                         'wrecked' if e['wrecked'] else 'intact')),
            'episodes': episodes}


def group(items, key):
    out = defaultdict(list)
    for item in items:
        out[key(item)].append(item)
    return out


# ---- three: age of the latched target ---------------------------------------------------------

AGE_BINS_S = (0, 4, 5, 6, 8, 10, 12)


def age(pattern):
    rows = []
    for run, result in runs(pattern):
        delay = result['latch_delay_steps'] / 50
        times = round_times(run)
        for a in result['attempts']:
            n, t = a.get('notes', {}), times.get(a['attempt'], {})
            at = n.get('close_time', n.get('reach_time'))
            if at is None or 'advance_end' not in t:
                continue
            latch = a['start_time']
            box = n.get('close_box_pos', n.get('reach_box_pos'))
            rows.append({'run': run.name, 'round': a['attempt'], 'success': a['label'] == 'success',
                         'closed': 'close_time' in n, 'age_s': at - latch, 'wait_s': delay,
                         'reach_s': t['advance_end'] - latch - delay, 'gate_s': at - t['advance_end'],
                         'moved_cm': float(np.hypot(box[0] - a['latched_box_pos'][0], box[1] - a['latched_box_pos'][1])) * 100})
    table = []
    for low, high in zip(AGE_BINS_S, list(AGE_BINS_S[1:]) + [None]):
        inside = [r for r in rows if r['age_s'] >= low and (high is None or r['age_s'] < high)]
        table.append({'from_s': low, 'to_s': high, 'rounds': len(inside), 'successes': sum(r['success'] for r in inside),
                      'mean_moved_cm': None if not inside else round(float(np.mean([r['moved_cm'] for r in inside])), 2)})
    mean = lambda k, rs: None if not rs else round(float(np.mean([r[k] for r in rs])), 2)
    return {'rounds': len(rows), 'by_age': table,
            'mean_split_s': {k: mean(k, rows) for k in ('wait_s', 'reach_s', 'gate_s', 'age_s')}, 'rows': rows}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('analysis', choices=['proprio', 'retry', 'age'])
    p.add_argument('pattern')
    p.add_argument('--rows', action='store_true', help='Also print every row')
    a = p.parse_args()
    out = {'proprio': proprio, 'retry': retry, 'age': age}[a.analysis](a.pattern)
    if not a.rows:
        for key in ('episodes', 'rows'):
            out.pop(key, None) if isinstance(out, dict) else None
    print(json.dumps(out, indent=1, ensure_ascii=False, default=float))
