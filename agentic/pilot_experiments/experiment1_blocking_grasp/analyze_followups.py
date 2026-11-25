"""Three follow-up analyses of the blocking grasp runs, from logs already on disk (cutoffs fixed
beforehand in stance-dynamic-20261125.md).

    python agentic/pilot_experiments/experiment1_blocking_grasp/analyze_followups.py proprio 'exp1-*-2026112[45]-*'
    python agentic/pilot_experiments/experiment1_blocking_grasp/analyze_followups.py retry 'exp1-move010-20261125-d1-*'
    python agentic/pilot_experiments/experiment1_blocking_grasp/analyze_followups.py age 'exp1-*-20261125-d1-*'

proprio  can the robot's own signals tell a hold from a failed one right after the hands close
retry    how often a failed first round is recovered, and whether the failed round wrecked the scene
age      how old the latched target is when the hands close, where that time goes, and success by it
phase    whether delay and speed act only through how far the box moved by the time the hands close
failures which failures are the body's (never got there) and which are the box's reaction
contacts how much the box's motion right after the first touch varies among similar touches
predict  how well a simple model says where the box is after a round, against "it stays put"
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

AGE_BINS_S = (0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14)


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


# ---- four: do delay and speed act only through the phase of the back-and-forth? ----------

def steps_of(run):
    return [json.loads(l) for l in (run/'grasp-steps.jsonl').read_text().splitlines()]


def fit_logistic(x, y, iterations=50):
    """Maximum-likelihood logistic regression by Newton steps; returns weights and log-likelihood."""
    x = np.c_[np.ones(len(x)), x]
    w = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-np.clip(x @ w, -30, 30)))
        hessian = x.T @ (x * (p * (1 - p))[:, None]) + 1e-6 * np.eye(x.shape[1])
        w += np.linalg.solve(hessian, x.T @ (y - p))
    p = np.clip(1 / (1 + np.exp(-np.clip(x @ w, -30, 30))), 1e-12, 1 - 1e-12)
    return w, float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def phase(pattern):
    from arena.box_layout import BoxMotion
    from scipy.stats import chi2, pearsonr
    rows = []
    for run, result in runs(pattern):
        lay = result.get('layout') or {}
        if not lay.get('speed_mps'):
            continue
        a = result['attempts'][0]
        n = a.get('notes', {})
        at = n.get('close_time', n.get('reach_time'))
        if at is None:
            continue
        steps = steps_of(run)
        start_y = next(r['box_com'][1] for r in steps if r['time'] >= result['clock_start'] - 1e-6)
        motion = BoxMotion(lay['segment_y'], lay['speed_mps'], start_y, lay['direction'])
        t0, t1 = a['start_time'] - result['clock_start'], at - result['clock_start']
        box = n.get('close_box_pos', n.get('reach_box_pos'))
        rows.append({'run': run.name, 'success': a['label'] == 'success', 'delay_steps': result['latch_delay_steps'],
                     'speed': lay['speed_mps'], 'elapsed_s': t1 - t0,
                     'phase': (lay['speed_mps'] * (t1 - t0) / (2 * motion.length)) % 1,
                     'scripted_cm': abs(motion.position(t1) - motion.position(t0)) * 100,
                     'measured_cm': abs(box[1] - a['latched_box_pos'][1]) * 100})
    y = np.array([r['success'] for r in rows], float)
    d = np.array([[r['measured_cm']] for r in rows])
    cond = np.array([[r['delay_steps'] == 194, r['delay_steps'] == 383, r['speed'] > .05] for r in rows], float)
    _, ll0 = fit_logistic(np.zeros((len(rows), 0)), y)
    w1, ll1 = fit_logistic(d, y)
    w2, ll2 = fit_logistic(np.c_[d, cond], y)
    _, ll3 = fit_logistic(cond, y)
    p_disp, p_cond = float(chi2.sf(2 * (ll1 - ll0), 1)), float(chi2.sf(2 * (ll2 - ll1), 3))
    p_disp_given_cond = float(chi2.sf(2 * (ll2 - ll3), 1))
    by = defaultdict(list)
    for r in rows:
        by[(r['delay_steps'], r['speed'])].append(r)
    conditions = {f'{k[0]} steps, {k[1]} m/s': {
        'rounds': len(v), 'successes': int(sum(r['success'] for r in v)),
        'elapsed_s': round(float(np.mean([r['elapsed_s'] for r in v])), 2),
        'phase_mean': round(float(np.mean([r['phase'] for r in v])), 2),
        'scripted_cm_mean': round(float(np.mean([r['scripted_cm'] for r in v])), 2),
        'measured_cm_mean': round(float(np.mean([r['measured_cm'] for r in v])), 2)} for k, v in sorted(by.items())}
    strata = {}
    for low, high in ((0, 3), (3, 5), (5, 20)):
        inside = [r for r in rows if low <= r['measured_cm'] < high]
        strata[f'{low}-{high} cm'] = {f'{k[0]}/{k[1]}': f"{sum(r['success'] for r in v if low <= r['measured_cm'] < high)}"
                                      f"/{sum(low <= r['measured_cm'] < high for r in v)}" for k, v in sorted(by.items())}
        strata[f'{low}-{high} cm']['all'] = f"{sum(r['success'] for r in inside)}/{len(inside)}"
    return {'rounds': len(rows), 'by_condition': conditions, 'success_by_measured_lateral_and_condition': strata,
            'displacement_coefficient_per_cm': round(float(w1[1]), 3),
            'p_displacement_alone': p_disp, 'p_delay_and_speed_beyond_displacement': p_cond,
            'p_displacement_beyond_delay_and_speed': p_disp_given_cond,
            'scripted_vs_measured_r': round(float(pearsonr([r['scripted_cm'] for r in rows], [r['measured_cm'] for r in rows])[0]), 3),
            'verdict': 'only through displacement' if p_cond > .05 and p_disp < .01 else 'delay or speed matter beyond displacement',
            'rows': rows}


# ---- five: which side of the robot-box pair failed ------------------------------------------

def failures(pattern):
    rows = []
    for run, result in runs(pattern):
        times = round_times(run)
        for a in result['attempts']:
            if a['label'] == 'success' or 'raise' not in times.get(a['attempt'], {}):
                continue                      # only rounds that reached for the box
            e = a['events']
            closed = 'closed' in e
            if a['label'] == 'no_contact':
                kind = 'body: no contact'
            elif a['label'] == 'touched_not_held':
                kind = 'object: closed, not held' if closed else 'body: grazed, hands never closed'
            elif a['label'] == 'knocked_over':
                kind = ('object: knocked over after closing' if closed and e['knocked_over'] >= e['closed']
                        else 'body: knocked over before closing')
            else:
                kind = 'object: slipped' if a['label'] == 'slipped' else 'object: held at cutoff'
            rows.append({'run': run.name, 'round': a['attempt'], 'kind': kind})
    count = Counter(r['kind'] for r in rows)
    obj = sum(v for k, v in count.items() if k.startswith('object'))
    share = None if not rows else obj / len(rows)
    return {'failed_rounds': len(rows), 'kinds': dict(count), 'object_share': share,
            'verdict': None if share is None else 'needed' if share >= .5 else 'not the main problem' if share <= .25 else 'partly',
            'rows': rows}


# ---- six: what the box does right after the first touch -------------------------------------

def contacts(pattern, window_s=.5, floor=.05):
    rows = []
    for run, result in runs(pattern):
        steps = steps_of(run)
        seen = set()
        for k in range(1, len(steps)):
            s = steps[k]
            if s['round'] in seen:
                continue
            left, right, other = s['left_n'] >= floor, s['right_n'] >= floor, s['robot_other_n'] >= floor
            if not (left or right or other):
                continue
            seen.add(s['round'])
            later = next((x for x in steps[k:] if x['time'] >= s['time'] + window_s - 1e-6), None)
            if later is None:
                continue
            part = 'both hands' if left and right else 'left hand' if left else 'right hand' if right else 'body'
            hand = [0, 1] if part in ('both hands', 'body') else [0] if left else [1]
            dt = s['time'] - steps[k - 1]['time']
            wrist_v = (np.asarray(s['wrist_pos'])[hand] - np.asarray(steps[k - 1]['wrist_pos'])[hand]).mean(0) / dt
            box_v = (np.asarray(s['box_com']) - np.asarray(steps[k - 1]['box_com'])) / dt
            rel = wrist_v - box_v
            offset = float(np.abs(np.asarray(s['wrist_pos'])[hand, 1] - s['box_com'][1]).min())
            rows.append({'run': run.name, 'round': s['round'], 'part': part, 'rel_speed': float(np.linalg.norm(rel[:2])),
                         'rel_lateral_mps': float(rel[1]), 'rel_forward_mps': float(rel[0]),
                         'inside': offset < .125,
                         'moved_cm': float(np.linalg.norm(np.subtract(later['box_com'][:2], s['box_com'][:2]))) * 100,
                         'turned_deg': abs((yaw_deg(later['box_quat']) - yaw_deg(s['box_quat']) + 180) % 360 - 180)})
    if not rows:
        return {'contacts': 0}
    cuts = np.quantile([r['rel_speed'] for r in rows], [1 / 3, 2 / 3])
    for r in rows:
        r['speed_bin'] = int(np.searchsorted(cuts, r['rel_speed']))
    groups = group(rows, lambda r: (r['part'], r['speed_bin'], r['inside']))
    out = {'contacts': len(rows), 'speed_cuts_mps': [round(float(c), 3) for c in cuts], 'note': 'box mass 0.05 kg'}
    for key in ('moved_cm', 'turned_deg'):
        total = float(np.std([r[key] for r in rows]))
        within = sum(len(g) * float(np.std([r[key] for r in g])) for g in groups.values()) / len(rows)
        ratio = within / total if total > 0 else None
        out[key] = {'overall_mean': round(float(np.mean([r[key] for r in rows])), 2), 'overall_sd': round(total, 2),
                    'within_group_sd': round(within, 2), 'ratio': None if ratio is None else round(ratio, 3),
                    'verdict': None if ratio is None else 'spread is large: needs prediction' if ratio >= .75
                    else 'simple rules suffice' if ratio <= .5 else 'partly'}
    out['groups'] = {str(k): {'n': len(g), 'moved_cm_mean': round(float(np.mean([r['moved_cm'] for r in g])), 2),
                              'moved_cm_sd': round(float(np.std([r['moved_cm'] for r in g])), 2),
                              'turned_deg_sd': round(float(np.std([r['turned_deg'] for r in g])), 2)}
                     for k, g in sorted(groups.items(), key=lambda kv: str(kv[0]))}
    out['rows'] = rows
    return out


# ---- seven: can a simple model say where the box ends up after a round ------------------------

def predict_rows(pattern):
    rows = []
    for run, result in runs(pattern):
        times = round_times(run)
        attempts = result['attempts']
        lay = result.get('layout') or {}
        steps = None
        for i, a in enumerate(attempts):
            t = times.get(a['attempt'], {})
            if a['label'] == 'success' or 'raise' not in t:
                continue
            notes = json_round_notes(run, a['attempt'])
            if 'anchor_root' not in notes or 'wrist_targets' not in notes:
                continue
            start = np.asarray(a['latched_box_pos'][:2])
            if i + 1 < len(attempts):
                end = np.asarray(attempts[i + 1]['latched_box_pos'][:2])
            else:
                steps = steps or steps_of(run)
                end = np.asarray(steps[-1]['box_com'][:2])
            steps = steps or steps_of(run)
            k = next((j for j, x in enumerate(steps) if x['time'] >= a['start_time'] - 1e-6), 0)
            nxt = steps[min(k + 1, len(steps) - 1)]
            vel = (np.asarray(nxt['box_com'][:2]) - np.asarray(steps[k]['box_com'][:2])) / max(nxt['time'] - steps[k]['time'], 1e-6)
            root = np.asarray(notes['anchor_root'][:2])
            pre, squeeze = np.asarray(notes['wrist_targets']['pre']), np.asarray(notes['wrist_targets']['squeeze'])
            closed = 'closed' in a['events']
            x = [*(start - root), yaw_deg(times[a['attempt']]['box_quat']) / 45, *vel,
                 *(pre[:, :2] - start).ravel(), *(squeeze[:, :2] - start).ravel(),
                 float(closed), (a['events']['closed'] - a['start_time']) if closed else 0., float(lay.get('speed_mps', 0.))]
            rows.append({'run': run.name, 'seed': result['seed'], 'round': a['attempt'], 'x': x,
                         'delta_cm': ((end - start) * 100).tolist()})
    return rows


def json_round_notes(run, index):
    notes = {}
    for line in (run/'rounds.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['round'] == index:
            notes.update(row)
    return notes


def predictability(pattern, l2=1., k=5):
    rows = predict_rows(pattern)
    train = [r for r in rows if r['seed'] % 2 == 0]
    test = [r for r in rows if r['seed'] % 2 == 1]
    if not train or not test:
        return {'rounds': len(rows)}
    xtr, xte = np.array([r['x'] for r in train]), np.array([r['x'] for r in test])
    ytr, yte = np.array([r['delta_cm'] for r in train]), np.array([r['delta_cm'] for r in test])
    mean, std = xtr.mean(0), xtr.std(0) + 1e-9
    ztr, zte = (xtr - mean) / std, (xte - mean) / std
    a = np.c_[np.ones(len(ztr)), ztr]
    w = np.linalg.solve(a.T @ a + l2 * np.diag([0] + [1] * ztr.shape[1]), a.T @ ytr)
    ridge = np.c_[np.ones(len(zte)), zte] @ w
    dist = np.linalg.norm(zte[:, None] - ztr[None], axis=2)
    knn = ytr[np.argsort(dist, axis=1)[:, :k]].mean(1)
    err = lambda pred: float(np.linalg.norm(pred - yte, axis=1).mean())
    base = float(np.linalg.norm(yte, axis=1).mean())
    best = min(err(ridge), err(knn))
    gain = 1 - best / base if base > 0 else None
    median_moved = float(np.median(np.linalg.norm(yte, axis=1)))
    return {'rounds': len(rows), 'train': len(train), 'test': len(test),
            'test_mean_moved_cm': round(base, 2), 'test_median_moved_cm': round(median_moved, 2),
            'error_cm': {'box stays put': round(base, 2), 'ridge': round(err(ridge), 2), f'{k}-nearest': round(err(knn), 2)},
            'best_reduction': None if gain is None else round(gain, 3),
            'verdict': None if gain is None else 'predictable with a simple model' if gain >= .5
            else 'not predictable with a simple model' if gain <= .2 else 'partly predictable'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('analysis', choices=['proprio', 'retry', 'age', 'phase', 'failures', 'contacts', 'predict'])
    p.add_argument('pattern')
    p.add_argument('--rows', action='store_true', help='Also print every row')
    a = p.parse_args()
    out = {'proprio': proprio, 'retry': retry, 'age': age, 'phase': phase, 'failures': failures,
           'contacts': contacts, 'predict': predictability}[a.analysis](a.pattern)
    if not a.rows:
        for key in ('episodes', 'rows'):
            out.pop(key, None) if isinstance(out, dict) else None
    print(json.dumps(out, indent=1, ensure_ascii=False, default=float))
