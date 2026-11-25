"""Analyses of the hammer grasp runs, with the cutoffs written in the README before the data existed.

    python agentic/pilot_experiments/experiment1b_hammer_grasp/analyze_hammer.py tolerance 20261126-i1 20261126-r1
    python agentic/pilot_experiments/experiment1b_hammer_grasp/analyze_hammer.py motion 20261126-v1 20261126-i1 20261126-r1
    python agentic/pilot_experiments/experiment1b_hammer_grasp/analyze_hammer.py retry 'exp1b-*-2026112[56]-*'
    python agentic/pilot_experiments/experiment1b_hammer_grasp/analyze_hammer.py failures 'exp1b-*-2026112[56]-*'
    python agentic/pilot_experiments/experiment1b_hammer_grasp/analyze_hammer.py proprio 'exp1b-*-2026112[56]-*'
    python agentic/pilot_experiments/experiment1b_hammer_grasp/analyze_hammer.py reach 'exp1b-map-20261125-m1-*'

tolerance  first-round success against an injected displacement (along / across the handle, turn)
motion     moving-hammer conditions: measured success against what the three curves predict
retry      how often a failed first round is recovered, and whether it wrecked the scene
failures   failure types of every reaching round, and body side against object side
proprio    can the robot's own signals, now with the fingers, tell a hold right after closing
reach      clearance, pelvis back-off and wrist gap over the map, for the comparison with the box
"""
import argparse
from collections import Counter, defaultdict
import fnmatch
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'scripts'))   # agentic/scripts: paths.py
sys.path.insert(0, str(HERE.parent/'experiment1_blocking_grasp'))
from paths import AGENTIC, RUNS
from analyze_followups import auc, features, fit_logistic, group, logistic, round_times, up_z, yaw_deg

CONFIG = json.loads((AGENTIC/'configs/hammer-grasp.json').read_text())
HELD = {'success': 1, 'touched_not_held': 0, 'slipped': 0, 'knocked_over': 0}
FINGERS = ('thumb_0', 'thumb_1', 'thumb_2', 'middle_0', 'middle_1', 'index_0', 'index_1')
KINDS = {'along': 0, 'across': 1, 'turn': 2}


def runs(pattern):
    for run in sorted(RUNS.iterdir()):
        if run.is_dir() and fnmatch.fnmatch(run.name, pattern):
            try:
                result = json.loads((run/'grasp-result.json').read_text())
            except (OSError, ValueError):
                continue
            if result.get('termination'):
                yield run, result


def frame():
    f = math.radians(CONFIG['scene']['facing_deg'])
    return np.array([math.cos(f), math.sin(f)]), np.array([-math.sin(f), math.cos(f)]), np.asarray(CONFIG['scene']['table_edge_point_m'])


# ---- step two: tolerance curves -----------------------------------------------------------------

def first_success(result):
    return bool(result['attempts']) and result['attempts'][0]['label'] == 'success'


def injected(result):
    """(kind, value) of an injection run; value in cm or degrees."""
    spec = ((result.get('layout') or {}).get('spec') or {}).get('inject')
    if not spec:
        return None
    for kind, key, scale in (('along', 'along_m', 100), ('across', 'across_m', 100), ('turn', 'turn_deg', 1)):
        if abs(spec.get(key, 0.)) > 1e-9:
            return kind, round(spec[key] * scale, 3)
    return None


def curves(inject_tag, region_tag):
    """{kind: [(value, rounds, first-round successes)]}, value 0 from the region acceptance runs on
    the same seeds."""
    zero = {r['seed']: first_success(r) for _, r in runs(f'exp1b-region-{region_tag}-s*')}
    out = defaultdict(lambda: defaultdict(list))
    for _, r in runs(f'exp1b-inj-*-{inject_tag}-s*'):
        kv = injected(r)
        if kv:
            out[kv[0]][kv[1]].append(first_success(r))
    seeds = {r['seed'] for _, r in runs(f'exp1b-inj-*-{inject_tag}-s*')}
    base = [zero[s] for s in sorted(seeds) if s in zero]
    table = {}
    for kind, levels in out.items():
        levels[0.] = base
        table[kind] = [(v, len(x), int(sum(x))) for v, x in sorted(levels.items())]
    return table


def half_point(points, side):
    """Where the first-round success rate first falls to half of the zero-level rate, going from
    zero towards ``side`` (+1 or -1), by linear interpolation between tested levels. None when the
    rate stays above half over the tested range."""
    rate = {v: s / n for v, n, s in points if n}
    if 0. not in rate:
        return None
    half = rate[0.] / 2
    path = sorted((v for v in rate if v * side >= 0), key=abs)
    for a, b in zip(path, path[1:]):
        if rate[b] <= half:
            return a if rate[a] <= half else a + (b - a) * (rate[a] - half) / (rate[a] - rate[b])
    return None


def tolerance(inject_tag, region_tag):
    table = curves(inject_tag, region_tag)
    return {kind: {'levels': [{'value': v, 'rounds': n, 'first_round_successes': s,
                               'rate': None if not n else round(s / n, 3)} for v, n, s in pts],
                   'half_point_negative': half_point(pts, -1), 'half_point_positive': half_point(pts, +1)}
            for kind, pts in table.items()}


def rate_at(points, value):
    """Success rate at ``value`` by linear interpolation of the tested levels, flat beyond the ends."""
    xs = np.array([v for v, n, _ in points if n], float)
    ys = np.array([s / n for v, n, s in points if n], float)
    return float(np.interp(value, xs, ys))


def predicted(points_by_kind, along_cm, across_cm, turn_deg):
    """The pre-registered prediction for one round: the along-curve at the measured along displacement,
    times the across and turn curves each relative to their zero level."""
    base = rate_at(points_by_kind['along'], 0.)
    p = rate_at(points_by_kind['along'], along_cm)
    for kind, value in (('across', across_cm), ('turn', turn_deg)):
        zero = rate_at(points_by_kind[kind], 0.)
        p *= 0. if zero == 0 else rate_at(points_by_kind[kind], value) / zero
    return min(max(p, 0.), max(base, p))


def motion(motion_tag, inject_tag, region_tag):
    table = curves(inject_tag, region_tag)
    by = defaultdict(list)
    for run, r in runs(f'exp1b-move*-{motion_tag}-s*'):
        a = r['attempts'][0] if r['attempts'] else None
        if a is None:
            continue
        n = a.get('notes', {})
        along, across = 100 * n.get('latch_to_close_along_m', np.nan), 100 * n.get('latch_to_close_across_m', np.nan)
        turn = n.get('latch_to_close_turn_deg', np.nan)
        if 'close_time' not in n and 'reach_box_pos' in n:
            # the hand never closed: measure where the hammer was when the gate gave up
            latched, at = np.asarray(a['latched_box_pos'][:2]), np.asarray(n['reach_box_pos'][:2])
            yaw0 = n.get('latched_yaw_rad', 0.)
            axis = np.array([-math.sin(yaw0), math.cos(yaw0)])
            d = at - latched
            along, across = 100 * float(d @ axis), 100 * float(d @ np.array([axis[1], -axis[0]]))
            turn = math.degrees((n.get('reach_box_yaw_rad', yaw0) - yaw0 + math.pi) % (2 * math.pi) - math.pi)
        p = None if any(map(math.isnan, (along, across, turn))) else predicted(table, along, across, turn)
        key = (r['latch_delay_steps'], ((r.get('layout') or {}).get('spec') or {}).get('motion', {}).get('speed_mps'))
        by[key].append({'run': run.name, 'success': first_success(r), 'episode_success': r['success'],
                        'along_cm': along, 'across_cm': across, 'turn_deg': turn, 'predicted': p})
    out = {}
    for (delay, speed), rows in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        preds = [x['predicted'] for x in rows if x['predicted'] is not None]
        k, n = sum(x['success'] for x in rows), len(rows)
        expected = float(np.sum(preds)) if preds else None
        sd = float(np.sqrt(np.sum([q * (1 - q) for q in preds]))) if preds else None
        out[f'{delay} steps, {speed} m/s'] = {
            'episodes': n, 'first_round_successes': k, 'episode_successes': sum(x['episode_success'] for x in rows),
            'predicted_first_round_successes': None if expected is None else round(expected, 1),
            'predicted_sd': None if sd is None else round(sd, 2),
            'within_2sd': None if expected is None else abs(k - expected) <= 2 * max(sd, .5),
            'mean_along_cm': round(float(np.nanmean([x['along_cm'] for x in rows])), 1),
            'mean_across_cm': round(float(np.nanmean([x['across_cm'] for x in rows])), 1)}
    return out


# ---- step three: retries and scene damage -----------------------------------------------------

def damage(before_pos, before_quat, after_pos, after_quat, layout, rest_height):
    """How much one failed round changed the scene, and whether that counts as wrecked (the box's
    definition: moved more than 5 cm, turned more than 20 degrees, tipped past 30 degrees, or left the
    region; for the hammer also fallen off the table)."""
    moved = float(np.hypot(after_pos[0] - before_pos[0], after_pos[1] - before_pos[1]))
    turned = abs((yaw_deg(after_quat) - yaw_deg(before_quat) + 180) % 360 - 180)
    tipped = up_z(after_quat) < math.cos(math.radians(30))
    fallen = rest_height - after_pos[2] > .05
    outside = None
    region = CONFIG['stand'].get('region')
    if layout and region and 'robot_xy' in layout:
        forward, left, edge = frame()
        depth = float((np.asarray(after_pos[:2]) - edge) @ forward)
        lateral = float((np.asarray(after_pos[:2]) - np.asarray(layout['robot_xy'])) @ left)
        outside = not (region['depth_m'][0] - .005 <= depth <= region['depth_m'][1] + .005
                       and region['lateral_m'][0] - .005 <= lateral <= region['lateral_m'][1] + .005)
    wrecked = moved > .05 or turned > 20 or tipped or fallen or bool(outside)
    return {'moved_cm': moved * 100, 'turned_deg': turned, 'tipped': tipped, 'fallen': fallen,
            'outside': outside, 'wrecked': wrecked}


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
            injected_at = (attempts[0].get('notes') or {}).get('injected')
            if injected_at:      # the injection moved it on purpose; damage counts from there
                before_pos = injected_at['to']
                w, z = math.cos(injected_at['to_yaw_rad'] / 2), math.sin(injected_at['to_yaw_rad'] / 2)
                before_quat = [w, 0., 0., z]
            release = result.get('drive_released_at')
            if (((result.get('layout') or {}).get('spec') or {}).get('motion')):
                if release is None or release > attempts[1]['start_time']:
                    before_pos = attempts[1]['latched_box_pos']
                elif release > attempts[0]['start_time']:
                    before_pos = result['drive_release_box']
            first.update(damage(before_pos, before_quat, attempts[1]['latched_box_pos'], times[1]['box_quat'],
                                result.get('layout'), result['rest_height']))
        episodes.append(first)
    split = lambda key: {str(k): {'episodes': len(v), 'recovered': sum(e['recovered'] for e in v)}
                         for k, v in sorted(group(episodes, key).items(), key=lambda kv: str(kv[0]))}
    scene = lambda e: 'single round' if 'wrecked' not in e else 'wrecked' if e['wrecked'] else 'intact'
    return {'first_round_failures': len(episodes), 'recovered': sum(e['recovered'] for e in episodes),
            'by_scene': split(scene), 'by_first_label': split(lambda e: e['first_label']),
            'by_first_label_and_scene': split(lambda e: (e['first_label'], scene(e))), 'episodes': episodes}


# ---- step three: failure types ------------------------------------------------------------------

def failure_class(a):
    """Body side: the hand never closed on it (no contact, touched only, knocked before closing).
    Object side: the hand closed and then it was not held, slipped, or was knocked."""
    closed = 'closed' in a['events']
    label = a['label']
    if not closed:
        return {'no_contact': 'body: no contact', 'knocked_over': 'body: knocked before closing'}.get(
            label, 'body: touched, hand never closed')
    return {'knocked_over': 'object: knocked after closing', 'slipped': 'object: slipped',
            'held_at_cutoff': 'object: held at cutoff'}.get(label, 'object: closed, not held')


def failures(pattern):
    rows = [(a['attempt'], failure_class(a), a['label']) for _, r in runs(pattern) for a in r['attempts']
            if a['label'] != 'success']
    total = len(rows)
    out = {'failed_rounds': total, 'labels': dict(Counter(x[2] for x in rows)), 'classes': dict(Counter(x[1] for x in rows))}
    for name, keep in (('first_round', lambda x: x[0] == 0), ('later_rounds', lambda x: x[0] > 0)):
        sub = [x for x in rows if keep(x)]
        obj = sum(x[1].startswith('object') for x in sub)
        out[name] = {'failed_rounds': len(sub), 'object_side_share': None if not sub else round(obj / len(sub), 3),
                     'labels': dict(Counter(x[2] for x in sub))}
    share = None if not total else sum(x[1].startswith('object') for x in rows) / total
    out['object_side_share'] = share
    out['verdict'] = None if share is None else ('needs prediction of the object reaction' if share >= .5 else
                                                  'not a main problem' if share <= .25 else 'partly needed')
    return out


# ---- step three: proprioceptive hold check, now with the fingers --------------------------------

def proprio_rows(pattern):
    rows = []
    hold_s = CONFIG['policy']['hold_s']
    for run, result in runs(pattern):
        times = round_times(run)
        steps = [json.loads(l)['after'] for l in (run/'execution.jsonl').read_text().splitlines()]
        hand = {round(g['time'], 3): g for g in map(json.loads, (run/'grasp-steps.jsonl').read_text().splitlines())}
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
                moments['hold_end'] = min(t['lift_end'] + hold_s, end)
            for name, when in moments.items():
                if when > end + 1e-6:
                    continue
                k = int(np.searchsorted(clock, when - 1e-6))
                if k <= 0 or k >= len(steps):
                    continue
                g = hand.get(round(steps[k]['time'], 3))
                if g is None or g.get('right_fingers') is None:
                    continue
                q, target = np.asarray(g['right_fingers']), np.asarray(g['right_finger_target'])
                arm = features(steps[k], steps[k - 1])
                arm = {k2: v for k2, v in arm.items() if not k2.startswith('res_left') and 'left' not in k2}
                fingers = {f'finger_{n}': v for n, v in zip(FINGERS, q)}
                fingers.update({f'finger_err_{n}': v for n, v in zip(FINGERS, q - target)})
                fingers.update(finger_err_sum=float(np.abs(q - target).sum()))
                rows.append({'run': run.name, 'seed': result['seed'], 'round': a['attempt'], 'moment': name,
                             'label': a['label'], 'held': HELD[a['label']], **arm, **fingers})
    return rows


def proprio(pattern):
    rows = proprio_rows(pattern)
    names = [k for k in rows[0] if k not in ('run', 'seed', 'round', 'moment', 'label', 'held')] if rows else []
    groups = {'arm_only': [n for n in names if not n.startswith('finger')], 'fingers_only': [n for n in names if n.startswith('finger')],
              'arm_and_fingers': names}
    report = {}
    for moment in ('close+0.2s', 'mid_lift', 'hold_end'):
        sub = [r for r in rows if r['moment'] == moment]
        train = [r for r in sub if r['seed'] % 2 == 0]
        test = [r for r in sub if r['seed'] % 2 == 1]
        y = lambda rs: np.array([r['held'] for r in rs], int)
        entry = {'rounds': len(sub), 'train': len(train), 'test': len(test), 'test_held': int(y(test).sum()) if test else 0,
                 'labels': dict(Counter(r['label'] for r in sub))}
        if train and test and 0 < y(train).sum() < len(train):
            for gname, cols in groups.items():
                x = lambda rs: np.array([[r[n] for n in cols] for r in rs], float)
                singles = []
                for i, n in enumerate(cols):
                    sign = 1 if (auc(x(train)[:, i], y(train)) or .5) >= .5 else -1
                    singles.append((n, auc(sign * x(test)[:, i], y(test))))
                score, _ = logistic(x(train), y(train).astype(float))
                entry[gname] = {'logistic_test_auc': auc(score(x(test)), y(test)),
                                'best_single_signals_test_auc': sorted([s for s in singles if s[1] is not None],
                                                                       key=lambda s: -s[1])[:4],
                                'logistic_test_accuracy': float(((score(x(test)) > 0) == y(test).astype(bool)).mean())}
            entry['always_held_test_accuracy'] = float(y(test).mean())
        report[moment] = entry
    a = (report.get('close+0.2s', {}).get('arm_and_fingers') or {}).get('logistic_test_auc')
    report['verdict_close+0.2s'] = None if a is None else 'feasible' if a >= .9 else 'not feasible' if a <= .7 else 'uncertain'
    return report


# ---- step three: reach over the map -------------------------------------------------------------

def reach(pattern):
    rows = []
    for _, r in runs(pattern):
        lay = r.get('layout') or {}
        a = r['attempts'][0] if r['attempts'] else None
        if a is None or 'depth_m' not in lay:
            continue
        n = a.get('notes', {})
        rows.append({'depth': lay['depth_m'], 'lateral': lay['lateral_m'], 'yaw': lay['yaw_deg'], 'extra': lay['stance_extra_m'],
                     'clearance_m': n.get('anchor_clearance_m'), 'backoff_m': n.get('backoff_m'),
                     'wrist_gap_m': n.get('wrist_gap_m'), 'closed': 'close_time' in n, 'first_success': a['label'] == 'success',
                     'success': r['success']})
    summary = {}
    for key, fn in (('depth_cm', lambda x: round(x['depth'] * 100)), ('lateral_cm', lambda x: round(x['lateral'] * 100)),
                    ('stance_extra_cm', lambda x: round(x['extra'] * 100)), ('yaw_deg', lambda x: round(x['yaw']))):
        summary[key] = {}
        for k, v in sorted(group(rows, fn).items()):
            b = [x['backoff_m'] for x in v if x['backoff_m'] is not None]
            summary[key][k] = {'episodes': len(v), 'successes': sum(x['success'] for x in v),
                               'first_round': sum(x['first_success'] for x in v), 'hand_closed': sum(x['closed'] for x in v),
                               'backoff_cm_median': None if not b else round(100 * float(np.median(b)), 1),
                               'backoff_over_15cm': sum(x > .15 for x in b)}
    return {'episodes': len(rows), 'by': summary}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('analysis', choices=['tolerance', 'motion', 'retry', 'failures', 'proprio', 'reach'])
    p.add_argument('args', nargs='+', help='tolerance: INJECT_TAG REGION_TAG; motion: MOTION_TAG INJECT_TAG REGION_TAG; else a run pattern')
    a = p.parse_args()
    fn = {'tolerance': tolerance, 'motion': motion, 'retry': retry, 'failures': failures, 'proprio': proprio, 'reach': reach}[a.analysis]
    print(json.dumps(fn(*a.args), indent=1, default=float, ensure_ascii=False))
