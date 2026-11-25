"""Experiment four's tables, deviation curve, failure labels and judgment, from run folders only.

    python analyze_exp4.py extract OUT.json 'exp4-*-20261125-p1-*' 'exp1-*move0*-20261125-d1-*' ...
    python analyze_exp4.py report OUT.json > report.json

``extract`` reads every finished run matching the patterns once and keeps one row per episode
(with one sub-row per round); ``report`` computes everything the record's pre-registered
analysis asks for from those rows. The rules are written in the record (README.md) before the
data were looked at; the thresholds below are those rules.
"""
import argparse
from collections import Counter, defaultdict
import fnmatch
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'scripts'))
from paths import RUNS   # noqa: E402

DELAY_S = {25: .5, 75: 1.5, 194: 3.88, 383: 7.66, 620: 12.4}
TOLERANCE_M = .03          # hands further than this from the box centre line at the close miss the box
MOVING_MPS = .02           # a box faster than this at the close is moving
BACKOFF_M = .15            # pelvis backed off this far: the reach could not arrive
FALL_HEIGHT_M = .5         # pelvis below this: fallen
UPDATE_WINDOW_S = .1       # steps this close to an update count as "around the update"
BINS_CM = (0, 1, 2, 3, 5, 8, 1000)
LABELS = ('backed_off', 'delay_beyond_limit', 'stale_target', 'stale_by_replan', 'hands_lag',
          'moving_at_close', 'other')


def jsonl(path):
    try:
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except OSError:
        return []


def finished(patterns):
    for run in sorted(RUNS.iterdir()):
        if run.is_dir() and any(fnmatch.fnmatch(run.name, p) for p in patterns):
            try:
                result = json.loads((run/'grasp-result.json').read_text())
            except (OSError, ValueError):
                continue
            if result.get('termination'):
                yield run, result


def condition(run, result):
    manifest = json.loads((run/'run.json').read_text())
    update = manifest.get('target_update') or 'blocking'
    delay = 'jitter' if manifest.get('latency_samples') else int(result['latch_delay_steps'])
    return update, delay, float((result.get('layout') or {}).get('speed_mps', 0.))


class Steps:
    """Per-step box and wrist measurements of one run, looked up by time."""
    def __init__(self, rows):
        self.t = np.array([r['time'] for r in rows])
        self.box = np.array([r['box_com'] for r in rows])
        self.wrist = np.array([r['wrist_pos'] for r in rows])

    def index(self, time):
        return int(np.clip(np.searchsorted(self.t, time - 1e-4), 0, len(self.t) - 1))

    def box_at(self, time):
        return self.box[self.index(time)]

    def speed_at(self, time, span=.1):
        i, j = self.index(time), self.index(time - span)
        dt = self.t[i] - self.t[j]
        return float(np.linalg.norm(self.box[i, :2] - self.box[j, :2]) / dt) if dt > 0 else 0.


def targets(rounds_rows):
    """Per round, the targets the policy aimed at: [(from time, box pos, info time)]."""
    out = defaultdict(list)
    for row in rounds_rows:
        if row['kind'] == 'latch':
            out[row['round']].append((row['latch_time'], row['box_pos'], row['latch_time']))
        elif row['kind'] == 'update':
            out[row['round']].append((row['now'], row['target_box_pos'], row['info_time']))
    return out


def in_force(aims, time):
    chosen = aims[0]
    for aim in aims:
        if aim[0] <= time + 1e-4:
            chosen = aim
    return chosen


def round_row(attempt, aims, steps, phases):
    n = attempt.get('notes', {})
    row = {'round': attempt['attempt'], 'label': attempt['label'], 'reached': 'raise' in phases,
           'backoff_m': n.get('backoff_m'), 'closed': 'close_time' in n}
    at = n.get('close_time', n.get('reach_time'))
    if at is None or not aims:
        return {**row, 'at': None}
    _, target, info_time = in_force(aims, at)
    box = steps.box_at(at)
    wrist = steps.wrist[steps.index(at)]
    mid_y = float(wrist[:, 1].mean())
    row.update(at=at, age_s=at - info_time,
               target_lateral_m=float(box[1] - target[1]),           # box minus where the robot thinks it is
               target_planar_m=float(np.linalg.norm(box[:2] - np.asarray(target[:2]))),
               hand_lateral_m=mid_y - float(box[1]),                  # hands' centre minus the box
               exec_lateral_m=mid_y - float(target[1]),               # hands' centre minus the aim
               box_speed_mps=steps.speed_at(at))
    return row


def label(first, attempt_count, success):
    """The record's failure rules, in order, for an episode's first round."""
    if success and first['label'] == 'success':
        return None
    if not first['reached']:
        return 'delay_beyond_limit'
    if first['at'] is None:
        return 'other'
    if not first['closed'] and (first['backoff_m'] or 0) >= BACKOFF_M:
        return 'backed_off'
    if not first['closed']:
        return 'hands_lag' if first['box_speed_mps'] > MOVING_MPS else 'other'
    hand, aim, execution = abs(first['hand_lateral_m']), abs(first['target_lateral_m']), abs(first['exec_lateral_m'])
    if hand > TOLERANCE_M:
        if aim >= execution:
            return 'stale_by_replan' if first['age_s'] <= .6 else 'stale_target'
        return 'hands_lag'
    if first['box_speed_mps'] > MOVING_MPS and first['label'] in ('knocked_over', 'touched_not_held', 'slipped'):
        return 'moving_at_close'
    return 'other'


def smoothness(run, start, end, updates):
    """Largest step of the consumed reference joints and wrist speed peak between start and end,
    around target updates and elsewhere."""
    plan_times = [p['time'] for p in jsonl(run/'plans.jsonl')]
    previous, out = None, {'update': [0., 0.], 'boundary': [0., 0.], 'other': [0., 0.]}
    for row in jsonl(run/'execution.jsonl'):
        t = row['before']['time']
        q = row.get('reference_joints')
        if t < start - 1e-6 or t > end + 1e-6 or q is None:
            previous = q
            continue
        jump = float(np.abs(np.subtract(q, previous)).max()) if previous is not None else 0.
        previous = q
        speed = float(np.linalg.norm(row['after']['wrist_velocity'], axis=1).max())
        where = ('update' if any(abs(t - u) <= UPDATE_WINDOW_S for u in updates) else
                 'boundary' if any(abs(t - p) <= UPDATE_WINDOW_S for p in plan_times) else 'other')
        out[where] = [max(out[where][0], jump), max(out[where][1], speed)]
    return {k: {'joint_step_rad': round(v[0], 4), 'wrist_speed_mps': round(v[1], 3)} for k, v in out.items()}


def episode(run, result):
    update, delay, speed = condition(run, result)
    rounds_rows = jsonl(run/'rounds.jsonl')
    aims = targets(rounds_rows)
    phases = defaultdict(dict)
    for row in rounds_rows:
        if row['kind'] == 'phase':
            phases[row['round']].setdefault(row['phase'], row['time'])
    steps = Steps(jsonl(run/'grasp-steps.jsonl'))
    rows = [round_row(a, aims.get(a['attempt'], []), steps, phases[a['attempt']]) for a in result['attempts']]
    changes = [r['now'] for r in rounds_rows if r['kind'] == 'update' and r.get('changed')]
    first = rows[0] if rows else {'reached': False, 'at': None, 'label': 'no_contact'}
    roots = [r['before']['root_pos'][2] for r in jsonl(run/'execution.jsonl')]
    start, last = result['clock_start'], result['last_time']
    calls = (int((last - start) // .5) + 1 if update == 'fixed' else 1 if update == 'anchored' else len(rows))
    raise_time = phases[0].get('raise')
    return {'run': run.name, 'seed': result['seed'], 'update': update, 'delay': delay, 'speed': speed,
            'success': bool(result['success']), 'first_success': bool(result['first_attempt_success']),
            'completion_s': result['completion_s'], 'fell': bool(roots) and min(roots) < FALL_HEIGHT_M,
            'attempts': len(rows), 'calls': calls, 'updates': len(changes),
            'label': label(first, len(rows), result['success']), 'rounds': rows,
            'smoothness': smoothness(run, raise_time, first['at'], [c for c in changes if c >= raise_time])
            if raise_time is not None and first.get('at') else None}


# ---- report ------------------------------------------------------------------------------------

def fit_logistic(x, y, iterations=50):
    x = np.c_[np.ones(len(x)), x]
    w = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-np.clip(x @ w, -30, 30)))
        w += np.linalg.solve(x.T @ (x * (p * (1 - p))[:, None]) + 1e-6 * np.eye(x.shape[1]), x.T @ (y - p))
    p = np.clip(1 / (1 + np.exp(-np.clip(x @ w, -30, 30))), 1e-12, 1 - 1e-12)
    return w, float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def rate(rows, key='success'):
    return sum(r[key] for r in rows) / len(rows) if rows else float('nan')


def cells(eps):
    table = defaultdict(list)
    for e in eps:
        table[(e['update'], str(e['delay']), e['speed'])].append(e)
    out = {}
    for (u, d, v), rows in sorted(table.items()):
        done = [r['completion_s'] for r in rows if r['success']]
        out[f'{u} {d} {v}'] = {'n': len(rows), 'success': sum(r['success'] for r in rows),
            'first_success': sum(r['first_success'] for r in rows),
            'completion_median_s': round(float(np.median(done)), 2) if done else None,
            'falls': sum(r['fell'] for r in rows), 'calls_mean': round(float(np.mean([r['calls'] for r in rows])), 1),
            'updates_mean': round(float(np.mean([r['updates'] for r in rows])), 1),
            'labels': dict(Counter(r['label'] for r in rows if r['label']))}
    return out


def judgment(eps):
    """The plan's verdict for each moving speed, with the record's pooling."""
    out = {}
    fixed_delays = [d for d in DELAY_S]
    for v in (.03, .1):
        pick = lambda u, ds: [e for e in eps if e['update'] == u and e['speed'] == v and e['delay'] in ds]
        anchored, fixed = pick('anchored', fixed_delays), pick('fixed', fixed_delays)
        gap = rate(anchored) - rate(fixed)
        drop = rate(pick('anchored', [25])) - rate(pick('anchored', [620]))
        local, closed = rate(pick('anchored', [25, 75])), rate(pick('anchored', [194, 383, 620]))
        holds = gap >= .25 and drop <= .25
        out[f'{v} m/s'] = {'anchored_minus_fixed_pp': round(100 * gap, 1), 'anchored_drop_0.5_to_12.4_pp': round(100 * drop, 1),
            'anchored_local_vs_closed_pp': round(100 * (local - closed), 1),
            'per_delay_pp': {DELAY_S[d]: round(100 * (rate(pick('anchored', [d])) - rate(pick('fixed', [d]))), 1)
                             for d in fixed_delays},
            'verdict': 'core finding holds' if holds else 'core finding does not hold',
            'local_only': bool(holds and local - closed > .25)}
    return out


def curve(eps):
    """First rounds of moving cells by |target deviation| at the close, per update mode, plus the test."""
    rows = [(e['update'], e['rounds'][0]) for e in eps if e['speed'] > 0 and e['delay'] != 'jitter'
            and e['rounds'] and e['rounds'][0].get('at') is not None]
    table = {}
    for u in ('blocking', 'fixed', 'anchored'):
        mine = [r for k, r in rows if k == u]
        table[u] = {f'{lo}-{hi}': f"{sum(r['label'] == 'success' for r in b)}/{len(b)}"
                    for lo, hi in zip(BINS_CM, BINS_CM[1:])
                    for b in [[r for r in mine if lo <= abs(r['target_lateral_m']) * 100 < hi]]}
    from scipy.stats import chi2
    y = np.array([r['label'] == 'success' for _, r in rows], float)
    d = np.array([[abs(r['target_lateral_m']) * 100] for _, r in rows])
    arm = np.array([[u == 'fixed', u == 'anchored'] for u, _ in rows], float)
    w1, ll1 = fit_logistic(d, y)
    _, ll2 = fit_logistic(np.c_[d, arm], y)
    p = float(chi2.sf(2 * (ll2 - ll1), 2))
    small = {u: [r for k, r in rows if k == u and abs(r['target_lateral_m']) < .02] for u in ('blocking', 'fixed', 'anchored')}
    others = small['blocking'] + small['fixed']
    shifted = p <= .05 and rate([{'success': r['label'] == 'success'} for r in small['anchored']]) <= \
        rate([{'success': r['label'] == 'success'} for r in others]) - .15
    return {'rounds': len(rows), 'by_update': table, 'coefficient_per_cm': round(float(w1[1]), 3),
            'p_update_beyond_deviation': p,
            'below_2cm': {u: f"{sum(r['label'] == 'success' for r in v)}/{len(v)}" for u, v in small.items()},
            'verdict': 'same curve: target deviation explains it' if p > .05 else
                       'bottleneck shifted' if shifted else 'update mode matters beyond deviation, not at small deviation'}


def motion_cost(eps):
    out = {}
    for u in ('blocking', 'fixed', 'anchored'):
        mine = [e['smoothness'] for e in eps if e['update'] == u and e['smoothness'] and e['delay'] != 'jitter']
        out[u] = {where: {k: round(float(np.median([m[where][k] for m in mine])), 4) for k in ('joint_step_rad', 'wrist_speed_mps')}
                  for where in ('update', 'boundary', 'other')} if mine else None
    static = {u: f"{sum(e['success'] for e in eps if e['update'] == u and e['speed'] == 0)}"
                 f"/{sum(e['update'] == u and e['speed'] == 0 for e in eps)}" for u in ('blocking', 'fixed', 'anchored')}
    return {'median_of_episode_maxima': out, 'static_success': static,
            'falls': {u: sum(e['fell'] for e in eps if e['update'] == u) for u in ('blocking', 'fixed', 'anchored')}}


def report(eps):
    fixed_eps = [e for e in eps if e['delay'] != 'jitter']
    by_update = defaultdict(Counter)
    for e in fixed_eps:
        if e['label']:
            by_update[e['update']][e['label']] += 1
    ages = defaultdict(list)
    for e in fixed_eps:
        r = e['rounds'][0] if e['rounds'] else {}
        if r.get('at') is not None:
            ages[(e['update'], e['delay'])].append(r['age_s'])
    return {'cells': cells(eps), 'judgment': judgment(fixed_eps), 'curve': curve(fixed_eps),
            'labels_by_update': {k: dict(v) for k, v in by_update.items()},
            'age_at_close_median_s': {f'{u} {d}': round(float(np.median(v)), 2) for (u, d), v in sorted(ages.items())},
            'motion': motion_cost(eps),
            'jitter': cells([e for e in eps if e['delay'] == 'jitter'])}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    x = sub.add_parser('extract')
    x.add_argument('out', type=Path)
    x.add_argument('patterns', nargs='+')
    r = sub.add_parser('report')
    r.add_argument('rows', type=Path)
    a = p.parse_args()
    if a.command == 'extract':
        rows = [episode(run, result) for run, result in finished(a.patterns)]
        a.out.write_text(json.dumps(rows))
        print(f'{len(rows)} episodes')
    else:
        print(json.dumps(report(json.loads(a.rows.read_text())), indent=1))
