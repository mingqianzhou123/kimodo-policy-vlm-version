"""Analyses of the wide setting (wide-range-20261125.md): the robot walks to the box every round.

    python analyze_wide.py static 'exp1-wide-static-20261125-r1-*'        # step one: gate, first-round attribution, crossed feet
    python analyze_wide.py dynamic 'exp1-wide-*-20261125-d1-*'           # step two: displacement curve, time split, labels, verdict
    python analyze_wide.py rounds 'dbg-exp1w-*'                           # one line per round

Every definition used here was written into the record before the acceptance and the moving
runs were looked at (section "事先写定的判定和分析口径").
"""
import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))  # agentic/scripts: paths.py
from paths import AGENTIC as ROOT, RUNS

CONFIG = json.loads((ROOT/'configs/grasp-loop.json').read_text())
EDGE = CONFIG['scene']['table_edge_x_m']
# The landing window, from the stand-still measurements: pelvis to the table edge over both stance
# rows of the workspace map (rule stance 17-22 cm, rule + 5 cm 21-27 cm after settling), and the box
# inside the validated lateral span (-10 to 0 cm; measured -9.3 to +0.1 cm in the region runs).
CLEARANCE_M = (.17, .27)
LATERAL_M = (-.105, .005)         # box centre to the robot's left (negative: to its right)
BACKOFF_M = .15                   # the pelvis backed away this far while the hands went out
# Out of reach when the hands should be there: the validated region plus half a map step.
REACH_LATERAL_M = (-.15, .05)
REACH_DEPTH_M = (.08, .24)
DISP_BINS_CM = (0, 3, 5, 8, 15, 30, 60, 1000)
CROSS_M = 0.                      # left ankle to the right of the right ankle (episode heading +x)


def load(pattern):
    """Run folders that ended with a verdict, with the records the analyses read."""
    out = []
    for run in sorted(Path(p) for p in glob.glob(str(RUNS/pattern))):
        try:
            result = json.loads((run/'grasp-result.json').read_text())
        except (OSError, ValueError):
            continue
        if not result.get('termination'):
            continue
        layout = json.loads((run/'layout.json').read_text()) if (run/'layout.json').is_file() else {}
        policy = [json.loads(x) for x in (run/'rounds.jsonl').read_text().splitlines()]
        steps = []
        with (run/'execution.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                b = row['before']
                steps.append({'time': b['time'], 'root': b['root_pos'], 'feet': b['foot_pos'], 'box': b['box_com'],
                              'box_quat': b['box_quat'], 'ref_feet': row.get('reference_foot_world'),
                              'phase': (row.get('grasp') or {}).get('phase'), 'round': (row.get('grasp') or {}).get('round')})
        walks = {}
        for folder in sorted(run.glob('walk-r*')):
            plans = [json.loads(x) for x in (folder/'plans.jsonl').read_text().splitlines()] if (folder/'plans.jsonl').is_file() else []
            walks[int(folder.name[6:])] = plans
        out.append({'name': run.name, 'seed': result.get('seed'), 'result': result, 'layout': layout,
                    'policy': policy, 'steps': steps, 'walks': walks})
    return out


def at(steps, time):
    """The step sample nearest to ``time``."""
    times = np.array([s['time'] for s in steps])
    return steps[int(np.argmin(np.abs(times - time)))]


def phase_times(policy, index):
    """First time each phase of round ``index`` was entered, from the policy's own log."""
    times = {}
    for row in policy:
        if row['round'] == index and row['kind'] in ('latch', 'phase'):
            times.setdefault('latch' if row['kind'] == 'latch' else row['phase'], row['time'])
    return times


def rounds(run):
    """One dict per round: phase times, the time split, where the box was, reach notes and labels."""
    res, steps, out = run['result'], run['steps'], []
    speed = float(run['layout'].get('speed_mps') or 0.)
    last = {}
    for row in run['policy']:
        last[row['round']] = row
    for a in res['attempts']:
        k, n = a['attempt'], a.get('notes', {})
        p, notes = phase_times(run['policy'], k), last.get(k, {})
        latch = p.get('latch', a['start_time'])
        wait_end = p.get('walk', p.get('raise'))
        anchor = p.get('raise')
        close = n.get('close_time')
        reach_end = close if close is not None else n.get('reach_time')
        latched = np.asarray(a['latched_box_pos'])
        r = {'run': run['name'], 'seed': run['seed'], 'round': k, 'label': a['label'], 'speed': speed,
             'delay_s': res.get('latch_delay_steps', 75) / 50, 'latch': latch, 'wait_end': wait_end,
             'walk_end': p.get('settle'), 'anchor': anchor, 'close': close, 'reach_end': reach_end,
             'walked': 'walk' in p, 'walk_plans': notes.get('walk_plans'), 'walk_timed_out': notes.get('walk_timed_out'),
             'stance_error_m': notes.get('stance_error_m'), 'clearance_m': n.get('anchor_clearance_m'),
             'lateral_m': n.get('anchor_lateral_m'), 'backoff_m': n.get('backoff_m'), 'wrist_gap_m': n.get('wrist_gap_m'),
             'turned': n.get('turned'), 'gate_fired': close is not None, 'drive_released_at': n.get('drive_released_at')}
        if wait_end is not None and anchor is not None and reach_end is not None:
            r.update(wait_s=wait_end - latch, walk_s=anchor - wait_end, reach_s=reach_end - anchor, age_s=reach_end - latch)
            marks = [latch, wait_end, anchor, reach_end]
            boxes = [np.asarray(at(steps, t)['box']) for t in marks]
            r.update(box_moved_wait_m=float(np.linalg.norm(boxes[1][:2] - boxes[0][:2])),
                     box_moved_walk_m=float(np.linalg.norm(boxes[2][:2] - boxes[1][:2])),
                     box_moved_reach_m=float(np.linalg.norm(boxes[3][:2] - boxes[2][:2])))
            end = boxes[3]
            r.update(disp_lateral_m=float(abs(end[1] - latched[1])), disp_m=float(np.linalg.norm(end[:2] - latched[:2])),
                     scripted_m=speed * (reach_end - latch))
            root = np.asarray(at(steps, reach_end)['root'])
            r.update(box_rel_lateral_m=float(end[1] - root[1]), box_depth_m=float(end[0] - EDGE))
        out.append(r)
    return out


def within_1450(run):
    """Succeeded before the old 1450-step (29 s) limit would have ended the episode."""
    r = run['result']
    return bool(r['success'] and r['completion_s'] + r['clock_start'] <= 1450 * .02 + 1e-6)


def landing_ok(r):
    return (r['clearance_m'] is not None and CLEARANCE_M[0] <= r['clearance_m'] <= CLEARANCE_M[1]
            and r['lateral_m'] is not None and LATERAL_M[0] <= r['lateral_m'] <= LATERAL_M[1])


def first_round_cause(r):
    """Why a first round failed: after the hands closed, the walk's landing, the pelvis backing off, other."""
    if r['label'] == 'success':
        return 'success'
    if r['gate_fired']:
        return 'after_close'
    if not landing_ok(r):
        return 'landing'
    if (r['backoff_m'] or 0.) >= BACKOFF_M:
        return 'backoff'
    return 'other'


def out_of_reach(r):
    """Walked and anchored, failed, and when the hands should have been beside the box it was not
    within the reachable span of where the robot stood."""
    if r['label'] == 'success' or not r['walked'] or 'box_rel_lateral_m' not in r:
        return False
    return not (REACH_LATERAL_M[0] <= r['box_rel_lateral_m'] <= REACH_LATERAL_M[1]
                and REACH_DEPTH_M[0] <= r['box_depth_m'] <= REACH_DEPTH_M[1])


def crossings(run):
    """Per walk: steps of its stop plan where the reference, and where the measured body, had the left
    ankle to the right of the right one (heading +x), and what the body did meanwhile."""
    out = []
    steps = run['steps']
    for k, plans in run['walks'].items():
        stop = next((p for p in plans if p['phase'] == 'stop'), None)
        if stop is None:
            continue
        t0, t1 = stop['time'], stop['time'] + stop['sent_steps'] / 50
        window = [s for s in steps if t0 <= s['time'] <= t1]
        ref = [s for s in window if s['ref_feet'] and s['ref_feet'][0][1] - s['ref_feet'][1][1] < CROSS_M]
        body = [s for s in window if s['feet'][0][1] - s['feet'][1][1] < CROSS_M]
        walk = [s for s in steps if s['round'] == k and s['phase'] == 'walk']
        ref_walk = [s for s in walk if s['ref_feet'] and s['ref_feet'][0][1] - s['ref_feet'][1][1] < CROSS_M and s['time'] < t0]
        slide = float(np.linalg.norm(np.subtract(window[-1]['root'][:2], window[0]['root'][:2]))) if window else None
        out.append({'run': run['name'], 'round': k, 'stop_ref_steps': len(ref), 'stop_body_steps': len(body),
                    'walk_ref_steps': len(ref_walk), 'stop_slide_m': slide,
                    'fell': min((s['root'][2] for s in window), default=1.) < .4})
    return out


def bins(values, edges_cm=DISP_BINS_CM):
    return np.digitize(np.asarray(values) * 100, edges_cm[1:-1])


def logistic(x, y):
    """Maximum-likelihood logistic regression (intercept added); returns coefficients and log-likelihood."""
    X = np.c_[np.ones(len(x)), np.asarray(x, dtype=float)]
    y = np.asarray(y, dtype=float)
    w = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p)
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-X @ w)), 1e-12, 1 - 1e-12)
    return w, float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def lr_test(firsts):
    """Success ~ lateral displacement, against + delay and speed levels (first rounds of moving
    conditions): displacement coefficient per cm with its Wald p, and the likelihood-ratio p."""
    from scipy.stats import chi2, norm
    y = np.array([x['label'] == 'success' for x in firsts], dtype=float)
    d = np.array([x['disp_lateral_m'] * 100 for x in firsts])
    w, ll = logistic(d, y)
    X = np.c_[np.ones(len(d)), d]
    p = 1 / (1 + np.exp(-X @ w))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    delays = sorted({x['delay_s'] for x in firsts})
    speeds = sorted({x['speed'] for x in firsts})
    extra = [[x['delay_s'] == v for v in delays[1:]] + [x['speed'] == v for v in speeds[1:]] for x in firsts]
    w2, ll2 = logistic(np.c_[d, np.array(extra, dtype=float)], y)
    df = len(delays) - 1 + len(speeds) - 1
    return {'per_cm': float(w[1]), 'p_displacement': float(2 * norm.sf(abs(w[1]) / np.sqrt(cov[1, 1]))),
            'p_conditions': float(chi2.sf(2 * (ll2 - ll), df)), 'n': len(y)}


def wrecked(run, rows):
    """Per round after the first: did the box move > 5 cm, turn > 20 degrees, topple or leave the wide
    range between this latch and the one before (from the drive's release when it was driven)?"""
    out, steps = [], run['steps']
    w = CONFIG['wide']
    released = run['result'].get('drive_released_at')
    for prev, cur in zip(rows, rows[1:]):
        t0 = prev['latch'] if released is None else max(prev['latch'], released)
        if released is None and float(run['layout'].get('speed_mps') or 0.) > 0:
            t0 = cur['latch']   # never touched: still driven, nothing to count
        a, b = at(steps, t0), at(steps, cur['latch'])
        yaw = lambda q: np.degrees(2 * np.arctan2(q[3], q[0]))
        turn = abs((yaw(b['box_quat']) - yaw(a['box_quat']) + 180) % 360 - 180)
        w_, x, y_, z = b['box_quat']
        up = 1 - 2 * (x * x + y_ * y_)
        depth = b['box'][0] - EDGE
        out.append({'moved_m': float(np.linalg.norm(np.subtract(b['box'][:2], a['box'][:2]))), 'turn_deg': float(turn),
                    'toppled': bool(up < np.cos(np.radians(30))),
                    'outside': not (w['segment_y_m'][0] <= b['box'][1] <= w['segment_y_m'][1]
                                    and w['depth_m'][0] <= depth <= w['depth_m'][1])})
    for o in out:
        o['wrecked'] = o['moved_m'] > .05 or o['turn_deg'] > 20 or o['toppled'] or o['outside']
    return out


def table(rows, keys):
    print('\t'.join(keys))
    for r in rows:
        print('\t'.join('' if r.get(k) is None else f'{r[k]:.3f}' if isinstance(r.get(k), float) else str(r[k]) for k in keys))


def static(pattern):
    runs = load(pattern)
    first = [rounds(r)[0] for r in runs]
    success = sum(r['result']['success'] for r in runs)
    print(f'episodes {len(runs)}  success {success}  first-round success {sum(f["label"] == "success" for f in first)}')
    print(f'success within 1450 steps (29 s): {sum(within_1450(r) for r in runs)}')
    causes = [first_round_cause(f) for f in first]
    print('first round:', {c: causes.count(c) for c in ('success', 'landing', 'backoff', 'after_close', 'other')})
    inside = [f for f in first if landing_ok(f)]
    print(f'landing inside the window: {len(inside)}, first-round success there {sum(f["label"] == "success" for f in inside)}; '
          f'outside: {len(first) - len(inside)}, success {sum(f["label"] == "success" for f in first if not landing_ok(f))}')
    table([dict(f, cause=c, episode=r['result']['termination']) for f, c, r in zip(first, causes, runs)],
          ['seed', 'episode', 'label', 'cause', 'clearance_m', 'lateral_m', 'stance_error_m', 'backoff_m', 'walk_plans',
           'walk_s', 'wait_s', 'reach_s'])
    cross = [c for r in runs for c in crossings(r)]
    print(f'walks {len(cross)}; stop plans whose reference crossed the feet {sum(c["stop_ref_steps"] > 0 for c in cross)}; '
          f'measured feet crossed {sum(c["stop_body_steps"] > 0 for c in cross)}; falls in a stop {sum(c["fell"] for c in cross)}')
    table([c for c in cross if c['stop_ref_steps'] or c['stop_body_steps']],
          ['run', 'round', 'stop_ref_steps', 'stop_body_steps', 'walk_ref_steps', 'stop_slide_m', 'fell'])
    walks = [f['walk_s'] for f in first if f.get('walk_s') is not None]
    print(f'first-round walk (wait end to anchor) median {np.median(walks):.2f} s, range {min(walks):.2f}-{max(walks):.2f} s')


def dynamic(pattern):
    runs = load(pattern)
    every = [x for r in runs for x in rounds(r)]
    by = {}
    for r in runs:
        key = (float(r['layout'].get('speed_mps') or 0.), r['result'].get('latch_delay_steps'))
        by.setdefault(key, []).append(r)
    print('condition\tepisodes\tsuccess\tfirst\tmedian_s')
    for key in sorted(by, key=lambda k: (k[1], k[0])):
        rs = by[key]
        done = [r['result']['completion_s'] for r in rs if r['result']['success']]
        print(f'{key}\t{len(rs)}\t{sum(r["result"]["success"] for r in rs)}\t'
              f'{sum(r["result"]["first_attempt_success"] for r in rs)}\t{np.median(done) if done else None}\t'
              f'within 1450 steps {sum(within_1450(r) for r in rs)}')
    moving = [x for x in every if x['speed'] > 0 and 'disp_m' in x]
    print('\nall rounds of moving conditions, by lateral displacement at close (or gate give-up)')
    for name, key in (('lateral', 'disp_lateral_m'), ('total', 'disp_m')):
        b = bins([x[key] for x in moving])
        for i in range(len(DISP_BINS_CM) - 1):
            sel = [x for x, j in zip(moving, b) if j == i]
            if sel:
                labels = {}
                for x in sel:
                    labels[x['label']] = labels.get(x['label'], 0) + 1
                print(f'{name} {DISP_BINS_CM[i]}-{DISP_BINS_CM[i+1]} cm\t{len(sel)}\t{sum(x["label"] == "success" for x in sel)}\t{labels}')
    firsts = [x for x in moving if x['round'] == 0]
    if firsts:
        d = np.array([x['disp_lateral_m'] for x in firsts])
        s = np.array([x['scripted_m'] for x in firsts])
        print(f'\nfirst rounds: corr(measured lateral, speed x latch-to-close) {np.corrcoef(d, s)[0, 1]:.3f}; '
              f'turned {sum(bool(x["turned"]) for x in firsts)} of {len(firsts)}')
        print('time split (first rounds, medians): wait / walk / reach+gate s; box moved in each, m')
        for key in sorted(by, key=lambda k: (k[1], k[0])):
            sel = [x for x in firsts if (x['speed'], int(round(x['delay_s'] * 50))) == key]
            if sel:
                med = lambda k: float(np.median([x[k] for x in sel]))
                print(f'{key}\t{med("wait_s"):.2f} / {med("walk_s"):.2f} / {med("reach_s"):.2f}\t'
                      f'{med("box_moved_wait_m"):.3f} / {med("box_moved_walk_m"):.3f} / {med("box_moved_reach_m"):.3f}')
        print('logistic (first rounds of moving conditions):', lr_test(firsts))
    print('\nverdict per delay (plan text): static, 80% line, moving rates')
    for delay in sorted({k[1] for k in by}):
        static_ = by.get((0., delay), [])
        base = sum(r['result']['success'] for r in static_) / max(len(static_), 1)
        cells = []
        for speed in CONFIG['wide']['speeds_mps']:
            rs = by.get((speed, delay), [])
            rate = sum(r['result']['success'] for r in rs) / max(len(rs), 1)
            word = 'speed only' if rate >= .8 * base else 'drop >= 25 pp' if base - rate >= .25 else 'between'
            cells.append(f'{speed}: {rate:.3f} ({100 * (rate - base):+.1f} pp, {word})')
        print(f'delay {delay}: static {base:.3f}, 80% {.8 * base:.3f}; ' + '; '.join(cells))
    print('\nretries after a failed first round (moving and static):')
    for key in sorted(by, key=lambda k: (k[1], k[0])):
        failed = intact_n = intact_ok = wrecked_n = wrecked_ok = 0
        for r in by[key]:
            rows = rounds(r)
            if rows[0]['label'] == 'success':
                continue
            failed += 1
            w = wrecked(r, rows)
            ok = r['result']['success']
            if w and w[0]['wrecked']:
                wrecked_n, wrecked_ok = wrecked_n + 1, wrecked_ok + ok
            else:
                intact_n, intact_ok = intact_n + 1, intact_ok + ok
        print(f'{key}: first failed {failed}; scene intact {intact_n} (rescued {intact_ok}); wrecked {wrecked_n} (rescued {wrecked_ok})')
    print('\nround labels incl. out of reach:')
    for x in every:
        x['out_of_reach'] = out_of_reach(x)
    table(every, ['run', 'round', 'label', 'out_of_reach', 'disp_lateral_m', 'disp_m', 'wait_s', 'walk_s', 'reach_s',
                  'box_rel_lateral_m', 'box_depth_m'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('what', choices=['static', 'dynamic', 'rounds'])
    p.add_argument('pattern')
    a = p.parse_args()
    if a.what == 'static':
        static(a.pattern)
    elif a.what == 'dynamic':
        dynamic(a.pattern)
    else:
        table([dict(x, cause=first_round_cause(x) if x['round'] == 0 else None) for r in load(a.pattern) for x in rounds(r)],
              ['run', 'round', 'label', 'cause', 'clearance_m', 'lateral_m', 'stance_error_m', 'backoff_m', 'walk_plans',
               'wait_s', 'walk_s', 'reach_s', 'disp_lateral_m'])
