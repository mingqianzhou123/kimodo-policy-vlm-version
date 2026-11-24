"""Summarize a set of blocking grasp runs: success by episode, attempts by label, contact and slip rates.

    python agentic/pilot_experiments/experiment1_blocking_grasp/report_grasp.py 'exp1-static-20261124-r1-*'
    python agentic/pilot_experiments/experiment1_blocking_grasp/report_grasp.py 'exp1-map-20261124-m1-*' --map      # success grid per map cell
    python agentic/pilot_experiments/experiment1_blocking_grasp/report_grasp.py 'exp1-move010-*' --rounds           # every round: reach and box motion

Reach notes (per round, from the simulator): clearance of the pelvis from the table edge when the
hands start to rise, how far the pelvis backed off while they went up and out, and how far each
wrist still was from its target when the hands closed or the gate gave up.
"""
import argparse
from collections import Counter

import numpy as np
import fnmatch
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))  # agentic/scripts: paths.py and the launcher
from paths import RUNS


def summarize(results):
    """``results``: one finished grasp-result per episode. Rates use the plan's denominators."""
    attempts = [a for r in results for a in r['attempts']]
    held = [a for a in attempts if a['held']]
    wins = [r for r in results if r['success']]
    share = lambda part, whole: None if not whole else part / whole
    return {'episodes': len(results), 'successes': len(wins), 'success_rate': share(len(wins), len(results)),
            'first_attempt_successes': sum(r['first_attempt_success'] for r in results),
            'terminations': dict(Counter(r['termination'] for r in results)),
            'attempts': len(attempts), 'attempt_labels': dict(Counter(a['label'] for a in attempts)),
            'contact_rate': share(sum(a['touched'] for a in attempts), len(attempts)),
            'slip_rate': share(sum(a['label'] == 'slipped' for a in held), len(held)),
            'attempts_to_success': sorted(r['attempts_to_success'] for r in wins),
            'completion_s': sorted(round(r['completion_s'], 2) for r in wins),
            'attempts_per_episode': sorted(r['attempt_count'] for r in results)}


def first_round(result):
    """One line of attribution for an episode's first round."""
    a = result['attempts'][0] if result['attempts'] else {'label': 'none'}
    n = a.get('notes', {})
    gap = n.get('wrist_gap_m')
    return {'label': a['label'], 'clearance_m': n.get('anchor_clearance_m'), 'backoff_m': n.get('backoff_m'),
            'wrist_gap_m': None if gap is None else max(gap), 'closed': 'close_time' in n}


def grid(results):
    """Success per map cell: {(extra, yaw): {(depth, lateral): (success, first round)}}."""
    cells = {}
    for r in results:
        lay = r.get('layout') or {}
        if 'depth_m' not in lay:
            continue
        key = (round(lay['stance_extra_m'], 3), round(lay['yaw_deg']))
        cells.setdefault(key, {})[(round(lay['depth_m'], 3), round(lay['lateral_m'], 3))] = (r['success'], first_round(r))
    return cells


def pick_region(results, config, rate=.9):
    """The region rule, fixed before the map was run (configs/grasp-loop.json, stand._region_rule).

    A cell stands for the square of half a grid step around its point. For each stance distance,
    take the rectangle of whole cells with the most cells whose episodes, both yaws pooled,
    succeed at least ``rate``; ties go to the higher rate, then the wider lateral span (the box
    moves sideways), then the nearer stance. Pulling every outer edge in by half a step leaves
    the span between the outermost measured points: nothing is extrapolated.
    """
    m = config['stand']['map']
    depths, laterals, yaws = sorted(m['depth_m']), sorted(m['lateral_m']), m['yaw_deg']
    key = lambda *v: tuple(round(float(x), 3) for x in v)
    best = None
    for extra in sorted(m['stance_extra_m']):
        outcome = {key(l['depth_m'], l['lateral_m'], l['yaw_deg']): r['success'] for r in results
                   if (l := r.get('layout') or {}).get('depth_m') is not None and abs(l['stance_extra_m'] - extra) < 1e-6}
        for i0 in range(len(depths)):
            for i1 in range(i0, len(depths)):
                for j0 in range(len(laterals)):
                    for j1 in range(j0, len(laterals)):
                        cells = [key(d, l, y) for d in depths[i0:i1 + 1] for l in laterals[j0:j1 + 1] for y in yaws]
                        if any(c not in outcome for c in cells):
                            continue
                        wins = sum(outcome[c] for c in cells)
                        if wins < rate * len(cells) - 1e-9:
                            continue
                        score = (len(cells), wins / len(cells), laterals[j1] - laterals[j0], -extra)
                        if best is None or score > best[0]:
                            best = (score, {'stance_extra_m': extra, 'region': {
                                'depth_m': [depths[i0], depths[i1]], 'lateral_m': [laterals[j0], laterals[j1]]},
                                'episodes': len(cells), 'successes': wins})
    return None if best is None else best[1]


BINS_CM = (0, 1, 2, 3, 5, 8, 12)


def displacement(results, bins=BINS_CM):
    """Every round that reached for the box, by how far the box was from its latched place when the
    open hands got there (closed on it, or the gate gave up): success rate per bin, lateral and total.

    The rounds come from all given runs together, so a long-delay condition contributes few of them.
    """
    rows = []
    for r in results:
        for a in r['attempts']:
            n = a.get('notes', {})
            # Runs before 2026-11-25 kept only the closing position; rounds that never closed then drop out.
            at = n.get('close_box_pos', n.get('reach_box_pos'))
            if at is None:
                continue
            latched = a['latched_box_pos']
            rows.append({'run': r['run'], 'round': a['attempt'], 'success': a['label'] == 'success', 'label': a['label'],
                         'lateral_cm': abs(at[1] - latched[1]) * 100,
                         'total_cm': float(np.hypot(at[0] - latched[0], at[1] - latched[1])) * 100})
    table = {}
    for key in ('lateral_cm', 'total_cm'):
        table[key] = []
        for low, high in zip(bins, list(bins[1:]) + [None]):
            inside = [x for x in rows if x[key] >= low and (high is None or x[key] < high)]
            table[key].append({'from_cm': low, 'to_cm': high, 'rounds': len(inside),
                               'successes': sum(x['success'] for x in inside),
                               'labels': dict(Counter(x['label'] for x in inside))})
    return rows, table


def motion(results):
    """Box motion across all rounds of moving-box episodes: what the latch missed by the time the hands closed."""
    closed = [a['notes'] for r in results for a in r['attempts'] if 'close_time' in a.get('notes', {})]
    released = [r for r in results if r.get('drive_released_at') is not None]
    mean = lambda xs: None if not xs else sum(xs) / len(xs)
    return {'rounds_closed': len(closed), 'latch_to_close_m_mean': mean([n['latch_to_close_m'] for n in closed]),
            'latch_to_close_m_max': max((n['latch_to_close_m'] for n in closed), default=None),
            'turned_rounds': sum(bool(n.get('turned')) for n in closed),
            'episodes_released': len(released),
            'released_round': dict(Counter(r['drive_released_round'] for r in released)),
            'release_s_after_start': sorted(round(r['drive_released_at'] - r['clock_start'], 2) for r in released)}


def load(pattern):
    results, unfinished = [], []
    for run in sorted(RUNS.iterdir()):
        if not fnmatch.fnmatch(run.name, pattern) or not run.is_dir():
            continue
        try:
            result = json.loads((run/'grasp-result.json').read_text())
        except (OSError, ValueError):
            result = {}
        if result.get('termination'):
            results.append({'run': run.name, **result})
        else:
            unfinished.append(run.name)
    return results, unfinished


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('pattern')
    p.add_argument('--json', action='store_true')
    p.add_argument('--map', action='store_true', help='Success grid per map cell and first-round attribution')
    p.add_argument('--rounds', action='store_true', help='Every round with its reach and box-motion notes')
    p.add_argument('--pick-region', action='store_true', help='Apply the region rule to a finished map and print it')
    p.add_argument('--displacement', action='store_true', help='Round success by how far the box moved from its latch')
    a = p.parse_args()
    results, unfinished = load(a.pattern)
    if a.displacement:
        rows, table = displacement(results)
        for key, bins in table.items():
            print(key)
            for b in bins:
                rate = '  -' if not b['rounds'] else f"{100 * b['successes'] / b['rounds']:3.0f}%"
                print(f"  {b['from_cm']:>3}-{b['to_cm'] or '':<3} cm  rounds {b['rounds']:3d}  success {rate}  {b['labels']}")
        raise SystemExit(0)
    if a.pick_region:
        from paths import AGENTIC
        print(json.dumps(pick_region(results, json.loads((AGENTIC/'configs/grasp-loop.json').read_text()))))
        raise SystemExit(0)
    summary = summarize(results)
    if any((r.get('layout') or {}).get('speed_mps') for r in results):
        summary['motion'] = motion(results)
    fmt = lambda v, k=100: '  -' if v is None else f'{v * k:3.0f}'
    if a.map:
        for (extra, yaw), cells in sorted(grid(results).items()):
            depths = sorted({d for d, _ in cells}); laterals = sorted({l for _, l in cells})
            print(f'\nstance +{extra * 100:.0f} cm, yaw {yaw} deg   rows: depth (cm), columns: lateral (cm, + = left)')
            print('      ' + ''.join(f'{l * 100:+6.0f}' for l in laterals))
            for d in depths:
                marks = ['   ok ' if cells.get((d, l), (None,))[0] else '   -- ' if (d, l) in cells else '    ? '
                         for l in laterals]
                print(f'{d * 100:5.0f} ' + ''.join(marks))
        print('\nrun / first round: label, clearance cm, pelvis back-off cm, worst wrist gap cm')
        for r in results:
            f = first_round(r)
            print(f"{r['run']:44s} {'OK  ' if r['success'] else 'FAIL'} {f['label']:16s} {fmt(f['clearance_m'])} "
                  f"{fmt(f['backoff_m'])} {fmt(f['wrist_gap_m'])}")
    if a.rounds:
        for r in results:
            for x in r['attempts']:
                n = x.get('notes', {})
                print(f"{r['run']:40s} r{x['attempt']} {x['label']:16s} back-off {fmt(n.get('backoff_m'))} gap "
                      f"{fmt(max(n['wrist_gap_m']) if 'wrist_gap_m' in n else None)} latch->close "
                      f"{fmt(n.get('latch_to_close_m'))} turned {n.get('turned', '-')}")
    if a.json:
        print(json.dumps({'summary': summary, 'unfinished': unfinished,
                          'episodes': [{k: r[k] for k in ('run', 'seed', 'success', 'termination', 'episode_steps',
                                                           'attempt_count', 'completion_s')} |
                                       {'labels': [x['label'] for x in r['attempts']]} for r in results]}, indent=1))
    elif not (a.map or a.rounds):
        for r in results:
            labels = ' / '.join(x['label'] for x in r['attempts'])
            print(f"{r['run']}  seed {r['seed']}  {r['termination']:8s} steps {r['episode_steps']:4d}  {labels}")
    if not a.json:
        print(json.dumps(summary, indent=1, ensure_ascii=False))
        if unfinished:
            print('no verdict:', ', '.join(unfinished))
