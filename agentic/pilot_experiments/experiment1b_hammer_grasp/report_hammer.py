"""Summarize hammer grasp runs: success by episode, labels by round, reach notes, map, region, bottlenecks.

    python agentic/pilot_experiments/experiment1b_hammer_grasp/report_hammer.py 'exp1b-region-20261126-r1-*'
    python agentic/pilot_experiments/experiment1b_hammer_grasp/report_hammer.py 'exp1b-map-20261125-m1-*' --map
    python agentic/pilot_experiments/experiment1b_hammer_grasp/report_hammer.py 'exp1b-map-20261125-m1-*' --pick-region
    python agentic/pilot_experiments/experiment1b_hammer_grasp/report_hammer.py 'dbg-exp1b-*' --rounds

Reach notes per round (from the simulator): the pelvis's clearance from the table edge when the
hand starts to rise, how far it backed off while the hand went up and out, and how far the right
wrist was from its grip pose when the hand closed or the gate gave up.
"""
import argparse
from collections import Counter
import fnmatch
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'scripts'))   # agentic/scripts: paths.py
sys.path.insert(0, str(HERE.parent/'experiment1_blocking_grasp'))
from paths import AGENTIC, RUNS
from report_grasp import summarize   # same denominators as the box

GATE_POSITION_M, GATE_ANGLE_RAD = .05, .25


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


def bottleneck(attempt, forward=(0., -1.)):
    """The first round's failure class, by the rule written in the README before the map was run."""
    label, n = attempt['label'], attempt.get('notes', {})
    if label == 'success':
        return 'success'
    closed = 'close_time' in n
    if not closed:
        gap = n.get('wrist_gap_m')
        vec = np.asarray(n.get('wrist_gap_vec_m', [0, 0, 0]), dtype=float)
        if gap is not None and gap > GATE_POSITION_M:
            along = abs(vec[:2] @ np.asarray(forward))
            if n.get('backoff_m', 0.) > .15 or along > .7 * gap:
                return 'out_of_reach'
            return 'placement'
        if n.get('wrist_angle_rad', 0.) > GATE_ANGLE_RAD:
            return 'orientation'
        return 'gate_other'
    if label == 'slipped' or label == 'held_at_cutoff':
        return 'slipped' if label == 'slipped' else 'held_at_cutoff'
    if label == 'knocked_over':
        return 'knocked'
    return 'closed_not_held'


def first(result):
    a = result['attempts'][0] if result['attempts'] else {'label': 'none', 'notes': {}}
    n = a.get('notes', {})
    return {'label': a['label'], 'class': bottleneck(a) if result['attempts'] else 'none',
            'clearance_m': n.get('anchor_clearance_m'), 'backoff_m': n.get('backoff_m'),
            'wrist_gap_m': n.get('wrist_gap_m'), 'angle_rad': n.get('wrist_angle_rad'), 'closed': 'close_time' in n}


def grid(results):
    """{(extra, yaw): {(depth, lateral): (success, first round)}}"""
    cells = {}
    for r in results:
        lay = r.get('layout') or {}
        if 'depth_m' not in lay or 'cell' not in (lay.get('spec') or {}):
            continue
        key = (round(lay['stance_extra_m'], 3), round(lay['yaw_deg']))
        cells.setdefault(key, {})[(round(lay['depth_m'], 3), round(lay['lateral_m'], 3))] = (r['success'], first(r))
    return cells


def pick_region(results, config, rate=.9):
    """The region rule written in the README before the map was run (the box rule, three yaws pooled).

    For each stance distance take the rectangle of whole grid cells (depth x lateral) with the most
    cells whose episodes, all yaws pooled, succeed at least ``rate``; ties: higher rate, wider lateral
    span, nearer stance. The region spans the outermost measured points (no extrapolation).
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
                            best = (score, {'stance_extra_m': extra, 'depth_m': [depths[i0], depths[i1]],
                                            'lateral_m': [laterals[j0], laterals[j1]],
                                            'yaw_deg': [min(yaws), max(yaws)], 'episodes': len(cells), 'successes': wins})
    return None if best is None else best[1]


def plot_map(results, config, path):
    """One panel per stance and handle yaw: depth against lateral, coloured by outcome."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from arena.hammer_layout import placeable
    m = config['stand']['map']
    cells = grid(results)
    fig, axes = plt.subplots(len(m['stance_extra_m']), len(m['yaw_deg']), figsize=(3.2 * len(m['yaw_deg']), 3.2 * len(m['stance_extra_m'])),
                             squeeze=False, sharex=True, sharey=True)
    colour = {'first': '#2e7d32', 'later': '#a5d6a7', 'fail': '#e57373', 'none': '#bdbdbd', 'missing': '#ffffff'}
    for i, extra in enumerate(m['stance_extra_m']):
        for j, yaw in enumerate(m['yaw_deg']):
            ax, got = axes[i][j], cells.get((round(extra, 3), round(yaw)), {})
            for d in m['depth_m']:
                for l in m['lateral_m']:
                    key = (round(d, 3), round(l, 3))
                    if not placeable(dict(depth=d, lateral=l, yaw=yaw, extra=extra), config):
                        state, text = 'none', 'n/a'
                    elif key not in got:
                        state, text = 'missing', '?'
                    else:
                        ok, f = got[key]
                        state = 'first' if f['label'] == 'success' else 'later' if ok else 'fail'
                        text = {'first': '1st', 'later': 'retry'}.get(state, f['class'].replace('_', ' ')[:11])
                    ax.add_patch(plt.Rectangle((l * 100 - 5, d * 100 - 2.5), 10, 5, color=colour[state], ec='white'))
                    ax.text(l * 100, d * 100, text, ha='center', va='center', fontsize=7)
            ax.set_xlim(min(m['lateral_m']) * 100 - 5, max(m['lateral_m']) * 100 + 5)
            ax.set_ylim(max(m['depth_m']) * 100 + 2.5, min(m['depth_m']) * 100 - 2.5)
            ax.set_title(f'stance +{extra * 100:.0f} cm, handle {yaw:+d} deg', fontsize=9)
            if i == len(m['stance_extra_m']) - 1:
                ax.set_xlabel('grip point lateral, cm (- = robot right)')
            if j == 0:
                ax.set_ylabel('grip point behind edge, cm')
    fig.suptitle('Hammer reach map. Dark green: first round; light green: later round; red: failed (first-round class);\n'
                 'grey: the hammer would hang over the table edge, not run', fontsize=9)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('pattern')
    p.add_argument('--json', action='store_true')
    p.add_argument('--map', action='store_true', help='Success grid per map cell and first-round attribution')
    p.add_argument('--rounds', action='store_true', help='Every round with its reach and hammer-motion notes')
    p.add_argument('--pick-region', action='store_true', help='Apply the region rule to a finished map and print it')
    p.add_argument('--svg', type=Path, help='With --map: also draw the map to this file')
    a = p.parse_args()
    results, unfinished = load(a.pattern)
    fmt = lambda v, k=100: '  -' if v is None else f'{v * k:4.0f}'
    if a.pick_region:
        print(json.dumps(pick_region(results, json.loads((AGENTIC/'configs/hammer-grasp.json').read_text()))))
        raise SystemExit(0)
    summary = summarize(results)
    summary['first_round_classes'] = dict(Counter(first(r)['class'] for r in results))
    if a.map and a.svg:
        sys.path.insert(0, str(AGENTIC))
        plot_map(results, json.loads((AGENTIC/'configs/hammer-grasp.json').read_text()), a.svg)
    if a.map:
        for (extra, yaw), cells in sorted(grid(results).items()):
            depths = sorted({d for d, _ in cells}); laterals = sorted({l for _, l in cells})
            print(f'\nstance +{extra * 100:.0f} cm, yaw {yaw:+d} deg   rows: depth (cm), columns: lateral (cm, + = left)')
            print('      ' + ''.join(f'{l * 100:+6.0f}' for l in laterals))
            for d in depths:
                marks = ['   ok ' if cells.get((d, l), (None,))[0] else '   -- ' if (d, l) in cells else '    ? '
                         for l in laterals]
                print(f'{d * 100:5.0f} ' + ''.join(marks))
    if a.map or not (a.rounds or a.json):
        print('\nrun / first round: label, class, clearance cm, back-off cm, wrist gap cm, angle deg')
        for r in results:
            f = first(r)
            print(f"{r['run']:46s} {'OK  ' if r['success'] else 'FAIL'} {f['label']:16s} {f['class']:15s} "
                  f"{fmt(f['clearance_m'])} {fmt(f['backoff_m'])} {fmt(f['wrist_gap_m'])} {fmt(f['angle_rad'], 57.3)}")
    if a.rounds:
        for r in results:
            for x in r['attempts']:
                n = x.get('notes', {})
                print(f"{r['run']:40s} r{x['attempt']} {x['label']:16s} back-off {fmt(n.get('backoff_m'))} gap "
                      f"{fmt(n.get('wrist_gap_m'))} angle {fmt(n.get('wrist_angle_rad'), 57.3)} latch->close "
                      f"{fmt(n.get('latch_to_close_m'))} along {fmt(n.get('latch_to_close_along_m'))} across "
                      f"{fmt(n.get('latch_to_close_across_m'))} turn {fmt(n.get('latch_to_close_turn_deg'), 1)}")
    if a.json:
        print(json.dumps({'summary': summary, 'unfinished': unfinished}, indent=1))
    else:
        print(json.dumps(summary, indent=1, ensure_ascii=False))
        if unfinished:
            print('no verdict:', ', '.join(unfinished))
