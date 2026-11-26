"""Which earlier box-grasp episodes could the two-hand gate fix (2026-11-26) have changed?

Before the fix, ``SynchronizedGripperGate`` closed both hands on the tick the second hand fired without
re-checking the hand that had fired earlier. The fix changes an episode only where, on a close tick,
one hand was outside its tolerance. This reads every finished box-grasp run (``run.json`` experiment
GRASP-LOOP, not the hammer) and, for each round's close tick (the first step whose grasp row says the
hands are closed), checks both hands' measured state *before* that step against the round's close
event: position within the gate tolerance of the event target (``wrist_targets.pre`` in rounds.jsonl),
wrist orientation against the event rotation (recomputed with ``grasp_policy.waypoints`` from the
round's latched box pose and anchor root), wrist speed. Output: one line per close where a hand was
outside, and ``<runs>/p2e2-gate-audit.json``.

    python exp2_gate_audit.py [--pattern 'exp1*']
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'scripts'))


def angle(a, b):
    return float(np.arccos(np.clip((np.trace(np.asarray(a).T @ np.asarray(b))-1)/2, -1, 1)))


def closes(run):
    """(round, before-sample) for the first closed step of each round."""
    seen, out = set(), []
    with (run/'execution.jsonl').open() as f:
        for line in f:
            if '"hands_closed": true' not in line:
                continue
            row = json.loads(line)
            r = row['grasp']['round']
            if r not in seen:
                seen.add(r)
                out.append((r, row['before']))
    return out


def audit_run(run, config, waypoints):
    rounds = {}
    for line in (run/'rounds.jsonl').read_text().splitlines():
        row = json.loads(line)
        if 'wrist_targets' in row and 'anchor_root' in row:
            rounds[row['round']] = row
    tol = config['gate']
    found = []
    for r, before in closes(run):
        info = rounds.get(r)
        if info is None:
            continue
        w = waypoints(info['box_pos'], info['box_quat'], info['anchor_root'][:2], config['policy'])
        targets = np.asarray(info['wrist_targets']['pre'])
        hands = []
        for side in range(2):
            d = float(np.linalg.norm(np.asarray(before['wrist_pos'][side])-targets[side]))
            a = angle(w['rotation'][side], before['wrist_rot'][side])
            v = float(np.linalg.norm(before['wrist_velocity'][side]))
            inside = d <= tol['position_tolerance_m'] and a <= tol['angle_tolerance_rad'] and v <= tol['speed_tolerance_mps']
            hands.append({'distance_m': d, 'angle_rad': a, 'speed_mps': v, 'inside': inside})
        found.append({'round': r, 'time': before['time'], 'hands': hands, 'all_inside': all(h['inside'] for h in hands)})
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pattern', default='*')
    args = p.parse_args()
    from paths import AGENTIC, RUNS
    from kimodo_vla.oracle.grasp_policy import waypoints
    config = json.loads((AGENTIC/'configs/grasp-loop.json').read_text())
    report = {'runs': 0, 'closes': 0, 'affected': []}
    for run in sorted(RUNS.glob(args.pattern)):
        manifest = run/'run.json'
        if not manifest.is_file() or not (run/'execution.jsonl').is_file() or not (run/'rounds.jsonl').is_file():
            continue
        record = json.loads(manifest.read_text())
        if record.get('experiment') != 'GRASP-LOOP' or record.get('status') != 'completed':
            continue
        report['runs'] += 1
        result = json.loads((run/'grasp-result.json').read_text()) if (run/'grasp-result.json').is_file() else {}
        for close in audit_run(run, config, waypoints):
            report['closes'] += 1
            if not close['all_inside']:
                entry = {'run': run.name, 'seed': record.get('seed'), 'success': result.get('success'),
                         'attempts_to_success': result.get('attempts_to_success'), **close}
                report['affected'].append(entry)
                print(json.dumps(entry))
    # The bug's signature is one hand outside while the other (the one that fired on this tick) is inside.
    # Both outside cannot come from this gate as configured now (older debug configurations, or a close
    # the audit's reconstructed event does not describe), so those are listed apart.
    one = [a for a in report['affected'] if sum(not h['inside'] for h in a['hands']) == 1]
    report['one_hand_outside'] = [a['run'] for a in one]
    report['both_outside'] = [a['run'] for a in report['affected'] if a not in one]
    out = RUNS/'p2e2-gate-audit.json'
    out.write_text(json.dumps(report, indent=1))
    print(report['runs'], 'runs', report['closes'], 'closes', len(report['affected']), 'with a hand outside on the close tick:',
          len(one), 'one hand (the bug),', len(report['both_outside']), 'both')
    print(out)


if __name__ == '__main__':
    main()
