"""Where a walk's stop error comes from, from the logs of existing runs (no new runs).

    python heft_walk_split.py --out <file.json> [--runs-glob ...]

For every finished forward / backward command-test run (HEFT gain 2 and gain 1, SONIC of experiment 2) it splits the
stop error of ``exp2_commands.analyze_run`` (measured at the same stop sample) into

- command -> reference: where the reference the tracker followed ended, minus the commanded end point;
- reference -> robot: where the robot ended, minus that reference end;

each along the start heading (signed: + is further forward in the start frame) and across it. The reference is the
frame the tracker followed at that sample (``heft_reference`` for HEFT, 8 frames behind the newest; the newest
decoded frame for SONIC); at the stop both stand still, so the lag does not enter.

It also reads the planner's own logs: when the stop plan was made (``plans.jsonl``) and how far from the goal the
body was then, signed along the walk (negative = short of it, positive = past it), and when the body first came
within the goal's tolerance; and, for every walk plan, how much of the reference's displacement over that plan
(``commands.jsonl`` ``sent_root_world``) the body covered over the same steps (shifted by the tracker's lag).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import exp2_commands as E  # noqa: E402

GROUPS = (('heft gain 2', 'p2e2heft-cmd-empty-{m}-20261127-s*', 8),
          ('heft gain 1', 'p2e2heft-gain1-cmd-empty-{m}-20261127-s*', 8),
          ('sonic gain 2', 'p2e2-cmd-empty-{m}-20261126-s*', 0),
          ('heft open loop', 'p2e2heft-openloop-cmd-empty-{m}-20261127-s*', 8),
          ('heft no lag', 'p2e2heft-nolag-cmd-empty-{m}-20261127-s*', 0))
TOLERANCE_M = .05


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def split_run(run, lag):
    target = json.loads((run/'target.json').read_text())
    steps = rows(run/'execution.jsonl')
    motion = next(m for m in E.MOTIONS if target['motion_goal']['name'] == f'p2e2-{m}')
    measured = E.analyze_run(run)
    i = measured['measured_index']
    heading = np.array(target['heading'])[:2, :2]
    start = np.array(steps[0]['after']['root_pos'][:2])
    command = np.array(E.MOTIONS[motion][1][:2])            # forward, left in the start frame
    local = lambda p: heading.T @ (np.asarray(p)[:2]-start)

    def reference(s):
        h = s.get('heft_reference')
        return np.asarray(h['body_pos'] if h is not None else s['reference_root_world'])

    ref_end, robot_end = local(reference(steps[i])), local(steps[i]['after']['root_pos'])
    out = {'run': run.name, 'motion': motion, 'seed': measured['seed'],
           'stop_along_cm': 100*measured['root_along_m'], 'stop_across_cm': 100*measured['root_across_m'],
           'cmd_to_ref_along_cm': 100*(ref_end-command)[0], 'cmd_to_ref_across_cm': 100*(ref_end-command)[1],
           'ref_to_robot_along_cm': 100*(robot_end-ref_end)[0], 'ref_to_robot_across_cm': 100*(robot_end-ref_end)[1]}
    sign = np.sign(command[0])                               # walk direction along the start heading
    plans = rows(run/'plans.jsonl')
    t_steps = np.array([s['after']['time'] for s in steps])
    robot = np.array([local(s['after']['root_pos']) for s in steps])
    stop = next((p for p in plans if p['phase'] == 'stop'), None)
    if stop is not None:
        k = int(np.searchsorted(t_steps, stop['time']-1e-6))
        out['stop_plan_s'] = stop['time']-steps[0]['before']['time']
        out['robot_at_stop_plan_along_cm'] = 100*sign*(robot[min(k, len(robot)-1)][0]-command[0])
        out['remaining_at_stop_plan_cm'] = 100*stop['root_remaining_m']
        within = np.nonzero(np.linalg.norm(robot-command, axis=1) <= TOLERANCE_M)[0]
        out['first_within_tol_s'] = float(t_steps[within[0]]-steps[0]['before']['time']) if len(within) else None
        out['walk_timed_out'] = any(p.get('walk_timed_out') for p in target.get('phases', []) if p['phase'] == 'stop')
    commands = rows(run/'commands.jsonl')
    coverage = []
    for p in (p for p in plans if p['phase'] in ('walk', 'stop')):
        sent = [c for c in commands if c['plan'] == p['plan']]
        if len(sent) < 2:
            continue
        a, b = local(sent[0]['sent_root_world']), local(sent[-1]['sent_root_world'])
        ref_d = sign*(b-a)[0]
        k0 = int(np.searchsorted(t_steps, sent[0]['time']-1e-6))+lag
        k1 = int(np.searchsorted(t_steps, sent[-1]['time']-1e-6))+lag
        if k1 >= len(robot) or abs(ref_d) < .02:
            continue
        coverage.append({'plan': p['plan'], 'phase': p['phase'], 'ref_cm': 100*ref_d,
                         'robot_cm': 100*sign*(robot[k1]-robot[k0])[0]})
    walk = [c for c in coverage if c['phase'] == 'walk']
    out['walk_plans'] = len(walk)
    if walk:
        out['walk_coverage'] = float(sum(c['robot_cm'] for c in walk)/sum(c['ref_cm'] for c in walk))
    stops = [c for c in coverage if c['phase'] == 'stop']
    if stops:
        out['stop_plan_ref_cm'], out['stop_plan_robot_cm'] = stops[0]['ref_cm'], stops[0]['robot_cm']
    out['plans'] = coverage
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    from paths import RUNS
    result = {}
    for label, pattern, lag in GROUPS:
        for motion in ('forward', 'backward'):
            for run in sorted(RUNS.glob(pattern.format(m=motion))):
                manifest = run/'run.json'
                if not manifest.exists() or json.loads(manifest.read_text()).get('status') != 'completed':
                    continue
                r = split_run(run, lag)
                result.setdefault(label, []).append(r)
                f = lambda k, d=1: '—' if r.get(k) is None else (f'{r[k]:+.{d}f}' if isinstance(r[k], float) else str(r[k]))
                print(f"| {label} | {motion} | {r['seed']} | {f('stop_along_cm')} / {f('stop_across_cm')} | "
                      f"{f('cmd_to_ref_along_cm')} / {f('cmd_to_ref_across_cm')} | "
                      f"{f('ref_to_robot_along_cm')} / {f('ref_to_robot_across_cm')} | {f('robot_at_stop_plan_along_cm')} | "
                      f"{f('stop_plan_s')} | {f('first_within_tol_s')} | {f('walk_coverage', 2)} | "
                      f"{f('stop_plan_ref_cm')} / {f('stop_plan_robot_cm')} |", flush=True)
    a.out.write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
