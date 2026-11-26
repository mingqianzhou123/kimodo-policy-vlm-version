"""Experiment 2's standard command test on HEFT (acceptance of the HEFT port, plan section 1.4), and the HITCH paper's
error definitions applied to HEFT and SONIC runs alike.

    python heft_commands.py run --gpus 0 0 [--only forward ...]        # HEFT, empty-handed, seeds 1000-1004
    python heft_commands.py analyze --date <heft date> --sonic-date <exp2 SONIC date>

``run`` is ``exp2_commands.py run --condition empty`` unchanged (same goals, seeds, start offsets, planner flags,
episode lengths), plus ``--tracker heft``; runs are named ``p2e2heft-cmd-empty-<motion>-<date>-s<seed>`` so that
experiment 2's own SONIC analysis (``p2e2-cmd-*``) never picks them up.

``analyze`` measures each run twice:
- experiment 2's protocol (``exp2_commands.analyze_run``: actual minus commanded at the stop, start-heading frame);
- HITCH appendix D: walk error = |executed - reference pelvis displacement| / reference distance over the command
  (walks with a reference distance >= 0.2 m); hand error = distance of each wrist from its reference wrist, both in
  their own pelvis frame, averaged over the frames where the reference wrist is >= 5 cm from its start in the
  reference torso frame. The reference is the frame the tracker followed at that step: the newest decoded frame for
  SONIC, ``heft_reference`` (8 frames older) for HEFT.
Writes ``<runs>/p2e2heft-cmd-empty-<date>.json`` and prints both tables, HEFT beside SONIC.

Diagnostic (not the protocol): ``--variant gain1`` appends ``--approach-gain 1`` after the frozen planner flags
(the launcher keeps the last value) and names the runs ``p2e2heft-gain1-cmd-...``; ``analyze --variant gain1``
prints those runs per seed with the signed along-walk error, beside the protocol (gain 2) runs of ``--gain2-date``.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import exp2_commands as E  # noqa: E402  (also puts agentic/scripts on the path)

PREFIX = 'p2e2heft'
VARIANTS = {'gain1': ['--approach-gain', '1']}   # diagnostics only; the protocol passes --approach-gain 2
WALK_MIN_M, MOVED_M = .2, .05
ACCEPT_WALK, ACCEPT_HAND_M = .15, .02   # plan section 1.4: walk error above 15% or hand error above 2 cm fails


def heft_run_name(condition, motion, date, seed, variant=None):
    return f"{PREFIX}{'-'+variant if variant else ''}-cmd-{condition}-{motion}-{date}-s{seed}"


def hitch_errors(run, fk, upto=None, after_time=None):
    """HITCH appendix D errors of one empty-handed run, over the whole command (up to sample ``upto``; from
    ``after_time`` on, if given), plus the body quantities of the command (yaw, pelvis height, torso pitch change)."""
    steps = [json.loads(x) for x in (run/'execution.jsonl').read_text().splitlines() if x.strip()]
    steps = [s for s in steps if 'reference_joints' in s][:None if upto is None else upto+1]
    if after_time is not None:
        steps = [s for s in steps if s['after']['time'] >= after_time-1e-6]
    if not steps:
        return {}

    def reference(s):
        h = s.get('heft_reference')
        if h is not None:
            return np.array(h['body_pos']), E.rotation_wxyz(h['body_quat_w']), np.array(h['joint_pos'])
        return np.array(s['reference_root_world']), E.rotation_wxyz(s['reference_root_quat']), np.array(s['reference_joints'])

    names = ['pelvis', 'torso_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link']
    out = {}
    actual_root = np.array([s['after']['root_pos'] for s in steps])
    refs = [reference(s) for s in steps]
    ref_root = np.array([r[0] for r in refs])
    dr, da = ref_root[-1, :2]-ref_root[0, :2], actual_root[-1, :2]-actual_root[0, :2]
    out['reference_distance_m'] = float(np.linalg.norm(dr))
    if np.linalg.norm(dr) >= WALK_MIN_M:
        out['walk_error'] = float(np.linalg.norm(da-dr)/np.linalg.norm(dr))
    ref_bodies = [fk(q, p, R, body_names=names) for p, R, q in refs]
    act_root_rot = [E.rotation_wxyz(s['after']['root_quat']) for s in steps]
    act_torso = [fk(s['after']['joints'], s['after']['root_pos'], R, body_names=['torso_link'])[1][0]
                 for s, R in zip(steps, act_root_rot)]
    yaw = lambda R: np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    pitch = lambda R: np.degrees(np.arctan2(R[0, 2]*np.cos(np.radians(yaw(R)))+R[1, 2]*np.sin(np.radians(yaw(R))), R[2, 2]))
    wrap = lambda a: (a+180) % 360-180
    out.update(yaw_executed_deg=float(wrap(yaw(act_root_rot[-1])-yaw(act_root_rot[0]))),
               yaw_reference_deg=float(wrap(yaw(refs[-1][1])-yaw(refs[0][1]))),
               pelvis_drop_executed_m=float(actual_root[-1, 2]-actual_root[0, 2]),
               pelvis_drop_reference_m=float(ref_root[-1, 2]-ref_root[0, 2]),
               torso_pitch_executed_deg=float(pitch(act_torso[-1])-pitch(act_torso[0])),
               torso_pitch_reference_deg=float(pitch(ref_bodies[-1][1][1])-pitch(ref_bodies[0][1][1])),
               fell=bool((actual_root[:, 2] < E.FALL_HEIGHT_M).any()))
    for side, k in (('left', 2), ('right', 3)):
        torso = [b[1][1].T@(b[0][k]-b[0][1]) for b in ref_bodies]
        moved = np.array([np.linalg.norm(t-torso[0]) >= MOVED_M for t in torso])
        if not moved.any():
            continue
        ref_local = np.array([R.T@(b[0][k]-p) for (p, R, _), b in zip(refs, ref_bodies)])
        act_local = np.array([R.T@(np.array(s['after']['wrist_pos'][k-2])-np.array(s['after']['root_pos']))
                              for s, R in zip(steps, act_root_rot)])
        out[f'hand_error_{side}_m'] = float(np.linalg.norm(act_local-ref_local, axis=1)[moved].mean())
    return out


def analyze_hitchref(date, fk):
    """Arm B: HITCH-planned references replayed open loop (heft_segments.py), measured over the planned motion
    (from the segment window's start to the end of the episode, the last frame held)."""
    rows = []
    for run in sorted(_runs().glob(f'{PREFIX}-hitchref-*-{date}')):
        manifest = run/'run.json'
        if not manifest.exists() or json.loads(manifest.read_text()).get('status') != 'completed':
            print('NOT-COMPLETED', run.name)
            continue
        target = json.loads((run/'target.json').read_text())
        tracker, motion = run.name.split('-')[2:4]
        row = {'run': run.name, 'tracker': tracker, 'motion': motion,
               **hitch_errors(run, fk, after_time=target['window_time'][0])}
        rows.append(row)
    return rows


def analyze(prefix, condition, date, fk, tracker):
    rows = []
    for run in sorted(_runs().glob(f'{prefix}-cmd-{condition}-*-{date}-s*')):
        manifest = run/'run.json'
        if not manifest.exists() or json.loads(manifest.read_text()).get('status') != 'completed':
            print('NOT-COMPLETED', run.name)
            continue
        if json.loads(manifest.read_text()).get('tracker', 'sonic') != tracker:
            continue
        row = E.analyze_run(run, fk)
        row.update(hitch_errors(run, fk))
        rows.append(row)
    return rows


def _runs():
    from paths import RUNS
    return RUNS


def hitch_table(rows):
    table = {}
    for motion in E.MOTIONS:
        r = [x for x in rows if x['motion'] == motion and not x['fell']]
        if not r:
            continue
        entry = {'n': len(r)}
        for key in ('walk_error', 'hand_error_right_m', 'hand_error_left_m'):
            v = np.array([x[key] for x in r if key in x])
            if len(v):
                entry[key] = {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else None}
        table[motion] = entry
    return table


def verdict(table):
    walks = [e['walk_error']['mean'] for m, e in table.items() if 'walk_error' in e]
    hands = [e['hand_error_right_m']['mean'] for m, e in table.items() if m == 'reach' and 'hand_error_right_m' in e]
    return {'worst_walk_error': max(walks) if walks else None, 'reach_hand_error_m': hands[0] if hands else None,
            'walk_ok': bool(walks) and max(walks) <= ACCEPT_WALK, 'hand_ok': bool(hands) and hands[0] <= ACCEPT_HAND_M}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['run', 'analyze', 'hitchref'])
    p.add_argument('--date', default=time.strftime('%Y%m%d'))
    p.add_argument('--sonic-date', help='date tag of experiment 2\'s SONIC empty-handed runs')
    p.add_argument('--gpus', type=int, nargs='+', default=[])
    p.add_argument('--only', nargs='+', choices=list(E.MOTIONS))
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--dds-domain', type=int, default=60)
    p.add_argument('--variant', choices=list(VARIANTS), help='a labelled diagnostic, not the protocol')
    p.add_argument('--gain2-date', help='with analyze --variant: the protocol HEFT runs to print beside it')
    args, extra = p.parse_known_args()
    if args.action == 'run':
        if not args.gpus:
            raise SystemExit('--gpus is required')
        E.run_name = lambda c, m, d, s: heft_run_name(c, m, d, s, args.variant)
        args.condition = 'empty'
        return E.launch(args, extra+['--tracker', 'heft']+VARIANTS.get(args.variant, []))
    from paths import G1_XML
    from kimodo_vla.core.types import G1_JOINT_NAMES
    from kimodo_vla.oracle.geometry import XmlFK
    fk = XmlFK(G1_XML, G1_JOINT_NAMES)
    if args.action == 'analyze' and args.variant:
        out = {}
        for label, prefix, date in ((args.variant, f'{PREFIX}-{args.variant}', args.date), ('gain2', PREFIX, args.gain2_date)):
            rows = [r for r in analyze(prefix, 'empty', date, fk, 'heft') if not args.only or r['motion'] in args.only]
            out[label] = {'date': date, 'runs': rows, 'protocol_table': E.summarize(rows), 'hitch_table': hitch_table(rows)}
            print(f'\n## {label} ({date})\n| motion | seed | along cm (signed) | across cm | root cm | walk error % (paper) | stopped |')
            for r in sorted(rows, key=lambda r: (r['motion'], r['seed'])):
                print(f"| {r['motion']} | {r['seed']} | {100*r['root_along_m']:+.1f} | {100*r['root_across_m']:+.1f} | "
                      f"{100*r['root_m']:.1f} | {100*r.get('walk_error', float('nan')):.0f} | {r['stopped']} |")
            for m, e in out[label]['protocol_table'].items():
                w = out[label]['hitch_table'].get(m, {}).get('walk_error')
                print(f"mean {m}: along {100*e['root_along_m']['mean']:+.1f} ± {100*e['root_along_m']['sd']:.1f}, "
                      f"root {100*e['root_m']['mean']:.1f} ± {100*e['root_m']['sd']:.1f} cm, "
                      f"walk {100*w['mean']:.0f} ± {100*w['sd']:.0f}%, fell {e['fell']}")
        (_runs()/f'{PREFIX}-{args.variant}-cmd-empty-{args.date}.json').write_text(json.dumps(out, indent=1))
        return
    if args.action == 'hitchref':
        rows = analyze_hitchref(args.date, fk)
        keys = ('walk_error', 'hand_error_right_m', 'yaw_executed_deg', 'yaw_reference_deg', 'pelvis_drop_executed_m',
                'pelvis_drop_reference_m', 'torso_pitch_executed_deg', 'torso_pitch_reference_deg', 'fell')
        print('| tracker | motion | ' + ' | '.join(keys) + ' |')
        for r in rows:
            print(f"| {r['tracker']} | {r['motion']} | " + ' | '.join(
                (f'{r[k]:.3f}' if isinstance(r.get(k), float) else str(r.get(k, '—'))) for k in keys) + ' |')
        out = _runs()/f'{PREFIX}-hitchref-{args.date}.json'
        out.write_text(json.dumps(rows, indent=1))
        print(out)
        return
    result = {}
    for tracker, prefix, date in (('heft', PREFIX, args.date), ('sonic', 'p2e2', args.sonic_date)):
        if date is None:
            continue
        rows = analyze(prefix, 'empty', date, fk, tracker)
        for r in rows:
            print(tracker, json.dumps(r))
        protocol, hitch = E.summarize(rows), hitch_table(rows)
        result[tracker] = {'date': date, 'runs': rows, 'protocol_table': protocol, 'hitch_table': hitch,
                           'acceptance': verdict(hitch)}
        print(f'\n## {tracker} ({date}): experiment 2 protocol\n'+E.markdown(protocol, f'{tracker} empty'))
        print(f'\n## {tracker}: HITCH definitions\n| 指令 | n | 走路误差 % | 右手误差 cm | 左手误差 cm |\n|---|---:|---:|---:|---:|')
        for m, e in hitch.items():
            cell = lambda k, s: (f"{s*e[k]['mean']:.1f} ± {s*e[k]['sd']:.1f}" if e[k]['sd'] is not None
                                 else f"{s*e[k]['mean']:.1f}") if k in e else '—'
            print(f"| {m} | {e['n']} | {cell('walk_error', 100)} | {cell('hand_error_right_m', 100)} | "
                  f"{cell('hand_error_left_m', 100)} |")
        print('acceptance', json.dumps(result[tracker]['acceptance']))
    out = _runs()/f'{PREFIX}-cmd-empty-{args.date}.json'
    out.write_text(json.dumps(result, indent=1))
    print(out)


if __name__ == '__main__':
    main()
