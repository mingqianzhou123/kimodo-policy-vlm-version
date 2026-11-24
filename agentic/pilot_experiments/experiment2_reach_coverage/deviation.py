"""Is the gap between command and outcome systematic or random? (native-generation-20261125.md, last section)

  triplets  one row per finished run: the command, where the reference ended (root displacement and
            heading change in the start frame, and the ground its legs alone cover), where the robot
            ended; reach runs go to a second table (target, reference and measured wrist pose over
            the 2 s hold):
                deviation.py triplets --walks 'exp2dev-*-20261125-r1-*' --reaches 'exp2n-reach-*-20261125-r1-*' --out <name>
  analyze   the three compensations on held-out runs, the scatter between repeats and the
            systematic bias, per direction, with the verdicts written down before the data:
                deviation.py analyze --triplets <runs>/<name>/triplets.json [--markdown]

Tables are experiment output: they go to ``<runs>/<name>/``, not into git.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))  # agentic/scripts: paths.py
from paths import RUNS
from kimodo_vla.oracle import demo_walks as dw
from kimodo_vla.oracle.geometry import angle_deg, rotation_wxyz

FIT_SEEDS, TEST_SEEDS = (1000, 1001, 1002), (1003, 1004)
DIRECTION_DEG = {'fwd': 0, 'fl': 45, 'left': 90, 'bl': 135, 'back': 180, 'br': -135, 'right': -90, 'fr': -45}
LINE_CM, LINE_DEG, EXPLAINED, MODEL_GAIN = 10., 10., .8, .25   # the pre-registered lines


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def yaw_deg(rotation):
    return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))


def walk_row(run):
    target = json.loads((run/'target.json').read_text())
    goal, heading = target['motion_goal'], np.array(target['heading'])
    steps = rows(run/'execution.jsonl')
    first, last = steps[0]['before'], steps[-1]['after']
    to_start = lambda w: heading[:2, :2].T @ (np.asarray(w)[:2]-np.asarray(first['root_pos'])[:2])
    turn = lambda q: yaw_deg(heading.T @ rotation_wxyz(q))
    start_yaw = turn(first['root_quat'])
    ref_root = np.array([s['reference_root_world'] for s in steps])
    ref_feet = np.array([s['reference_foot_world'] for s in steps])
    rot = np.array([rotation_wxyz(s['after']['root_quat']) for s in steps])
    root = np.array([s['after']['root_pos'] for s in steps])
    tilt = np.degrees(np.arccos(np.clip(rot[:, 2, 2], -1, 1)))
    name = goal['name']
    direction, size = name.rsplit('-', 1)
    root_goal = goal['root']
    group = next((g for g in ('open', 'closed', 'comp') if f'-{g}-' in run.name), 'other')
    target = ([0., 0.] if direction == 'turn' else
              (int(size)/100*np.array([np.cos(np.radians(DIRECTION_DEG[direction])), np.sin(np.radians(DIRECTION_DEG[direction]))])).tolist())
    return {'run': run.name, 'group': group, 'target_xy': target,
            'seed': target.get('noise_seed'), 'command': name, 'direction': direction,
            'size': int(size)/(1 if direction == 'turn' else 100),
            'command_xy': root_goal['xy'], 'command_yaw_deg': root_goal.get('yaw_deg', 0.),
            'speed_m_s': root_goal.get('speed_m_s'), 'yaw_rate_deg_s': root_goal.get('yaw_rate_deg_s'),
            'reference_xy': to_start(ref_root[-1]).tolist(),
            'reference_yaw_deg': (turn(steps[-1]['reference_root_quat'])-start_yaw+180) % 360-180 if 'reference_root_quat' in steps[-1] else None,
            'reference_legs_xy': (heading[:2, :2].T @ dw.leg_displacement(ref_feet, ref_root)).tolist(),
            'actual_xy': to_start(last['root_pos']).tolist(),
            'actual_yaw_deg': float(np.degrees(np.unwrap(np.radians([turn(s['after']['root_quat']) for s in steps])))[-1]-start_yaw),
            'fell': bool((root[:, 2] < .4).any() or (tilt > 60).any()),
            'walk_timed_out': any(p.get('walk_timed_out') for p in target.get('phases', []))}


def reach_row(run):
    target = json.loads((run/'target.json').read_text())
    steps = rows(run/'execution.jsonl')
    commands = rows(run/'commands.jsonl')
    i = ['left', 'right'].index(target['active_sides'][0])
    arrival = target['arrival_time']
    hold = [s for s in steps if arrival < s['after']['time'] <= arrival+2.+1e-6]
    sent = {round(c['time'], 3): c for c in commands}
    if not hold:
        return None
    ref = np.array([sent[round(s['after']['time'], 3)]['sent_wrist'][i] for s in hold if round(s['after']['time'], 3) in sent])
    act = np.array([s['after']['wrist_pos'][i] for s in hold])
    act_rot = [np.array(s['after']['wrist_rot'][i]) for s in hold]
    goal_pos, goal_rot = np.array(target['targets_world'][i][0]), np.array(target['wrist_rotation_world'][i])
    return {'run': run.name, 'command': target['motion_goal']['name'], 'arm_ik': '-ik-' in run.name,
            'target_wrist': goal_pos.tolist(), 'target_wrist_rotation': goal_rot.tolist(),
            'reference_wrist': ref.mean(0).tolist() if len(ref) else None, 'measured_wrist': act.mean(0).tolist(),
            'measured_rotation_error_deg': float(np.mean([angle_deg(r, goal_rot) for r in act_rot]))}


def triplets(args):
    walks = [walk_row(r) for p in args.walks for r in sorted(RUNS.glob(p)) if (r/'run.json').exists()
             and json.loads((r/'run.json').read_text()).get('status') == 'completed']
    reaches = [x for p in args.reaches for r in sorted(RUNS.glob(p)) if (r/'run.json').exists()
               and json.loads((r/'run.json').read_text()).get('status') == 'completed' for x in [reach_row(r)] if x]
    out = RUNS/args.out
    out.mkdir(parents=True, exist_ok=True)
    (out/'triplets.json').write_text(json.dumps({'walks': walks, 'reaches': reaches}, indent=1))
    print(len(walks), 'walks,', len(reaches), 'reaches ->', out/'triplets.json')


def features(c):
    c = np.atleast_2d(c)
    d = np.linalg.norm(c, axis=1, keepdims=True)
    return np.hstack([c, d, c*d, np.ones_like(d)])


def fit_model(c, a, lam=1e-3):
    X = features(c)
    return np.linalg.solve(X.T @ X+lam*np.eye(X.shape[1]), X.T @ a)


def analyze(args):
    data = json.loads(Path(args.triplets).read_text())
    walks = [w for w in data['walks'] if not w['fell']]
    fell = [w['run'] for w in data['walks'] if w['fell']]
    opened = [w for w in walks if w['group'] == 'open']
    closed = [w for w in walks if w['group'] == 'closed']
    comp = [w for w in walks if w['group'] == 'comp']
    directions = list(dict.fromkeys(w['direction'] for w in opened))
    out = {'fell': fell, 'directions': {}}

    def vec(w, key):
        return np.array([w[key+'_yaw_deg']]) if w['direction'] == 'turn' else np.array(w[key+'_xy'])

    def coef(ws):
        c, a = np.array([vec(w, 'command') for w in ws]), np.array([vec(w, 'actual') for w in ws])
        return float((a*c).sum()/(c*c).sum())

    walk_fit = [w for w in opened if w['direction'] != 'turn' and w['seed'] in FIT_SEEDS]
    W = fit_model(np.array([w['command_xy'] for w in walk_fit]), np.array([w['actual_xy'] for w in walk_fit])) if walk_fit else None
    turn_fit = [w for w in opened if w['direction'] == 'turn' and w['seed'] in FIT_SEEDS]
    turn_ab = np.polyfit([w['command_yaw_deg'] for w in turn_fit], [w['actual_yaw_deg'] for w in turn_fit], 1) if len(turn_fit) > 1 else None

    def model_predict(w, model=None, ab=None):
        if w['direction'] == 'turn':
            ab = turn_ab if ab is None else ab
            return np.array([np.polyval(ab, w['command_yaw_deg'])])
        return features(np.array(w['command_xy'])) @ (W if model is None else model)

    unit = lambda d: 'deg' if d == 'turn' else 'cm'
    scale = lambda d: 1. if d == 'turn' else 100.
    for d in directions:
        ws = [w for w in opened if w['direction'] == d]
        fit = [w for w in ws if w['seed'] in FIT_SEEDS]
        test = [w for w in ws if w['seed'] in TEST_SEEDS]
        if not fit or not test:
            continue
        s = scale(d)
        k = coef(fit)
        raw = np.array([np.linalg.norm(vec(w, 'actual')-vec(w, 'command')) for w in test])*s
        resid_k = np.array([np.linalg.norm(vec(w, 'actual')-k*vec(w, 'command')) for w in test])*s
        resid_m = np.array([np.linalg.norm(vec(w, 'actual')-model_predict(w)) for w in test])*s
        explained = 1-float((resid_k**2).mean()/(raw**2).mean()) if (raw**2).mean() > 0 else None
        # scatter between repeats and systematic bias, per command (all seeds)
        scatter, bias = [], []
        for command in dict.fromkeys(w['command'] for w in ws):
            group = np.array([vec(w, 'actual') for w in ws if w['command'] == command])
            target = vec(next(w for w in ws if w['command'] == command), 'command')
            mean = group.mean(0)
            scatter.append(float(np.sqrt(((group-mean)**2).sum(1).mean()))*s)
            bias.append(float(np.linalg.norm(mean-target))*s)
        # leave one size out: fit on the other sizes' fit seeds, report on this size's test seeds
        loo_k, loo_m = [], []
        for size in dict.fromkeys(w['size'] for w in ws):
            fit_o = [w for w in fit if w['size'] != size]
            test_o = [w for w in test if w['size'] == size]
            if not fit_o or not test_o:
                continue
            k_o = coef(fit_o)
            loo_k += [np.linalg.norm(vec(w, 'actual')-k_o*vec(w, 'command'))*s for w in test_o]
            if d == 'turn':
                others = [w for w in turn_fit if w['size'] != size]
                ab = np.polyfit([w['command_yaw_deg'] for w in others], [w['actual_yaw_deg'] for w in others], 1)
                loo_m += [np.linalg.norm(vec(w, 'actual')-model_predict(w, ab=ab))*s for w in test_o]
            else:
                others = [w for w in walk_fit if w['size'] != size]
                M = fit_model(np.array([w['command_xy'] for w in others]), np.array([w['actual_xy'] for w in others]))
                loo_m += [np.linalg.norm(vec(w, 'actual')-model_predict(w, model=M))*s for w in test_o]
        closed_d = [w for w in closed if w['direction'] == d and w['seed'] in TEST_SEEDS]
        closed_err = np.array([np.linalg.norm(vec(w, 'actual')-vec(w, 'command')) for w in closed_d])*s
        line = LINE_DEG if d == 'turn' else LINE_CM
        row = {'unit': unit(d), 'coefficient': k, 'runs_fit': len(fit), 'runs_test': len(test),
               'uncompensated': float(raw.mean()), 'coefficient_error': float(resid_k.mean()),
               'coefficient_explained': explained, 'model_error': float(resid_m.mean()),
               'coefficient_error_loo_size': float(np.mean(loo_k)) if loo_k else None,
               'model_error_loo_size': float(np.mean(loo_m)) if loo_m else None,
               'closed_loop_error': float(closed_err.mean()) if len(closed_err) else None, 'closed_runs': len(closed_err),
               # the coefficient compensation actually run (commands scaled by 1/k), error to the intended target
               'compensated_measured': float(np.mean([np.linalg.norm(np.subtract(w['actual_xy'], w['target_xy'])) for w in comp
                                                      if w['direction'] == d])*s) if any(w['direction'] == d for w in comp) else None,
               'compensated_estimate_same_commands': float(np.mean([np.linalg.norm(vec(w, 'actual')-k*vec(w, 'command'))*s for w in test
                                                                    if any(c['command'] == w['command'] for c in comp)])) if comp and d != 'turn' else None,
               'scatter': float(np.mean(scatter)), 'bias': float(np.mean(bias))}
        # the pre-registered verdict, in the pre-registered order
        row['calibration_passes'] = bool(row['coefficient_error'] <= line or (explained is not None and explained >= EXPLAINED))
        row['model_passes'] = bool(row['model_error'] <= (1-MODEL_GAIN)*row['coefficient_error']
                                   and (row['closed_loop_error'] is None or row['model_error'] <= row['closed_loop_error']))
        row['mostly_random'] = bool(row['scatter'] > row['bias'])
        row['verdict'] = ('mostly random: only closed loop' if row['mostly_random'] else
                          'calibration is enough' if row['calibration_passes'] else
                          'grounds for a trained module' if row['model_passes'] else 'closed loop only')
        out['directions'][d] = row
    reaches = data.get('reaches', [])
    if reaches:
        def offsets(ik):
            r = [x for x in reaches if x['arm_ik'] == ik and x['reference_wrist']]
            return np.array([np.subtract(x['measured_wrist'], x['reference_wrist']) for x in r])
        out['reach'] = {}
        for ik in (False, True):
            o = offsets(ik)
            if len(o):
                mean = o.mean(0)
                out['reach']['ik' if ik else 'kimodo'] = {
                    'runs': len(o), 'mean_offset_cm': (100*mean).tolist(), 'mean_offset_norm_cm': 100*float(np.linalg.norm(mean)),
                    'scatter_cm': 100*float(np.sqrt(((o-mean)**2).sum(1).mean())),
                    'mean_distance_cm': 100*float(np.linalg.norm(o, axis=1).mean())}
    path = Path(args.triplets).with_name('analysis.json')
    path.write_text(json.dumps(out, indent=1))
    print(f"{'direction':<10}{'k':>6}{'raw':>8}{'coef':>8}{'expl':>6}{'model':>8}{'k-loo':>8}{'m-loo':>8}{'closed':>8}{'scatter':>8}{'bias':>8}{'comp':>8}{'c-est':>8}  verdict")
    for d, r in out['directions'].items():
        f = lambda x: '-' if x is None else f'{x:.1f}'
        print(f"{d:<10}{r['coefficient']:>6.2f}{f(r['uncompensated']):>8}{f(r['coefficient_error']):>8}"
              f"{'-' if r['coefficient_explained'] is None else format(r['coefficient_explained'], '.2f'):>6}"
              f"{f(r['model_error']):>8}{f(r['coefficient_error_loo_size']):>8}{f(r['model_error_loo_size']):>8}"
              f"{f(r['closed_loop_error']):>8}{f(r['scatter']):>8}{f(r['bias']):>8}{f(r['compensated_measured']):>8}"
              f"{f(r['compensated_estimate_same_commands']):>8}  {r['verdict']} ({r['unit']})")
    if fell:
        print('fell (left out):', fell)
    print(json.dumps(out.get('reach', {})))
    print(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('triplets')
    t.add_argument('--walks', nargs='+', default=[])
    t.add_argument('--reaches', nargs='+', default=[])
    t.add_argument('--out', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('--triplets', required=True)
    args = p.parse_args()
    {'triplets': triplets, 'analyze': analyze}[args.cmd](args)


if __name__ == '__main__':
    main()
