"""Who is responsible for the poor walking: how we use Kimodo, Kimodo itself, or SONIC?

Test one gives SONIC walking references that did not come from Kimodo (segments of the recorded
Arena SONIC demonstrations) and measures how much of each it walks. Test two
(``rollout_kimodo.py --sweep``) measures the gait Kimodo generates under different uses, next to
the demonstrations' gait measured here. Subcommands:

  mine   find directional walking windows in the demonstrations, pick segments, write one
         reference file per segment and the demonstrations' gait numbers:
             walk_attribution.py mine --data <HumanoidArena_dataset_v3_1 copy> --out <segments dir>
         or the same from the LAFAN1 motion capture retargeted to G1 (retargeted against foot slip):
             walk_attribution.py mine --lafan <LAFAN1_Retargeting_Dataset copy> --out <segments dir>
  suite  replay every segment in the simulator from experiment two's open-floor start:
             walk_attribution.py suite --segments <dir> --tag <tag> --gpus 0 6 7 0 6 7
  score  score finished replays (completion, sideways and heading drift, falls, measured gait):
             walk_attribution.py score '<run prefix>*' [--markdown]
  cross  the cross-check: run named conditions of the test-two sweep config live in the
         simulator (Kimodo replanning from the measured body, SONIC executing), then score
         the first seconds of each walk as test two scores the generated one:
             walk_attribution.py cross --conditions base text --tag <tag> --gpus 0 6 7 \
                 --text-cache <dir> --lead-dirs lafan=<dir> arena=<dir>
             walk_attribution.py score-cross '<tag>-*'
  table  test two's table: each sweep condition's gait per direction next to the recorded
         references' (medians over seeds):
             walk_attribution.py table --sweep <run id> [<run id> ...] --baseline <segments dir> [...]

Segment files go to ``--out``; runs to the usual runs directory as ``<tag>-<segment>``.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from paths import AGENTIC as ROOT, RUNS
from kimodo_vla.oracle import demo_walks as dw

TASKS = ('HOI_double_desk', 'HOI_football', 'HOI_grap_cup', 'HOI_pp_box', 'HSI_boxing',
         'HSI_open_door', 'HSI_sit_sofa', 'HSI_vision_navi')
PER_DIRECTION = {'forward': 8, 'backward': 8, 'left': 5, 'right': 5, 'turn': 8}
LEAD_S = (3.4, 5.)     # lead-in before the window: at least Kimodo's history, started where the root is slowest
START_XY = (-4., -4.)  # experiment two's open-floor start


def episodes(data, lafan):
    """(task, episode index, actions) of every demonstration episode, or of every LAFAN1 G1 walk."""
    if lafan:
        for i, path in enumerate(sorted(Path(lafan).glob('g1/walk*.csv'))):
            yield 'lafan1-'+path.stem, i, dw.lafan1_actions(np.loadtxt(path, delimiter=','))
        return
    from kimodo_vla.benchmarks.arena import demos
    for task in TASKS:
        directory = demos.dataset_dir(Path(data), task, 'sonic')
        if not directory.is_dir():
            print(f'{task}: not in {data}, skipped', flush=True)
            continue
        for episode in demos.read_dataset(directory)[1]:
            yield task, episode.index, demos.read_episode(episode)[1]


def candidates(data, lafan=None):
    out = []
    for task, index, actions in episodes(data, lafan):
        for w in dw.find_windows(actions, lead_s=LEAD_S[0]):
            summary = dw.window_summary(actions, w['begin'], w['end'])
            if dw.classify(summary['displacement_m'], summary['yaw_deg'], summary['seconds']) != w['direction']:
                w['end'] = w['begin']+100      # the merged window curves: keep its first 2 s
                summary = dw.window_summary(actions, w['begin'], w['end'])
                if dw.classify(summary['displacement_m'], summary['yaw_deg'], 2.) != w['direction']:
                    continue
            d = np.array(summary['displacement_m'])
            purity = (abs(summary['yaw_deg']) if w['direction'] == 'turn'
                      else float(abs(d @ {'forward': [1, 0], 'backward': [-1, 0], 'left': [0, 1],
                                          'right': [0, -1]}[w['direction']]))/summary['distance_m'])
            out.append(dict(w, task=task, episode=index, summary=summary, purity=purity,
                            actions=actions))
    return out


def pick(found):
    """Per direction: purest and longest first, one window per episode, tasks taken in turn."""
    chosen = []
    for direction, n in PER_DIRECTION.items():
        pool = sorted([c for c in found if c['direction'] == direction],
                      key=lambda c: (-round(c['purity'], 2), -c['summary']['seconds']))
        by_task, used = {}, set()
        for c in pool:
            if (c['task'], c['episode']) not in used:
                by_task.setdefault(c['task'], []).append(c)
                used.add((c['task'], c['episode']))
        while len([c for c in chosen if c['direction'] == direction]) < n and any(by_task.values()):
            for task in sorted(by_task):
                if by_task[task] and len([c for c in chosen if c['direction'] == direction]) < n:
                    chosen.append(by_task[task].pop(0))
    return chosen


def lead_start(actions, begin):
    """The slowest frame (root ground speed) between LEAD_S[1] and LEAD_S[0] before the window."""
    root = dw.integrate_root(actions)
    lo, hi = max(0, begin-int(LEAD_S[1]*dw.RATE)), begin-int(LEAD_S[0]*dw.RATE)
    speed = dw.RATE*np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1)
    return int(lo+np.argmin(speed[lo:hi+1])), float(speed[lo:hi+1].min())


def mine(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    if bool(args.data) == bool(args.lafan):
        raise SystemExit('Give --data (Arena demonstrations) or --lafan (LAFAN1 G1 retargeting), not both')
    found = candidates(args.data, args.lafan)
    counts = {d: sum(c['direction'] == d for c in found) for d in dw.DIRECTIONS}
    print('candidate windows (one episode can give several):', counts, flush=True)
    fk = dw.g1_fk()
    rows = []
    for c in pick(found):
        a = c['actions']
        start, speed = lead_start(a, c['begin'])
        name = f"{c['direction']}-{c['task'].lower()}-e{c['episode']:03d}-f{c['begin']}"
        source = ({'dataset': 'lvhaidong/LAFAN1_Retargeting_Dataset (g1, resampled 30 -> 50 Hz)', 'file': c['task'][7:]}
                  if args.lafan else {'dataset': 'WilliamWang16/HumanoidArena_dataset_v3_1@ca04c2d5', 'task': c['task'],
                                      'tracker': 'sonic', 'episode': c['episode']})
        segment = {'name': name, 'direction': c['direction'], 'source': dict(source, frames=[start, c['end']]),
                   'window': [c['begin']-start, c['end']-start], 'lead_start_speed_m_s': speed,
                   'actions': a[start:c['end']+1].tolist()}
        (out/f'{name}.json').write_text(json.dumps(segment))
        feet = dw.reference_feet(a[c['begin']:c['end']+1], fk)
        rows.append({'name': name, 'direction': c['direction'], 'task': c['task'], 'episode': c['episode'],
                     'window': c['summary'], 'lead_s': (c['begin']-start)/dw.RATE, 'lead_start_speed_m_s': speed,
                     'gait': dw.stepping(feet)})
        print(name, f"{c['summary']['speed_m_s']:.2f} m/s, yaw {c['summary']['yaw_deg']:.0f} deg, "
                    f"slide {rows[-1]['gait']['slide_m_s']:.3f} m/s, both moving {rows[-1]['gait']['both_moving']:.2f}", flush=True)
    summary = {'candidates': counts, 'segments': rows, 'baseline': baseline(rows)}
    (out/'segments.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary['baseline'], indent=1))


GAIT_KEYS = ('slide_m_s', 'slide_p90_m_s', 'both_moving', 'step_length_m', 'cadence_hz')


def baseline(rows):
    """Median and range of each gait number, per direction, over the chosen windows."""
    out = {}
    for direction in dw.DIRECTIONS:
        group = [r for r in rows if r['direction'] == direction]
        if not group:
            continue
        out[direction] = {'windows': len(group),
                          'speed_m_s': float(np.median([r['window']['speed_m_s'] for r in group])),
                          'yaw_rate_deg_s': float(np.median([abs(r['window']['yaw_rate_deg_s']) for r in group]))}
        for key in GAIT_KEYS:
            values = [r['gait'][key] for r in group if r['gait'][key] is not None]
            out[direction][key] = [float(np.median(values)), float(min(values)), float(max(values))] if values else None
    return out


def suite(args):
    segments = sorted(p for p in Path(args.segments).glob('*.json') if p.name != 'segments.json')
    if args.only:
        segments = [p for p in segments if any(p.stem.startswith(o) for o in args.only)]
    queue, lock = list(segments), threading.Lock()

    def worker(gpu, slot):
        while True:
            with lock:
                if not queue:
                    return
                path = queue.pop(0)
            run_id = f'{args.tag}-{path.stem}'
            if (RUNS/run_id/'execution.jsonl').is_file() and json.loads((RUNS/run_id/'run.json').read_text()).get('status') == 'completed':
                continue
            segment = dw.load_segment(path)
            steps = int(round((dw.BLEND_S+dw.SETTLE_S)*dw.RATE))+len(segment['actions'])+50
            cmd = [sys.executable, str(ROOT/'scripts/run_oracle.py'), '--run-id', run_id, '--reference-source', 'demo',
                   '--demo-segment', str(path), '--start-xy', *map(str, START_XY), '--sim-gpu', str(gpu),
                   '--steps', str(steps)]
            if args.gpu_min_free_mib:
                cmd += ['--gpu-min-free-mib', str(args.gpu_min_free_mib)]
            began = time.time()
            env = dict(os.environ, KVLA_DDS_DOMAIN=str(args.dds_domain+slot))  # in-process DDS only
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            print(f'{run_id} on GPU {gpu}: exit {result.returncode} in {time.time()-began:.0f} s', flush=True)
            if result.returncode:
                print(result.stderr[-1500:], flush=True)
    threads = []
    for i, gpu in enumerate(args.gpus):
        threads.append(threading.Thread(target=worker, args=(gpu, i)))
        threads[-1].start()
        time.sleep(args.stagger)
    for t in threads:
        t.join()


def run_queue(jobs, gpus, stagger, dds_domain):
    """Run (run_id, command) jobs over the GPUs, one at a time per listed GPU; finished runs are skipped."""
    queue, lock = list(jobs), threading.Lock()

    def worker(gpu, slot):
        while True:
            with lock:
                if not queue:
                    return
                run_id, cmd = queue.pop(0)
            if (RUNS/run_id/'run.json').is_file() and json.loads((RUNS/run_id/'run.json').read_text()).get('status') == 'completed':
                continue
            cmd = [c.replace('{gpu}', str(gpu)) for c in cmd]
            began = time.time()
            env = dict(os.environ, KVLA_DDS_DOMAIN=str(dds_domain+slot))
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            print(f'{run_id} on GPU {gpu}: exit {result.returncode} in {time.time()-began:.0f} s', flush=True)
            if result.returncode:
                print(result.stderr[-1500:], flush=True)
    threads = []
    for i, gpu in enumerate(gpus):
        threads.append(threading.Thread(target=worker, args=(gpu, i)))
        threads[-1].start()
        time.sleep(stagger)
    for t in threads:
        t.join()


def cross(args):
    from rollout_kimodo import condition_goal, lead_through_window
    cfg = json.loads(Path(args.config).read_text())
    inputs = RUNS/f'{args.tag}-inputs'
    inputs.mkdir(exist_ok=True)
    jobs = []
    for name in args.conditions:
        use = {**cfg['base'], **cfg['conditions'][name]}
        for direction in args.directions or cfg['directions']:
            rate = use['turn_rate'] if direction == 'turn' else use['speed']
            speed = rate[direction] if isinstance(rate, dict) else rate
            for i, seed in enumerate(cfg['seeds'][:args.repeats]):
                run_id = f'{args.tag}-{name}-{direction}-s{seed}'
                goal = inputs/f'{name}-{direction}.json'
                goal.write_text(json.dumps(condition_goal(direction, speed, cfg)))
                cmd = [sys.executable, str(ROOT/'scripts/run_oracle.py'), '--run-id', run_id, '--motion-goal', str(goal),
                       '--start-xy', *map(str, START_XY), '--sim-gpu', '{gpu}', '--policy-gpu', '{gpu}',
                       '--share-policy-sim-gpu', '--gpu-min-free-mib', '16384', '--noise-seed', str(seed),
                       '--walk-steps', str(use['walk_steps']), '--constraint-cfg', str(use['constraint_cfg']),
                       '--root-constraints', use['root_constraints'], '--history-frames', str(use['history_frames'])]
                lead_s = 0.
                if use.get('text'):
                    cmd += ['--text', cfg['texts'][direction], '--text-cfg', str(use.get('text_cfg', 2.)), '--text-cache', args.text_cache]
                if use.get('lead'):
                    names = cfg['lead_segments'][use['lead']][direction]
                    lead = lead_through_window(Path(dict(args.lead_dirs)[use['lead']])/f'{names[i % len(names)]}.json',
                                               inputs/f'{name}-{direction}-s{seed}-lead.json')
                    cmd += ['--lead-segment', str(lead)]
                    lead_s = (len(json.loads(lead.read_text())['actions'])+50)/dw.RATE
                cmd += ['--steps', str(int(round((.7+lead_s+cfg['seconds']+.6)*dw.RATE)))]
                jobs.append((run_id, cmd))
    run_queue(jobs, args.gpus, args.stagger, args.dds_domain)


def score_measured(steps, target, seconds):
    """Progress, drift, falls and gait of a live walk over its first ``seconds`` (from the walk phase's start)."""
    walk = next((p for p in target['phases'] if p['phase'] == 'walk'), None)
    if walk is None:
        return None
    goal = target['motion_goal']['root']
    after = [s['after'] for s in steps]
    time_ = np.array([x['time'] for x in after])
    i0 = int(np.argmin(abs(time_-walk['time'])))
    i1 = i0+int(round(seconds*dw.RATE))
    if i1 >= len(after):
        return None
    root = np.array([x['root_pos'] for x in after])
    rotations = np.array([dw.rotation_wxyz(x['root_quat']) for x in after])
    tilt = np.degrees(np.arccos(np.clip(rotations[:i1+1, 2, 2], -1, 1)))
    d, yaw = dw.heading_frame_motion(root, rotations, i0, i1)
    out = {'fell': bool((root[:i1+1, 2] < .40).any() or (tilt > 60).any()), 'moved_m': d.tolist(), 'yaw_deg': yaw,
           'gait_actual': dw.stepping(np.array([x['foot_pos'] for x in after])[i0:i1+1])}
    if goal.get('yaw_deg'):
        out['progress'] = yaw/(goal['yaw_rate_deg_s']*seconds)
        out['drift_cm'] = 100*float(np.hypot(*d))
    else:
        along = np.asarray(goal['xy'])/np.linalg.norm(goal['xy'])
        out['progress'] = float(d @ along)/(goal['speed_m_s']*seconds)
        out['lateral_cm'] = 100*float(abs(along[0]*d[1]-along[1]*d[0]))
    out['heading_drift_deg'] = None if goal.get('yaw_deg') else yaw
    return out


def score_cross(args):
    rows = []
    for run in sorted(RUNS.glob(args.pattern)):
        if not (run/'run.json').is_file() or run.name.endswith('-inputs'):
            continue
        manifest = json.loads((run/'run.json').read_text())
        if manifest.get('status') != 'completed':
            continue
        steps = [json.loads(x) for x in (run/'execution.jsonl').read_text().splitlines() if x.strip()]
        target = json.loads((run/'target.json').read_text())
        result = score_measured(steps, target, args.seconds)
        result = {'run': run.name, 'direction': target['motion_goal']['name'], 'seed': manifest['noise_seed'],
                  'condition': {k: manifest.get(k) for k in ('text', 'root_constraints', 'history_frames', 'walk_steps',
                                                             'lead_segment')}, 'measured': result}
        (run/'cross-result.json').write_text(json.dumps(result, indent=1))
        rows.append(result)
    name = args.pattern.rstrip('*').rstrip('-')
    (RUNS/f'{name}-cross-summary.json').write_text(json.dumps(rows, indent=1))
    for r in rows:
        m = r['measured']
        print(r['run'], 'no walk window' if m is None else
              f"progress {m['progress']:.2f} fell {m['fell']} slide {m['gait_actual']['slide_m_s'] or 0:.3f} "
              f"both {m['gait_actual']['both_moving']:.2f}")


def sweep_table(args):
    rows = []
    for run_id in args.sweep:
        rows += json.loads((RUNS/run_id/'sweep.json').read_text())
    baselines = {Path(d).name: json.loads((Path(d)/'segments.json').read_text())['baseline'] for d in args.baseline}
    order = list(dict.fromkeys(r['condition'] for r in rows))
    table = {}
    for name in order:
        for direction in dw.DIRECTIONS:
            group = [r['walk'] for r in rows if r['condition'] == name and r['direction'] == direction and r['walk']]
            if not group:
                continue
            cell = {'n': len(group), 'progress': float(np.median([g['progress'] for g in group]))}
            for k in GAIT_KEYS:
                v = [g['gait'][k] for g in group if g['gait'][k] is not None]
                cell[k] = float(np.median(v)) if v else None
            table.setdefault(name, {})[direction] = cell
    out = {'conditions': table, 'baselines': baselines}
    (RUNS/f'{args.sweep[0]}-table.json').write_text(json.dumps(out, indent=1))
    f = lambda v, d=2: '—' if v is None else f'{v:.{d}f}'
    for direction in dw.DIRECTIONS:
        print(f'\n#### {direction}\n')
        print('| 来源 / 用法 | 次数 | 站立脚滑动 m/s | 两脚同时动 | 步长 m | 步频 Hz | 完成指令比例 |')
        print('|---|---:|---:|---:|---:|---:|---:|')
        for name, b in baselines.items():
            if direction in b:
                c = b[direction]
                print(f"| 录制参考 {name} | {c['windows']} | {f(c['slide_m_s'][0], 3)} | {f(c['both_moving'][0])} | "
                      f"{f(c['step_length_m'][0])} | {f(c['cadence_hz'][0])} | — |")
        for name in order:
            c = table.get(name, {}).get(direction)
            if c:
                print(f"| Kimodo {name} | {c['n']} | {f(c['slide_m_s'], 3)} | {f(c['both_moving'])} | {f(c['step_length_m'])} | "
                      f"{f(c['cadence_hz'])} | {f(c['progress'])} |")


def score(args):
    rows = []
    for run in sorted(RUNS.glob(args.pattern)):
        manifest = json.loads((run/'run.json').read_text())
        if manifest.get('status') != 'completed' or not manifest.get('demo_segment'):
            continue
        segment = dw.load_segment(manifest['demo_segment'])
        steps = [json.loads(x) for x in (run/'execution.jsonl').read_text().splitlines() if x.strip()]
        result = dw.score_run(steps, json.loads((run/'target.json').read_text()), segment)
        result.update(run=run.name, segment=segment['name'], source=segment['source'])
        (run/'demo-result.json').write_text(json.dumps(result, indent=1))
        rows.append(result)
    summary = summarize(rows)
    name = args.pattern.rstrip('*').rstrip('-')
    (RUNS/f'{name}-demo-summary.json').write_text(json.dumps({'runs': rows, 'summary': summary}, indent=1))
    if args.markdown:
        print(markdown(summary))
    else:
        print(json.dumps(summary, indent=1))


def summarize(rows):
    out = {}
    for direction in dw.DIRECTIONS:
        group = [r for r in rows if r.get('complete') and r['direction'] == direction]
        if not group:
            continue
        completion = [r['completion'] for r in group]
        s = {'runs': len(group), 'fell': sum(r['fell'] for r in group),
             'completion_median': float(np.median(completion)), 'completion_range': [min(completion), max(completion)],
             'at_least_half': sum(c >= .5 for c in completion), 'at_least_80': sum(c >= .8 for c in completion),
             'heading_drift_deg_median': float(np.median([r['heading_drift_deg'] for r in group])),
             'reference_speed_m_s': float(np.median([r['reference']['speed_m_s'] for r in group]))}
        key = 'drift_cm' if direction == 'turn' else 'lateral_cm'
        s[key+'_median'] = float(np.median([r[key] for r in group]))
        for k in GAIT_KEYS:
            values = [r['gait_actual'][k] for r in group if r['gait_actual'][k] is not None]
            s['actual_'+k] = float(np.median(values)) if values else None
        out[direction] = s
    return out


def markdown(summary):
    lines = ['| 方向 | 次数 | 跌倒 | 完成比例 中位数（范围） | ≥50% | ≥80% | 横向 / 位移漂移 cm | 朝向漂移 ° | 参考速度 m/s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d, s in summary.items():
        drift = s.get('lateral_cm_median', s.get('drift_cm_median'))
        lines.append(f"| {d} | {s['runs']} | {s['fell']} | {100*s['completion_median']:.0f}% "
                     f"({100*s['completion_range'][0]:.0f}% 到 {100*s['completion_range'][1]:.0f}%) | {s['at_least_half']} | "
                     f"{s['at_least_80']} | {drift:.0f} | {s['heading_drift_deg_median']:.0f} | {s['reference_speed_m_s']:.2f} |")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    m = sub.add_parser('mine')
    m.add_argument('--data', help='local copy of WilliamWang16/HumanoidArena_dataset_v3_1')
    m.add_argument('--lafan', help='local copy of lvhaidong/LAFAN1_Retargeting_Dataset (its g1/walk*.csv)')
    m.add_argument('--out', required=True)
    s = sub.add_parser('suite')
    s.add_argument('--segments', required=True)
    s.add_argument('--tag', required=True)
    s.add_argument('--gpus', type=int, nargs='+', required=True, help='a GPU named twice runs two replays on it at once')
    s.add_argument('--only', nargs='*', help='segment name prefixes to run')
    s.add_argument('--gpu-min-free-mib', type=int)
    s.add_argument('--stagger', type=float, default=20., help='seconds between starting workers')
    s.add_argument('--dds-domain', type=int, default=20, help='first DDS domain; worker k uses this plus k')
    c = sub.add_parser('score')
    c.add_argument('pattern')
    c.add_argument('--markdown', action='store_true')
    x = sub.add_parser('cross')
    x.add_argument('--config', default=str(ROOT/'configs/walk-attribution.json'))
    x.add_argument('--conditions', nargs='+', required=True)
    x.add_argument('--directions', nargs='*')
    x.add_argument('--repeats', type=int, default=5)
    x.add_argument('--tag', required=True)
    x.add_argument('--gpus', type=int, nargs='+', required=True)
    x.add_argument('--text-cache')
    x.add_argument('--lead-dirs', nargs='*', default=[], type=lambda v: v.split('=', 1))
    x.add_argument('--stagger', type=float, default=20.)
    x.add_argument('--dds-domain', type=int, default=60)
    y = sub.add_parser('score-cross')
    y.add_argument('pattern')
    y.add_argument('--seconds', type=float, default=4.)
    t = sub.add_parser('table')
    t.add_argument('--sweep', nargs='+', required=True)
    t.add_argument('--baseline', nargs='+', required=True, help='segment dirs written by mine')
    args = p.parse_args()
    {'table': sweep_table, 'mine': mine, 'suite': suite, 'score': score, 'cross': cross, 'score-cross': score_cross}[args.command](args)


if __name__ == '__main__':
    main()
