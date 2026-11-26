"""Experiment 2, part A: the standardized command test (protocol frozen in README.md, 2026-11-26).

    python exp2_commands.py run --condition empty --gpus 0 0 2 2 3 3 [--only forward turn]
    python exp2_commands.py analyze --condition empty [--date 20261126]

``run`` launches one episode per (motion, seed) through ``agentic/scripts/run_oracle.py --motion-goal``
with the 2026-11-25 native planner flags, two per listed GPU slot, each slot on its own DDS domain.
Runs are named ``p2e2-cmd-<condition>-<motion>-<date>-s<seed>``; goal files go to
``<runs>/p2e2-cmd-goals/``. Finished runs are skipped.

``analyze`` measures each finished run at the moment the robot has stopped (``measure``) and writes
``<runs>/p2e2-cmd-<condition>-<date>.json`` plus the README table (mean and standard deviation of
actual minus commanded over the five repeats).
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

HERE = Path(__file__).resolve().parent
AGENTIC = HERE.parents[1]
sys.path.insert(0, str(AGENTIC/'scripts'))

START_XY = (-4., -4.)
START_RADIUS_M = .05
SEEDS = (1000, 1001, 1002, 1003, 1004)
PLANNER_FLAGS = ['--generator', 'native', '--walk-steps', '50', '--segment-seconds', '5',
                 '--transition-frames', '5', '--approach-gain', '2']
# name -> (goal, commanded change in the start frame: forward m, left m, yaw deg, pelvis height m, torso pitch deg)
MOTIONS = {
    'forward': ({'root': {'xy': [.5, 0.], 'speed_m_s': .35}}, (.5, 0., 0., 0., 0.)),
    'sidestep': ({'root': {'xy': [0., .5], 'speed_m_s': .35}}, (0., .5, 0., 0., 0.)),
    'backward': ({'root': {'xy': [-.5, 0.], 'speed_m_s': .35}}, (-.5, 0., 0., 0., 0.)),
    'turn': ({'root': {'xy': [0., 0.], 'yaw_deg': 90., 'yaw_rate_deg_s': 45.}}, (0., 0., 90., 0., 0.)),
    'squat': ({'squat': {'drop_m': .15}}, (0., 0., 0., -.15, 0.)),
    'lean': ({'lean': {'pitch_deg': 30., 'by': 'hips'}}, (0., 0., 0., 0., 30.)),
    'reach': ({'hand': {'side': 'right', 'position': [.35, -.2, .95], 'approach': [1., 0., 0.], 'duration_s': 1.5}},
              (0., 0., 0., 0., 0.)),
}
# Stop criterion: every one of these holds on STOP_STEPS consecutive control steps after the last phase began.
STOP_SPEED_M_S, STOP_YAW_DEG_S, STOP_VERTICAL_M_S, STOP_STEPS, RATE = .05, 5., .05, 25, 50.
FALL_HEIGHT_M, FALL_TILT_DEG = .40, 60.
CARRY_STEPS = 3000   # 60 s: grasp, the 0.6 m clearing walk and the command, with room for one retried grasp


def start_offset(seed, radius=START_RADIUS_M):
    """The start perturbation of one repeat: uniform over a disc of ``radius`` metres, fixed by the seed."""
    u, v = np.random.default_rng(seed).random(2)
    r, a = radius*np.sqrt(u), 2*np.pi*v
    return np.array([r*np.cos(a), r*np.sin(a)])


def goal_for(motion):
    goal, _ = MOTIONS[motion]
    return {'name': f'p2e2-{motion}', **json.loads(json.dumps(goal))}


def run_name(condition, motion, date, seed):
    return f'p2e2-cmd-{condition}-{motion}-{date}-s{seed}'


# ---- measurement --------------------------------------------------------------------------------
def rotation_wxyz(q):
    from scipy.spatial.transform import Rotation
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def yaw_in(heading, rotation):
    local = heading.T @ rotation
    return float(np.degrees(np.arctan2(local[1, 0], local[0, 0])))


def pitch_in(heading, rotation):
    up = heading.T @ rotation[:, 2]
    return float(np.degrees(np.arctan2(up[0], up[2])))


def rates(time, root, yaw, span=5):
    """Root horizontal speed, vertical speed and yaw rate per sample, over +-``span`` steps."""
    n = len(time)
    lo, hi = np.clip(np.arange(n)-span, 0, n-1), np.clip(np.arange(n)+span, 0, n-1)
    dt = np.maximum(time[hi]-time[lo], 1e-9)
    return (np.linalg.norm(root[hi, :2]-root[lo, :2], axis=1)/dt, np.abs(root[hi, 2]-root[lo, 2])/dt,
            np.abs(yaw[hi]-yaw[lo])/dt)


def stop_index(time, root, yaw, after_time):
    """Index of the last sample of the first STOP_STEPS-long still window after ``after_time``, or None."""
    speed, vertical, turning = rates(time, root, yaw)
    still = (speed < STOP_SPEED_M_S) & (vertical < STOP_VERTICAL_M_S) & (turning < STOP_YAW_DEG_S) & (time >= after_time-1e-6)
    count = 0
    for i, ok in enumerate(still):
        count = count+1 if ok else 0
        if count == STOP_STEPS:
            return i
    return None


def measure(samples, heading, commanded, last_phase_time, torso=None, wrist_goal=None):
    """Actual minus commanded for one episode, at the stop (see the README protocol).

    ``samples``: the measured states after each control step (dicts with time, root_pos, root_quat,
    wrist_pos, wrist_rot), from the start of the measured command on. ``heading``: the start frame's
    3x3 heading rotation. ``commanded``: (forward m, left m, yaw deg, pelvis height m, torso pitch deg).
    ``torso``: torso_link rotations per sample, or None. ``wrist_goal``: (side index, position, rotation).
    """
    time = np.array([s['time'] for s in samples])
    root = np.array([s['root_pos'] for s in samples], dtype=float)
    rotation = [rotation_wxyz(s['root_quat']) for s in samples]
    yaw = np.unwrap(np.radians([yaw_in(heading, r) for r in rotation]))
    yaw = np.degrees(yaw-yaw[0])
    tilt = np.degrees(np.arccos(np.clip([r[2, 2] for r in rotation], -1, 1)))
    fell = bool((root[:, 2] < FALL_HEIGHT_M).any() or (tilt > FALL_TILT_DEG).any())
    i = stop_index(time, root, yaw, last_phase_time)
    stopped = i is not None
    i = len(time)-1 if i is None else i
    local = heading[:2, :2].T @ (root[i, :2]-root[0, :2])
    forward, left, turn, height, pitch = commanded
    out = {'fell': fell, 'stopped': stopped, 'measured_at_s': float(time[i]-time[0]), 'measured_index': int(i),
           'root_along_m': float(local[0]-forward), 'root_across_m': float(local[1]-left),
           'root_m': float(np.hypot(local[0]-forward, local[1]-left)),
           'yaw_deg': float(yaw[i]-turn), 'pelvis_height_m': float(root[i, 2]-root[0, 2]-height)}
    if torso is not None:
        out['torso_pitch_deg'] = float(pitch_in(heading, torso[i])-pitch_in(heading, torso[0])-pitch)
    if wrist_goal is not None:
        side, position, goal_rotation = wrist_goal
        actual = np.asarray(samples[i]['wrist_pos'][side])
        actual_rotation = np.asarray(samples[i]['wrist_rot'][side])
        cos = (np.trace(np.asarray(goal_rotation).T @ actual_rotation)-1)/2
        out['wrist_m'] = float(np.linalg.norm(actual-np.asarray(position)))
        out['wrist_deg'] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return out


def analyze_run(run, fk=None):
    """One empty-handed run: the whole episode is the measured command."""
    target = json.loads((run/'target.json').read_text())
    steps = [json.loads(x) for x in (run/'execution.jsonl').read_text().splitlines() if x.strip()]
    samples = [s['after'] for s in steps]
    heading = np.array(target['heading'])
    goal = target['motion_goal']
    motion = next(m for m in MOTIONS if goal['name'] == f'p2e2-{m}')
    phases = {p['phase']: p['time'] for p in target.get('phases', [])}
    last = phases.get('hold', target['arrival_time'])
    torso = None
    if fk is not None:
        torso = [fk(s['joints'], s['root_pos'], rotation_wxyz(s['root_quat']), body_names=['torso_link'])[1][0] for s in samples]
    wrist = None
    if target.get('active_sides'):
        side = ('left', 'right').index(target['active_sides'][0])
        wrist = (side, np.array(target['targets_world'])[side, 0], np.array(target['wrist_rotation_world'])[side])
    return {'run': run.name, 'motion': motion, 'seed': target.get('noise_seed'),
            **measure(samples, heading, MOTIONS[motion][1], last, torso, wrist)}


def analyze_carry_run(run, fk=None):
    """One carrying run: the measured command starts when carry.json says so (kimodo_vla/oracle/carry_policy.py)."""
    record = json.loads((run/'run.json').read_text())
    motion = next(m for m in MOTIONS if record['carry_goal']['name'] == f'p2e2-{m}')
    out = {'run': run.name, 'motion': motion, 'seed': record['seed']}
    carry = run/'carry.json'
    stages = {x['stage']: x for x in json.loads(carry.read_text())['stages']} if carry.exists() else {}
    if 'command' not in stages:
        out['carry_status'] = 'no_grasp' if 'clear' not in stages else 'clear_never_stopped'
        return out
    t0 = stages['command']['time']
    target = json.loads((run/'carry-command'/'target.json').read_text())
    steps = [json.loads(x) for x in (run/'execution.jsonl').read_text().splitlines() if x.strip()]
    steps = [s for s in steps if s['after']['time'] >= t0-1e-6]
    samples = [s['after'] for s in steps]
    phases = {p['phase']: p['time'] for p in target.get('phases', [])}
    torso = None
    if fk is not None:
        torso = [fk(s['joints'], s['root_pos'], rotation_wxyz(s['root_quat']), body_names=['torso_link'])[1][0] for s in samples]
    out.update(carry_status='measured', **measure(samples, np.array(target['heading']), MOTIONS[motion][1],
                                                 phases.get('hold', target['arrival_time']), torso))
    i = out['measured_index']
    grasp = steps[i].get('grasp') or {}
    z0, zi = samples[0]['box_com'][2], samples[i]['box_com'][2]
    rest = json.loads((run/'grasp-result.json').read_text())['rest_height']
    # Held: both hands press on it and it is clear of its resting height (as the grasp judge's lift, 5 cm).
    out.update(box_drop_m=float(z0-zi), box_above_rest_m=float(zi-rest),
               held_at_measure=bool(grasp.get('left_n', 0) >= .05 and grasp.get('right_n', 0) >= .05 and zi-rest >= .05),
               clear_lift_m=stages['clear'].get('lift_m'))
    return out


QUANTITIES = (('root_m', '根部位置 cm', 100), ('root_along_m', '沿朝向 cm', 100), ('root_across_m', '横向 cm', 100),
              ('yaw_deg', '朝向 °', 1), ('pelvis_height_m', '骨盆高度 cm', 100), ('torso_pitch_deg', '躯干俯仰 °', 1),
              ('wrist_m', '手腕位置 cm', 100), ('wrist_deg', '手腕朝向 °', 1))


def summarize(results):
    """Per motion: mean and sample standard deviation of every quantity, falls and not-stopped counts."""
    table = {}
    for motion in MOTIONS:
        rows = [r for r in results if r['motion'] == motion]
        if not rows:
            continue
        unmeasured = [r for r in rows if r.get('carry_status', 'measured') != 'measured']
        rows = [r for r in rows if r.get('carry_status', 'measured') == 'measured']
        entry = {'n': len(rows), 'fell': sum(r['fell'] for r in rows), 'not_stopped': sum(not r['stopped'] for r in rows),
                 'unmeasured': len(unmeasured), 'dropped': sum(not r.get('held_at_measure', True) for r in rows)}
        for key, _, _ in QUANTITIES:
            values = np.array([r[key] for r in rows if key in r and not r['fell']], dtype=float)
            if len(values):
                entry[key] = {'mean': float(values.mean()), 'sd': float(values.std(ddof=1)) if len(values) > 1 else None}
        table[motion] = entry
    return table


def markdown(table, condition):
    lines = [f'| 指令（{condition}） | n | 摔倒 | 未停稳 | 箱子掉了 | ' + ' | '.join(label for _, label, _ in QUANTITIES) + ' |',
             '|---|---:|---:|---:|---:|' + '---:|'*len(QUANTITIES)]
    for motion, e in table.items():
        cells = []
        for key, _, scale in QUANTITIES:
            if key not in e:
                cells.append('—')
            else:
                sd = e[key]['sd']
                cells.append(f"{scale*e[key]['mean']:.1f} ± {scale*sd:.1f}" if sd is not None else f"{scale*e[key]['mean']:.1f}")
        lines.append(f"| {motion} | {e['n']} | {e['fell']} | {e['not_stopped']} | {e['dropped']} | " + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


# ---- launching ----------------------------------------------------------------------------------
def launch(args, extra):
    from paths import AGENTIC as ROOT, RUNS
    from kimodo_vla.oracle.locomotion import episode_seconds, load_goal
    carry = args.condition == 'carry'
    seeds = args.seeds or SEEDS
    jobs = [(m, s) for m in MOTIONS if (not args.only or m in args.only) and not (carry and m == 'reach') for s in seeds]
    goals = RUNS/'p2e2-cmd-goals'
    goals.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def worker(slot, gpu):
        while True:
            with lock:
                if not jobs:
                    return
                motion, seed = jobs.pop(0)
            run_id = run_name(args.condition, motion, args.date, seed)
            manifest = RUNS/run_id/'run.json'
            if manifest.exists() and json.loads(manifest.read_text()).get('status') == 'completed':
                print('SKIP', run_id, flush=True)
                continue
            if (RUNS/run_id).exists():
                print('EXISTS-UNFINISHED', run_id, flush=True)
                continue
            goal = goal_for(motion)
            path = goals/f"{goal['name']}.json"
            path.write_text(json.dumps(goal, indent=1))
            if carry:   # box placed by the seed in the standing region; grasp, clear, command (README protocol)
                cmd = [sys.executable, 'scripts/run_oracle.py', '--run-id', run_id, '--grasp-loop', '--stand-still',
                       '--box-in-region', '--seed', str(seed), '--carry-goal', str(path), '--steps', str(CARRY_STEPS),
                       '--sim-gpu', str(gpu), '--policy-gpu', str(gpu), '--share-policy-sim-gpu',
                       '--gpu-min-free-mib', '16384', *extra]
            else:
                xy = np.array(START_XY)+start_offset(seed)
                steps = int(round(episode_seconds(load_goal(goal))*RATE))
                cmd = [sys.executable, 'scripts/run_oracle.py', '--run-id', run_id, '--motion-goal', str(path),
                       '--start-xy', f'{xy[0]:.4f}', f'{xy[1]:.4f}', '--noise-seed', str(seed), '--steps', str(steps),
                       '--sim-gpu', str(gpu), '--policy-gpu', str(gpu), '--share-policy-sim-gpu',
                       '--gpu-min-free-mib', '16384', *PLANNER_FLAGS, *extra]
            env = dict(os.environ, KVLA_DDS_DOMAIN=str(args.dds_domain+slot))
            from exp2_replay import wait_free
            if not wait_free(gpu):
                print('GPU-BUSY', run_id, 'gpu', gpu, flush=True)
                continue
            print('START', run_id, 'gpu', gpu, flush=True)
            done = subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print('COMPLETE' if done.returncode == 0 else 'FAILED', run_id, flush=True)
            time.sleep(5)

    threads = [threading.Thread(target=worker, args=(slot, gpu)) for slot, gpu in enumerate(args.gpus)]
    for i, thread in enumerate(threads):
        thread.start()
        time.sleep(20)   # stagger simulator start-up
    for thread in threads:
        thread.join()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['run', 'analyze', 'list'])
    p.add_argument('--condition', choices=['empty', 'carry'], default='empty')
    p.add_argument('--seeds', type=int, nargs='+', help='carry: replacement layout seeds for failed grasps')
    p.add_argument('--date', default=time.strftime('%Y%m%d'))
    p.add_argument('--gpus', type=int, nargs='+', default=[])
    p.add_argument('--only', nargs='+', choices=list(MOTIONS))
    p.add_argument('--dds-domain', type=int, default=40)
    args, extra = p.parse_known_args()
    if args.action == 'list':
        for m in MOTIONS:
            for s in SEEDS:
                print(run_name(args.condition, m, args.date, s), np.round(np.array(START_XY)+start_offset(s), 4).tolist(),
                      json.dumps(goal_for(m)))
        return
    if args.action == 'run':
        if not args.gpus:
            raise SystemExit('--gpus is required')
        return launch(args, extra)
    from paths import RUNS, G1_XML
    from kimodo_vla.core.types import G1_JOINT_NAMES
    from kimodo_vla.oracle.geometry import XmlFK
    fk = XmlFK(G1_XML, G1_JOINT_NAMES)
    results = []
    for run in sorted(RUNS.glob(f'p2e2-cmd-{args.condition}-*-{args.date}-s*')):
        manifest = run/'run.json'
        if not manifest.exists() or json.loads(manifest.read_text()).get('status') != 'completed':
            print('NOT-COMPLETED', run.name)
            continue
        result = analyze_carry_run(run, fk) if args.condition == 'carry' else analyze_run(run, fk)
        results.append(result)
        print(json.dumps(result))
    table = summarize(results)
    out = RUNS/f'p2e2-cmd-{args.condition}-{args.date}.json'
    out.write_text(json.dumps({'runs': results, 'table': table}, indent=1))
    print(markdown(table, args.condition))
    print(out)


if __name__ == '__main__':
    main()
