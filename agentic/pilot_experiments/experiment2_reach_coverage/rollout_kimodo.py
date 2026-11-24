"""Roll a moving goal out through Kimodo alone, without the simulator.

After each plan the "measured" body is the reference just sent, except that its ground position
advances by only ``--carry`` of what the reference asked for (1 = a perfect tracker; SONIC, which
is never told the root position, covers less). This shows in half a minute what Kimodo generates
under the goal's conditions (does it step or slide) and whether replanning from the measured
position still converges on the goal. The start pose is the first measured sample of an
existing run.

``--against RUN`` repeats a simulator run's goal, seed and plan length from that run's own first
sample with a perfect tracker, and compares the result with what the run measured 0.5 to 5 s
after its walk began. That is Kimodo's prediction over several chained plans (ten 0.5 s plans
for 5 s), next to "the body stays where it was".

``--sweep CONFIG`` measures the gait Kimodo generates under different ways of using it (text,
root constraints, history, plan length, guidance, speed; ``walk-attribution.json`` in this folder),
each walking direction and seed once with a perfect tracker, and writes one table next to the
demonstrations' gait (``walk_attribution.py mine``). The model is loaded once.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))  # agentic/scripts: paths.py and the launcher
from paths import CHECKPOINTS, RUNS, sim_cache
from analyze_motion import gait
from kimodo_vla.oracle.geometry import angle_deg, rotation_wxyz
from kimodo_vla.oracle.locomotion import LocomotionPolicy, episode_seconds


def follow(policy, start, seconds, carry=1.):
    """Call the policy for ``seconds``; returns the executed 50 Hz rows with the body's ground position added."""
    log = policy.run/'commands.jsonl'
    body = np.array(start['root_pos'][:2], dtype=float)
    history, k, read, executed = [dict(start)], 0, 0, []
    pose = dict(root_pos=start['root_pos'], root_quat=start['root_quat'], joints=start['joints'],
                wrist_pos=start['wrist_pos'], wrist_rot=start['wrist_rot'])
    while k < seconds*50:
        out = policy({'task': 'rollout', 'observation': {'oracle_history': history}})
        sent = [json.loads(x) for x in log.read_text().splitlines()[read:]] if log.exists() else []
        read += len(sent)
        base, history = body.copy(), []
        for i in range(len(out['action_chunk'])):
            k += 1
            if sent:  # a generated plan: its reference starts on the body, the body covers `carry` of it
                row = sent[i]
                rotation = np.array(row['sent_root_rotation_world'])
                quat = np.roll(Rotation.from_matrix(rotation).as_quat(), 1).tolist()
                body = base+carry*(np.array(row['sent_root_world'][:2])-base)
                root = [body[0], body[1], row['sent_root_world'][2]]
                wrist, wrist_rot = policy.fk(row['sent_joints'], root, rotation)
                pose = dict(root_pos=root, root_quat=quat, joints=row['sent_joints'],
                            wrist_pos=wrist.tolist(), wrist_rot=wrist_rot.tolist(),
                            reference_root_pos=row['sent_root_world'], reference_root_quat=quat,
                            reference_joints=row['sent_joints'])
                executed.append(dict(row, body_xy=body.tolist()))
            history.append({**start, **pose, 'time': start['time']+k/50})
    return executed


CHAIN_HORIZONS = (.5, 1., 1.5, 3., 5.)


def chain_errors(rows, measured, begin):
    """Torso and wrist error of a perfect-tracker rollout against a run's measured samples, by horizon.

    ``rows`` are the rollout's executed rows, ``measured`` the run's ``after`` samples, ``begin`` the
    time the walk began in both. Persistence compares the sample at ``begin`` with the later one.
    """
    def nearest(items, when):
        best = min(items, key=lambda x: abs(x['time']-when))
        return best if abs(best['time']-when) < .011 else None
    now, out = nearest(measured, begin), {}
    for horizon in CHAIN_HORIZONS:
        predicted, actual = nearest(rows, begin+horizon), nearest(measured, begin+horizon)
        if predicted is None or actual is None or now is None:
            continue
        actual_rot, now_rot = rotation_wxyz(actual['root_quat']), rotation_wxyz(now['root_quat'])
        out[str(horizon)] = {
            'kimodo': {'root_pos_cm': 100*float(np.linalg.norm(np.array(predicted['sent_root_world'])-actual['root_pos'])),
                       'root_rot_deg': angle_deg(np.array(predicted['sent_root_rotation_world']), actual_rot),
                       'wrist_pos_cm': (100*np.linalg.norm(np.array(predicted['sent_wrist'])-actual['wrist_pos'], axis=1)).tolist()},
            'persistence': {'root_pos_cm': 100*float(np.linalg.norm(np.array(now['root_pos'])-actual['root_pos'])),
                            'root_rot_deg': angle_deg(now_rot, actual_rot),
                            'wrist_pos_cm': (100*np.linalg.norm(np.array(now['wrist_pos'])-actual['wrist_pos'], axis=1)).tolist()}}
    return out


DIRECTION_GOALS = {'forward': [1, 0], 'backward': [-1, 0], 'left': [0, 1], 'right': [0, -1]}


def condition_goal(direction, speed, cfg):
    """The goal of one sweep rollout: far enough that the measured seconds are all walking."""
    if direction == 'turn':
        return {'name': 'turn', 'root': {'xy': [0., 0.], 'yaw_deg': cfg['turn_deg'], 'yaw_rate_deg_s': speed}}
    return {'name': direction, 'root': {'xy': (np.array(DIRECTION_GOALS[direction])*cfg['distance_m']).tolist(),
                                        'speed_m_s': speed}}


def lead_through_window(segment_path, out):
    """A copy of a recorded segment whose lead-in runs to the end of its directional window."""
    segment = json.loads(Path(segment_path).read_text())
    end = int(segment['window'][1])
    actions = segment['actions'][:end+1]
    actions.append([0., 0.]+actions[-1][2:])     # the window's last pose once more, so the window can end there
    segment['actions'], segment['window'] = actions, [end, end+1]
    out.write_text(json.dumps(segment))
    return out


def score_walk(rows, direction, speed, seconds):
    """Gait and progress of the generated walk over its first ``seconds`` (perfect tracker)."""
    from kimodo_vla.oracle.demo_walks import stepping
    walk = [r for r in rows if r['phase'] == 'walk'][:int(round(seconds*50))]
    if len(walk) < 25:
        return None
    feet = np.array([r['sent_feet'] for r in walk])
    root = np.array([r['sent_root_world'] for r in walk])
    rot = np.array([r['sent_root_rotation_world'] for r in walk])
    yaw = np.unwrap(np.arctan2(rot[:, 1, 0], rot[:, 0, 0]))
    c, sn = np.cos(yaw[0]), np.sin(yaw[0])
    d = root[-1, :2]-root[0, :2]
    local = np.array([c*d[0]+sn*d[1], -sn*d[0]+c*d[1]])
    out = {'gait': stepping(feet), 'seconds': len(walk)/50, 'yaw_deg': float(np.degrees(yaw[-1]-yaw[0]))}
    if direction == 'turn':
        out['progress'] = out['yaw_deg']/(speed*out['seconds'])
    else:
        along = np.array(DIRECTION_GOALS[direction], dtype=float)
        out['moved_m'] = float(local @ along)
        out['progress'] = out['moved_m']/(speed*out['seconds'])
        out['sideways_m'] = float(abs(along[0]*local[1]-along[1]*local[0]))
    return out


def sweep(args):
    """Every condition x direction x seed of the sweep config; one results file and a table."""
    from kimodo_vla.model.native_policy import NativeKimodoPolicy
    cfg = json.loads(Path(args.sweep).read_text())
    out = RUNS/args.run_id
    out.mkdir(exist_ok=False)
    (out/'config.json').write_text(json.dumps(cfg, indent=1))
    start = json.loads((RUNS/cfg['start_from']/'execution.jsonl').open().readline())['before']
    model = NativeKimodoPolicy(CHECKPOINTS/'Kimodo-G1-RP-v1', constraint_cfg=2.).to('cuda').eval()
    text_cache = Path(args.text_cache) if args.text_cache else sim_cache()/'text-eval'
    results = []
    only = set(args.only or cfg['conditions'])
    for name, change in cfg['conditions'].items():
        if name not in only:
            continue
        use = {**cfg['base'], **change}
        for direction in args.directions or cfg['directions']:
            rate = use['turn_rate'] if direction == 'turn' else use['speed']
            speed = rate[direction] if isinstance(rate, dict) else rate
            for i, seed in enumerate(cfg['seeds']):
                run = out/f'{name}-{direction}-s{seed}'
                run.mkdir()
                goal = run/'goal.json'
                goal.write_text(json.dumps(condition_goal(direction, speed, cfg)))
                lead = None
                if use.get('lead'):
                    names = cfg['lead_segments'][use['lead']][direction]
                    lead = lead_through_window(Path(dict(args.lead_dirs)[use['lead']])/f'{names[i % len(names)]}.json',
                                               run/'lead.json')
                began = time.time()
                policy = LocomotionPolicy(CHECKPOINTS/'Kimodo-G1-RP-v1', text_cache, run, 'hold',
                    constraint_cfg=use['constraint_cfg'], variant='bounded', joint_speed=4.,
                    support_constraints='none', hand_constraints='keyframes', trajectory_projection='none',
                    motion_goal=goal, noise_seed=seed, walk_steps=use['walk_steps'],
                    text=('auto' if use.get('generator') == 'native' else cfg['texts'][direction]) if use.get('text') else None,
                    text_cfg=use.get('text_cfg', 2.),
                    root_constraints=use['root_constraints'], history_frames=use['history_frames'],
                    lead_segment=lead, model=model, generator=use.get('generator', 'policy'),
                    segment_seconds=use.get('segment_seconds', 4.), segment_steps=use.get('segment_steps', 100),
                    transition_frames=use.get('transition_frames', 5), approach_gain=use.get('approach_gain', 1.),
                    text_until_m=use.get('text_until_m', 0.))
                lead_s = 0. if lead is None else (len(json.loads(lead.read_text())['actions'])+50)/50
                rows = follow(policy, start, .7+lead_s+cfg['seconds']+.1)
                result = {'condition': name, 'direction': direction, 'seed': seed, 'use': use, 'speed': speed,
                          'lead': None if lead is None else json.loads(lead.read_text())['name'],
                          'wall_s': time.time()-began,
                          'walk': score_walk(rows, direction, speed, cfg['seconds'])}
                (run/'result.json').write_text(json.dumps(result, indent=1))
                for f in run.glob('plan-*.npz'):
                    f.unlink()              # the per-plan arrays are 0.2 MB each; the table needs none of them
                (run/'commands.jsonl').unlink(missing_ok=True)
                results.append(result)
                w = result['walk']
                print(f"{name} {direction} s{seed}: " + ('no walk' if w is None else
                      f"progress {w['progress']:.2f}, slide {w['gait']['slide_m_s'] or 0:.3f}, "
                      f"both {w['gait']['both_moving']:.2f}, steps {w['gait']['steps']}"), flush=True)
    (out/'sweep.json').write_text(json.dumps(results, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-id', required=True)
    p.add_argument('--motion-goal', type=Path)
    p.add_argument('--start-from', help='finished run whose first measured sample is the start pose')
    p.add_argument('--against', help='finished simulator run to repeat with a perfect tracker and compare with')
    p.add_argument('--carry', type=float, default=1., help='share of each commanded ground displacement the body covers')
    p.add_argument('--walk-steps', type=int, default=75)
    p.add_argument('--noise-seed', type=int, default=1000)
    p.add_argument('--constraint-cfg', type=float, default=2.)
    p.add_argument('--joint-speed', type=float, default=4.)
    p.add_argument('--seconds', type=float, help='default: the goal\'s longest possible episode')
    p.add_argument('--sweep', help='sweep config (walk-attribution.json in this folder): gait under each way of using Kimodo')
    p.add_argument('--only', nargs='*', help='with --sweep: these conditions only')
    p.add_argument('--directions', nargs='*', help='with --sweep: these directions only')
    p.add_argument('--text-cache', help='with --sweep: the phrase embeddings (default <KVLA_SIM_CACHE>/text-eval)')
    p.add_argument('--lead-dirs', nargs='*', default=[], type=lambda x: x.split('=', 1),
                   help='with --sweep: SET=DIR for each set of recorded segments the config names (walk_attribution.py mine output)')
    args = p.parse_args()
    if args.sweep:
        return sweep(args)
    if args.against:
        theirs = json.loads((RUNS/args.against/'target.json').read_text())
        manifest = json.loads((RUNS/args.against/'run.json').read_text())
        goal_file = RUNS/args.run_id/'goal.json'
        args.start_from, args.noise_seed, args.walk_steps = args.against, theirs['noise_seed'], theirs['walk_steps']
        args.constraint_cfg, args.joint_speed, args.carry = 2., manifest['joint_speed_rad_s'], 1.
    elif not (args.motion_goal and args.start_from):
        raise SystemExit('Give --motion-goal and --start-from, or --against')
    run = RUNS/args.run_id
    run.mkdir(exist_ok=False)
    if args.against:
        goal_file.write_text(json.dumps(theirs['motion_goal']))
        args.motion_goal = goal_file
    start = json.loads((RUNS/args.start_from/'execution.jsonl').open().readline())['before']
    policy = LocomotionPolicy(CHECKPOINTS/'Kimodo-G1-RP-v1', sim_cache()/'text-eval', run, 'hold',
        constraint_cfg=args.constraint_cfg, variant='bounded', joint_speed=args.joint_speed,
        support_constraints='none', hand_constraints='keyframes', trajectory_projection='none',
        motion_goal=args.motion_goal, noise_seed=args.noise_seed, walk_steps=args.walk_steps)
    rows = follow(policy, start, args.seconds or episode_seconds(policy.goal), args.carry)
    heading = np.array(policy.target_info['heading'])
    body = np.array([r['body_xy'] for r in rows])
    moved = (body[-1]-np.array(start['root_pos'][:2])) @ heading[:2, :2]
    walking = [r for r in rows if r['phase'] in ('walk', 'stop')]
    result = {'run': args.run_id, 'goal': policy.goal, 'options': vars(args) | {'motion_goal': str(args.motion_goal)},
              'body_moved_start_frame_m': moved.tolist(), 'phases': policy.phases,
              'min_root_height_m': float(min(r['sent_root_world'][2] for r in rows))}
    line = f"{args.run_id}: moved fwd/left {moved[0]:.2f}/{moved[1]:.2f} m, phases " \
           + ' '.join(f"{x['phase']}@{x['elapsed_s']:.1f}" for x in policy.phases)
    if walking:
        g = result['gait_walking'] = gait([r['sent_feet'] for r in walking], [r['sent_root_world'] for r in walking])
        line += f", walking: both feet moving {100*g['both_moving_share']:.0f}%, feet {g['foot_travel_m']:.2f} m, " \
                f"max lift {g['max_lift_cm']:.1f} cm"
    if 'hand' in policy.goal and policy.hand_start is not None:
        i = policy.hand_index
        final = np.linalg.norm(np.array(rows[-1]['sent_wrist'][i])-policy.hand_goal)
        result['reference_wrist_goal_cm_final'] = 100*float(final)
        line += f", reference wrist ends {100*final:.1f} cm from target"
    if args.against:
        steps = [json.loads(x) for x in (RUNS/args.against/'execution.jsonl').read_text().splitlines()]
        walk = next((x for x in policy.phases if x['phase'] == 'walk'), None)
        theirs_walk = next((x for x in theirs['phases'] if x['phase'] == 'walk'), None)
        if walk is None or theirs_walk is None or abs(walk['time']-theirs_walk['time']) > 1e-6:
            raise SystemExit('The rollout and the run did not begin a walk at the same time')
        result['chain_prediction'] = chain_errors(rows, [x['after'] for x in steps], walk['time'])
        line += ' | chain root cm kimodo/persistence: ' + ' '.join(
            f"{h}s {e['kimodo']['root_pos_cm']:.0f}/{e['persistence']['root_pos_cm']:.0f}"
            for h, e in result['chain_prediction'].items())
    (run/'rollout.json').write_text(json.dumps(result, indent=1, default=str))
    print(line)


if __name__ == '__main__':
    main()
