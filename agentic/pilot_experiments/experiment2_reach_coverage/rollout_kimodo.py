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
"""
import argparse
import json
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
    args = p.parse_args()
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
