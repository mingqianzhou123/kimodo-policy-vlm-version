"""Roll a moving goal out through Kimodo alone, as if the tracker followed every reference exactly.

No simulator: after each plan the "measured" body is the reference just sent. This shows what
Kimodo generates under the goal's conditions (does it step or slide, does the root reach the end
of its path) in about half a minute, and is the kinematic best case for the same goal in the
simulator. The start pose is the first measured sample of an existing run.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from paths import CHECKPOINTS, RUNS, sim_cache
from analyze_motion import gait
from kimodo_vla.oracle.locomotion import HISTORY_SOURCES, ROOT_CONDITIONS, LocomotionPolicy, end_time


def follow(policy, start, seconds):
    """Call the policy until ``seconds`` have passed, feeding each sent reference back as the measurement."""
    def sample(k, pose=None):
        row = dict(start, time=start['time']+k/50)
        if pose is not None:
            quat = np.roll(Rotation.from_matrix(pose['sent_root_rotation_world']).as_quat(), 1).tolist()
            wrist, wrist_rot = policy.fk(pose['sent_joints'], pose['sent_root_world'], np.array(pose['sent_root_rotation_world']))
            row.update(root_pos=pose['sent_root_world'], root_quat=quat, joints=pose['sent_joints'],
                       reference_root_pos=pose['sent_root_world'], reference_root_quat=quat,
                       reference_joints=pose['sent_joints'], wrist_pos=wrist.tolist(), wrist_rot=wrist_rot.tolist())
        return row
    log = policy.run/'commands.jsonl'
    history, k, read, last = [sample(0)], 0, 0, None
    while k < seconds*50:
        out = policy({'task': 'rollout', 'observation': {'oracle_history': history}})
        sent = [json.loads(x) for x in log.read_text().splitlines()[read:]] if log.exists() else []
        read += len(sent)
        poses = sent or [last]*len(out['action_chunk'])
        history = []
        for pose in poses:
            k += 1
            history.append(sample(k, pose))
        last = poses[-1]
    return [json.loads(x) for x in log.read_text().splitlines()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-id', required=True)
    p.add_argument('--motion-goal', required=True, type=Path)
    p.add_argument('--start-from', required=True, help='finished run whose first measured sample is the start pose')
    p.add_argument('--history-source', choices=HISTORY_SOURCES, default='reference')
    p.add_argument('--root-condition', choices=ROOT_CONDITIONS, default='path')
    p.add_argument('--sent-steps', type=int, default=25)
    p.add_argument('--noise-seed', type=int, default=1000)
    p.add_argument('--constraint-cfg', type=float, default=2.)
    p.add_argument('--joint-speed', type=float, default=4.)
    p.add_argument('--hold-s', type=float, default=2.)
    args = p.parse_args()
    run = RUNS/args.run_id
    run.mkdir(exist_ok=False)
    start = json.loads((RUNS/args.start_from/'execution.jsonl').open().readline())['before']
    policy = LocomotionPolicy(CHECKPOINTS/'Kimodo-G1-RP-v1', sim_cache()/'text-eval', run, 'hold',
        constraint_cfg=args.constraint_cfg, variant='bounded', joint_speed=args.joint_speed,
        support_constraints='none', hand_constraints='keyframes', trajectory_projection='none',
        motion_goal=args.motion_goal, history_source=args.history_source, noise_seed=args.noise_seed,
        sent_steps=args.sent_steps, root_condition=args.root_condition)
    rows = follow(policy, start, end_time(policy.goal)+args.hold_s)
    root = np.array([r['sent_root_world'] for r in rows])
    heading = np.array(policy.target_info['heading'])
    moved = (root[-1]-np.array(start['root_pos'])) @ heading
    result = {'run': args.run_id, 'goal': policy.goal, 'options': vars(args) | {'motion_goal': str(args.motion_goal)},
              'root_moved_start_frame_m': moved[:2].tolist(), 'min_root_height_m': float(root[:, 2].min()),
              'gait_sent': gait([r['sent_feet'] for r in rows]), 'gait_raw': gait([r['raw_feet'] for r in rows])}
    (run/'rollout.json').write_text(json.dumps(result, indent=1, default=str))
    g = result['gait_sent']
    print(f"{args.run_id}: moved fwd/left {moved[0]:.2f}/{moved[1]:.2f} m, feet slide {100*g['slide_share']:.0f}% "
          f"of {g['travel_m']:.2f} m, swings {g['swings']}, max lift {g['max_lift_cm']:.1f} cm, min root z {root[:, 2].min():.2f}")


if __name__ == '__main__':
    main()
