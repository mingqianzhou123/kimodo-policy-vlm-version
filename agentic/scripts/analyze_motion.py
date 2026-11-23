"""Score finished moving-goal runs: did it fall, how far did it get, where did the wrist end up.

Reads runs only. Each run gets a ``motion-result.json`` next to its logs; the table goes to stdout
and to ``<runs>/<pattern without *>/motion.json``. Definitions:

fell        pelvis below 0.40 m, or tilted more than 60 degrees from upright, at any step
completion  root displacement along the commanded direction / commanded distance, at the end
arrived     the final phase began, and 2 s of it fit in the run: the walk got within tolerance and
            stopped (or the lean ended), and for a hand goal the reach had its full time
reached     not fallen, arrived, and the wrist within 10 cm of the target on every step of the 2 s
            after the commanded arrival
The 2 s after arrival are the "hold window" every hold number is taken over.
Errors come in two legs: target -> reference (what Kimodo, after handoff and joint bounds, asked
the tracker for) and reference -> actual (what SONIC did with it).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from paths import RUNS, G1_XML
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.oracle.geometry import XmlFK, rotation_wxyz, angle_deg

FALL_HEIGHT_M, FALL_TILT_DEG, REACH_M, HOLD_S = .40, 60., .10, 2.


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def tilt_deg(rotation):
    return float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1))))


def pitch_deg(rotation, heading):
    """Forward lean of a body's up axis in the start frame's forward-up plane; positive leans forward."""
    up = heading.T @ rotation[:, 2]
    return float(np.degrees(np.arctan2(up[0], up[2])))


def gait(feet, root, moving=.10, lift=.03):
    """Does a 50 Hz foot trajectory [T, 2, 3] step, or slide and jitter? ``root`` is [T, 3].

    In a walk one foot is planted while the other moves, and the feet travel about as far as the
    body does. ``both_moving_share`` is the share of steps on which both ankles move faster than
    ``moving`` m/s over the ground, which no stepping gait does for long. ``foot_travel_m`` is the
    two feet's ground distance and ``root_travel_m`` the root's path length: both grow with
    back-and-forth jitter, so compare them with the commanded distance.
    """
    feet, root = np.asarray(feet, dtype=float), np.asarray(root, dtype=float)
    speed = 50*np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=-1)
    height = feet[..., 2]-feet[..., 2].min(axis=0)
    swing = height > lift
    return {'both_moving_share': float((speed > moving).all(axis=1).mean()),
            'foot_travel_m': float(speed.sum()/50),
            'root_travel_m': float(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=-1).sum()),
            'swings': int((swing[1:] & ~swing[:-1]).sum()), 'max_lift_cm': 100*float(height.max())}


def analyze(run, fk=None):
    run = Path(run)
    target = json.loads((run/'target.json').read_text())
    goal = target['motion_goal']
    steps = rows(run/'execution.jsonl')
    plans = rows(run/'plans.jsonl')
    after = [s['after'] for s in steps]
    time = np.array([a['time'] for a in after])
    start = steps[0]['before']
    heading = np.array(target['heading'])
    root = np.array([a['root_pos'] for a in after])
    rotation = np.array([rotation_wxyz(a['root_quat']) for a in after])
    reference_root = np.array([s['reference_root_world'] for s in steps])
    tilt = np.array([tilt_deg(r) for r in rotation])
    fell = bool((root[:, 2] < FALL_HEIGHT_M).any() or (tilt > FALL_TILT_DEG).any())
    arrival = target['arrival_time']
    hold = (time > arrival+1e-6) & (time <= arrival+HOLD_S+1e-6)
    arrived = bool(time[-1] >= arrival+HOLD_S-1e-6)
    phases = {p['phase']: p for p in target.get('phases', [])}
    out = {'run': run.name, 'goal': goal['name'], 'seed': target.get('noise_seed'),
           'steps': len(steps), 'fell': fell, 'min_root_height_m': float(root[:, 2].min()), 'max_tilt_deg': float(tilt.max()),
           'hold_steps': int(hold.sum()), 'arrived': arrived, 'plans': len(plans),
           'phase_times_s': {name: p['elapsed_s'] for name, p in phases.items()},
           'walk_timed_out': bool(phases.get('stop', {}).get('walk_timed_out', False)),
           'inference_s_mean': float(np.mean([p['inference_s'] for p in plans])),
           'root_tracking_cm': {'mean': 100*float(np.linalg.norm(root-reference_root, axis=1).mean()),
                                'max': 100*float(np.linalg.norm(root-reference_root, axis=1).max())},
           'kimodo_root_path_cm_max': 100*float(max(p['root_path_error_m'] for p in plans))}
    out['gait'] = {'reference': gait([s['reference_foot_world'] for s in steps], reference_root),
                   'actual': gait([a['foot_pos'] for a in after], root)}
    if fk is not None:
        torso = [fk(a['joints'], a['root_pos'], r, body_names=['torso_link'])[1][0] for a, r in zip(after, rotation)]
        lean = np.array([pitch_deg(r, heading) for r in torso])
        out['torso_pitch_deg'] = {'start': float(lean[0]), 'max': float(lean.max()),
                                  'hold_mean': float(lean[hold].mean()) if hold.any() else None}
        if 'lean' in goal and hold.any():
            asked = [pitch_deg(fk(s['reference_joints'], s['reference_root_world'], rotation_wxyz(s['reference_root_quat']),
                                  body_names=['torso_link'])[1][0], heading) for s in steps]
            first = pitch_deg(fk(start['joints'], start['root_pos'], rotation_wxyz(start['root_quat']),
                                 body_names=['torso_link'])[1][0], heading)
            waist = G1_JOINT_NAMES.index('waist_pitch_joint')
            pelvis = np.array([pitch_deg(r, heading) for r in rotation])
            out['lean_parts_deg'] = {  # trunk pitch = pelvis pitch + waist joint, over the hold window
                'waist_joint_reference': float(np.degrees(np.mean(np.array([s['reference_joints'][waist] for s in steps])[hold]))),
                'waist_joint_actual': float(np.degrees(np.mean(np.array([a['joints'][waist] for a in after])[hold]))),
                'pelvis_pitch_change_actual': float(pelvis[hold].mean()-pelvis[0])}
            out['lean'] = {'commanded_deg': goal['lean']['pitch_deg'],
                           'reference_deg': float(np.mean(np.array(asked)[hold])-first),
                           'actual_deg': float(lean[hold].mean()-first), 'actual_max_deg': float(lean.max()-first)}
    if 'root' in goal:
        command = np.array(goal['root']['xy'], dtype=float)
        distance = float(np.linalg.norm(command))
        local = (root[:, :2]-np.array(start['root_pos'])[:2]) @ heading[:2, :2]
        along = command/distance
        yaw = np.degrees([np.arctan2(*(heading.T @ r)[[1, 0], 0]) for r in rotation])
        out['root'] = {'commanded_m': distance, 'direction': along.tolist(),
                       'completion': float(local[-1] @ along)/distance,
                       'final_error_cm': 100*float(np.linalg.norm(local[-1]-command)),
                       'sideways_cm': 100*float(abs(along[0]*local[-1, 1]-along[1]*local[-1, 0])),
                       'final_yaw_deg': float(yaw[-1]), 'max_abs_yaw_deg': float(abs(yaw).max())}
    if target.get('active_sides'):
        i = ('left', 'right').index(target['active_sides'][0])
        wrist_goal = np.array(target['targets_world'])[i, 0]
        wrist_goal_rotation = np.array(target['wrist_rotation_world'])[i]
        wrist = np.array([a['wrist_pos'] for a in after])[:, i]
        reference = np.array([s['reference_wrist_world'] for s in steps])[:, i]
        wrist_rotation = np.array([a['wrist_rot'] for a in after])[:, i]
        distance = np.linalg.norm(wrist-wrist_goal, axis=1)
        angle = np.array([angle_deg(r, wrist_goal_rotation) for r in wrist_rotation])
        hand = {'side': target['active_sides'][0], 'reached': bool(not fell and arrived and (distance[hold] <= REACH_M).all()),
                'closest_cm': 100*float(distance.min())}
        if hold.any():
            hand.update(position_cm={'mean': 100*float(distance[hold].mean()), 'max': 100*float(distance[hold].max()),
                                     'final': 100*float(distance[-1])},
                        orientation_deg={'mean': float(angle[hold].mean()), 'max': float(angle[hold].max())},
                        target_to_reference_cm=100*float(np.linalg.norm(reference[hold]-wrist_goal, axis=1).mean()),
                        reference_to_actual_cm=100*float(np.linalg.norm(wrist[hold]-reference[hold], axis=1).mean()),
                        kimodo_keyframe_cm=100*float(np.mean([p['kimodo_wrist_goal_m'] for p in plans if 'kimodo_wrist_goal_m' in p])),
                        kimodo_keyframe_deg=float(np.mean([p['kimodo_wrist_goal_deg'] for p in plans if 'kimodo_wrist_goal_deg' in p])))
        out['hand'] = hand
    # One verdict per run for the progress page: a hand goal must be reached, any other goal arrived at without a fall.
    out['pass'] = bool(out['hand']['reached'] if 'hand' in out else out['arrived'] and not fell)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('pattern', help='run name pattern, e.g. "exp2-sidestep-*"')
    args = p.parse_args()
    runs = sorted(r for r in RUNS.glob(args.pattern) if (r/'execution.jsonl').exists() and (r/'target.json').exists()
                  and 'motion_goal' in json.loads((r/'target.json').read_text()))
    if not runs:
        raise SystemExit(f'No finished moving-goal runs under {RUNS} match {args.pattern}')
    fk = XmlFK(G1_XML, G1_JOINT_NAMES)
    results = []
    for run in runs:
        result = analyze(run, fk)
        (run/'motion-result.json').write_text(json.dumps(result, indent=1))
        results.append(result)
        line = f"{result['run']:<48} fell={int(result['fell'])} minz={result['min_root_height_m']:.2f} " \
               f"track={result['root_tracking_cm']['mean']:.1f}cm"
        if 'root' in result:
            r = result['root']
            walked = result['phase_times_s'].get('stop', float('nan'))-result['phase_times_s'].get('walk', float('nan'))
            line += f" done={100*r['completion']:.0f}% end={r['final_error_cm']:.1f}cm yaw={r['final_yaw_deg']:.1f} walk={walked:.1f}s timeout={int(result['walk_timed_out'])}"
        g = result['gait']
        line += f" bothfeet ref/act={100*g['reference']['both_moving_share']:.0f}/{100*g['actual']['both_moving_share']:.0f}% feet ref/act={g['reference']['foot_travel_m']:.2f}/{g['actual']['foot_travel_m']:.2f}m"
        if 'lean' in result:
            line += f" lean asked/ref/actual={result['lean']['commanded_deg']:.0f}/{result['lean']['reference_deg']:.1f}/{result['lean']['actual_deg']:.1f}deg"
        if 'hand' in result and 'position_cm' in result['hand']:
            h = result['hand']
            line += f" reached={int(h['reached'])} wrist={h['position_cm']['mean']:.1f}/{h['position_cm']['max']:.1f}cm " \
                    f"{h['orientation_deg']['mean']:.0f}deg goal->ref={h['target_to_reference_cm']:.1f} ref->act={h['reference_to_actual_cm']:.1f}"
        print(line)
    out = RUNS/args.pattern.replace('*', '').rstrip('-')/'motion.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(out)


if __name__ == '__main__':
    main()
