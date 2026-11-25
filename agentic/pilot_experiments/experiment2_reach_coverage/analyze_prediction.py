"""How far Kimodo's prediction at plan time is from what SONIC then did.

Reads finished runs only. For every saved plan, the 50-frame raw Kimodo output
(30 Hz, before handoff and IK) is decoded to root and wrist poses and compared
with the measured state at the same simulation time, 0.5, 1.0 and 1.5 s ahead.
"Persistence" is the trivial predictor that assumes the body stays where it was
measured at plan time; a prediction is only informative where it beats that.

Plans are grouped by what the robot was doing when they were made: reaching and holding for the
standing runs, walk / stop / reach for moving goals. A horizon is only a prediction where a
condition covers it: the torso where the root is conditioned at that frame, a wrist up to its
last conditioned frame. The ``covered`` columns keep only those; the plain wrist columns of a
moving-goal run pool the conditioned hand only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))  # agentic/scripts: paths.py and the launcher
from paths import RUNS, G1_XML
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.model.features import KimodoFeatures
from kimodo_vla.motion.g1_reference import rot6d_row_to_matrix
from kimodo_vla.oracle.geometry import Frame, XmlFK, rotation_wxyz, angle_deg

HORIZONS = (0., .5, 1., 1.5)
# "aligned" is Kimodo's prediction moved rigidly so that its frame zero sits on the measured root:
# the predicted displacement from where the body is, without the offset the raw output starts with.
SOURCES = ('kimodo', 'aligned', 'persistence')


def decode_plan(features, fk, frame, motion_features):
    """World root position/rotation and wrist position/rotation of every predicted frame."""
    out = features.representation.inverse(torch.from_numpy(motion_features)[None], is_normalized=False)
    local, root = out['local_rot_mats'][0], out['root_positions'][0]
    actions = features.codec.encode(local, root, previous_root_position=root[0],
                                    hand_binary=torch.zeros(len(root), 2)).numpy()
    root_rot = frame.heading @ rot6d_row_to_matrix(torch.from_numpy(actions[:, 3:9])).numpy()
    root_pos = frame.world(root.numpy())
    wrists = [fk(a[9:38], p, r) for a, p, r in zip(actions, root_pos, root_rot)]
    return root_pos, root_rot, np.array([w[0] for w in wrists]), np.array([w[1] for w in wrists])


def measured_series(run):
    rows = [json.loads(x) for x in (Path(run)/'execution.jsonl').read_text().splitlines()]
    samples = [rows[0]['before']] + [r['after'] for r in rows]
    return {'time': np.array([s['time'] for s in samples]),
            'root_pos': np.array([s['root_pos'] for s in samples]),
            'root_rot': np.array([rotation_wxyz(s['root_quat']) for s in samples]),
            'wrist_pos': np.array([s['wrist_pos'] for s in samples]),
            'wrist_rot': np.array([s['wrist_rot'] for s in samples])}


def measured_at(series, time):
    """Positions linearly interpolated, rotations from the nearest 50 Hz sample; None outside the record."""
    t = series['time']
    if not t[0]-1e-9 <= time <= t[-1]+1e-9:
        return None
    hi = int(np.clip(np.searchsorted(t, time), 1, len(t)-1))
    lo = hi-1
    w = float(np.clip((time-t[lo])/(t[hi]-t[lo]), 0, 1))
    near = hi if w > .5 else lo
    return {'root_pos': (1-w)*series['root_pos'][lo]+w*series['root_pos'][hi],
            'wrist_pos': (1-w)*series['wrist_pos'][lo]+w*series['wrist_pos'][hi],
            'root_rot': series['root_rot'][near], 'wrist_rot': series['wrist_rot'][near]}


def errors(pred, actual):
    return {'root_pos_cm': 100*float(np.linalg.norm(pred['root_pos']-actual['root_pos'])),
            'root_rot_deg': angle_deg(pred['root_rot'], actual['root_rot']),
            'wrist_pos_cm': (100*np.linalg.norm(pred['wrist_pos']-actual['wrist_pos'], axis=-1)).tolist(),
            'wrist_rot_deg': [angle_deg(a, b) for a, b in zip(pred['wrist_rot'], actual['wrist_rot'])]}


def analyze_run(run, features, fk, horizons=HORIZONS):
    run = Path(run)
    target = json.loads((run/'target.json').read_text())
    frame = Frame(target['origin'], target['heading'])
    series = measured_series(run)
    start = series['time'][0]  # the policy's clock starts at the first measured sample
    moving = 'motion_goal' in target
    sides = [('left', 'right').index(s) for s in target['active_sides']] if moving else [0, 1]
    rows = []
    for plan in (json.loads(x) for x in (run/'plans.jsonl').read_text().splitlines()):
        if round((plan['time']-start)*30) != plan['cut']:
            raise ValueError(f"{run.name} plan {plan['plan']}: task start does not match the plan's frame index")
        data = np.load(run/f"plan-{plan['plan']:03d}.npz")
        root_pos, root_rot, wrist_pos, wrist_rot = decode_plan(features, fk, frame, data['motion_features'])
        # The policy saved the same quantities; a mismatch means this decode is not the policy's.
        saved = max(np.abs(wrist_pos-data['projected_wrist_world']).max(),
                    np.abs(root_pos-data['raw_future_root_world']).max())
        if saved > 1e-5:
            raise ValueError(f"{run.name} plan {plan['plan']}: decoded plan differs from the saved one by {saved:.2e} m")
        now = measured_at(series, plan['time'])
        for horizon in horizons:
            k = int(round(horizon*30))
            actual = measured_at(series, start+(plan['cut']+k)/30)
            if actual is None or k >= len(root_pos):     # past the record, or past this plan's canvas
                continue
            pred = {'root_pos': root_pos[k], 'root_rot': root_rot[k],
                    'wrist_pos': wrist_pos[k], 'wrist_rot': wrist_rot[k]}
            offset = now['root_pos']-root_pos[0]
            aligned = dict(pred, root_pos=root_pos[k]+offset, wrist_pos=wrist_pos[k]+offset)
            hand_frames = sorted(set(plan['keyframes'])|set(plan['dense_hand_frames']))
            reaching = plan['time'] < target['arrival_time']-1e-9
            rows.append({'run': run.name, 'plan': plan['plan'], 'plan_time': plan['time'], 'horizon_s': horizon,
                         'constrained_frames': hand_frames, 'reaching': reaching,
                         'phase': plan['phase'] if moving else 'reaching' if reaching else 'holding',
                         # The standing runs condition the root on every future frame (support "root").
                         'root_covered': k in plan['root_frames'] if moving else k >= 1,
                         'wrist_covered': bool(hand_frames) and 1 <= k <= hand_frames[-1], 'wrist_sides': sides,
                         'kimodo': errors(pred, actual), 'aligned': errors(aligned, actual),
                         'persistence': errors(now, actual)})
    return rows


def summarize(rows):
    """Mean, 90th percentile and maximum per phase and horizon; wrists pool the conditioned hands."""
    def stats(values):
        v = np.asarray(values, dtype=float)
        if not v.size:
            return None
        return {'mean': float(v.mean()), 'p90': float(np.percentile(v, 90)), 'max': float(v.max()), 'n': int(v.size)}

    def phase(r):
        return r.get('phase') or ('reaching' if r['reaching'] else 'holding')

    def wrists(r, source, key):
        return [r[source][key][i] for i in r.get('wrist_sides', range(len(r[source][key])))]
    table = {}
    for name in ['all']+sorted({phase(r) for r in rows}, key=('reaching', 'holding', 'walk', 'stop', 'reach').index):
        for horizon in sorted({r['horizon_s'] for r in rows}):
            chosen = [r for r in rows if r['horizon_s'] == horizon and name in ('all', phase(r))]
            if chosen:
                rooted = [r for r in chosen if r.get('root_covered', True)]
                handed = [r for r in chosen if r.get('wrist_covered', True)]
                table[f'{name}@{horizon}'] = {source: {
                    'root_pos_cm': stats([r[source]['root_pos_cm'] for r in chosen]),
                    'root_rot_deg': stats([r[source]['root_rot_deg'] for r in chosen]),
                    'wrist_pos_cm': stats([x for r in chosen for x in wrists(r, source, 'wrist_pos_cm')]),
                    'wrist_rot_deg': stats([x for r in chosen for x in wrists(r, source, 'wrist_rot_deg')]),
                    'covered': {
                        'root_pos_cm': stats([r[source]['root_pos_cm'] for r in rooted]),
                        'root_rot_deg': stats([r[source]['root_rot_deg'] for r in rooted]),
                        'wrist_pos_cm': stats([x for r in handed for x in wrists(r, source, 'wrist_pos_cm')]),
                        'wrist_rot_deg': stats([x for r in handed for x in wrists(r, source, 'wrist_rot_deg')])}}
                    for source in SOURCES if source in chosen[0]}
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', default='settled-20261123-r1', help='every run named <suite>-<case> with saved plans')
    p.add_argument('--horizons', type=float, nargs='+', default=list(HORIZONS),
                   help='seconds ahead; a plan counts only for horizons its canvas reaches (native walks: up to 5 s)')
    p.add_argument('--out', help='summary file name under <runs>/<suite>/ (default prediction.json)')
    args = p.parse_args()
    runs = sorted(r for r in RUNS.glob(args.suite+'-*') if (r/'plans.jsonl').exists())
    if not runs:
        raise SystemExit(f'No runs with saved plans under {RUNS} for {args.suite}')
    features, fk = KimodoFeatures(50.), XmlFK(G1_XML, G1_JOINT_NAMES)
    rows = [row for run in runs for row in analyze_run(run, features, fk, args.horizons)]
    table = summarize(rows)
    cell = lambda s: '-' if s is None else f"{s['mean']:.2f}/{s['p90']:.2f}/{s['max']:.2f}"
    print(f'{len(runs)} runs, {len(rows)} plan-horizon pairs; each cell is mean/p90/max')
    print(f"{'phase@horizon':<16}{'source':<12}{'root cm':>18}{'root deg':>18}{'wrist cm':>18}{'wrist deg':>20}"
          f"{'covered wrist cm':>20}{'plans':>7}")
    for key, entry in table.items():
        for source, m in entry.items():
            print(f"{key:<16}{source:<12}{cell(m['root_pos_cm']):>18}{cell(m['root_rot_deg']):>18}"
                  f"{cell(m['wrist_pos_cm']):>18}{cell(m['wrist_rot_deg']):>20}"
                  f"{cell(m['covered']['wrist_pos_cm']):>20}{m['root_pos_cm']['n']:>7}")
    out = RUNS/args.suite/(args.out or 'prediction.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'runs': [r.name for r in runs], 'summary': table, 'rows': rows}, indent=1))
    print(out)


if __name__ == '__main__':
    main()
