"""How far the tracker's box estimate is from the simulator's truth, per replan and at the close.

    python check_tracker.py RUN [RUN ...]

Each line of ``tracker.jsonl`` is one estimate the policy used. It is compared with the box's
measured centre at the same time (``grasp-steps.jsonl``). An estimate whose tracked top point
sits more than 3 cm above or below the box's true top face is counted as "on the background"
(the depth there belongs to something else); an estimate made while either point was not
visible, or from an older frame because the newest one was not usable, as "lost".
"""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'scripts'))
from paths import RUNS   # noqa: E402

HALF, OFF_TOP_M = .105, .03


def rows_of(run):
    steps = [json.loads(l) for l in (run/'grasp-steps.jsonl').read_text().splitlines()]
    t = np.array([s['time'] for s in steps])
    box = np.array([s['box_com'] for s in steps])
    out = []
    for line in (run/'tracker.jsonl').read_text().splitlines():
        r = json.loads(line)
        if 'com' not in r:
            continue
        i = int(np.argmin(np.abs(t - r['now'])))
        error = np.asarray(r['com']) - box[i]
        top_z = r['points'][0][2] if r.get('points') else np.nan
        raw = np.nan
        if r.get('points') and r.get('frame_time') is not None:
            # the newest tracked top point against the true top centre at that frame: tracking and depth only
            j = int(np.argmin(np.abs(t - r['frame_time'])))
            raw = 100 * float(np.linalg.norm(np.asarray(r['points'][0][:2]) - box[j][:2]))
        out.append({'now': r['now'], 'status': r['status'], 'age_s': r['now'] - r['used_time'],
                    'lateral_cm': 100 * float(error[1]), 'planar_cm': 100 * float(np.linalg.norm(error[:2])),
                    'raw_top_planar_cm': raw, 'background': bool(abs(top_z - (box[i][2] + HALF)) > OFF_TOP_M),
                    'visible': r.get('visible'), 'compute_s': r.get('compute_s'), 'tracked': r.get('tracked'),
                    'calls': r.get('calls')})
    return out


def summary(run):
    rows = rows_of(run)
    if not rows:
        return {'run': run.name, 'estimates': 0}
    planar = np.array([r['planar_cm'] for r in rows])
    last = rows[-1]
    return {'run': run.name, 'estimates': len(rows), 'planar_median_cm': round(float(np.median(planar)), 2),
            'planar_max_cm': round(float(planar.max()), 2),
            'lateral_abs_median_cm': round(float(np.median([abs(r['lateral_cm']) for r in rows])), 2),
            'raw_top_planar_median_cm': round(float(np.nanmedian([r['raw_top_planar_cm'] for r in rows])), 2),
            'status': {s: sum(r['status'] == s for r in rows) for s in {r['status'] for r in rows}},
            'background': sum(r['background'] for r in rows), 'age_median_s': round(float(np.median([r['age_s'] for r in rows])), 3),
            'tracker_fps': round(last['tracked'] / last['compute_s'], 1) if last.get('compute_s') else None,
            'rows': rows}


if __name__ == '__main__':
    print(json.dumps([summary(RUNS/n) for n in sys.argv[1:]], indent=1))
