"""Back-projection error of a known point on the box, with the live and with the stale camera pose.

    python check_camera.py RUN [RUN ...]

For every frame kept in a run's ``frames/`` folder, the simulator's box pose at that time gives
the centre of the top face; it is projected with the frame's camera pose, the depth image is
read there, and the pixel is lifted back to 3D. With a correct pose the two points agree to the
depth image's precision. The same is done with the pose IsaacLab reports (``*_usd``), which
stays at the spawn pose. Frames where the point is hidden (the depth there is nearer than the
point) are counted apart.
"""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'scripts'))
sys.path.insert(0, str(HERE.parents[1]))
from paths import RUNS   # noqa: E402
from arena.camera_pose import backproject, project   # noqa: E402
from arena.frame_stream import box_points   # noqa: E402
from kimodo_vla.vlm.box_tracker import depth_at   # noqa: E402

HALF = .105


def check(run):
    steps = [json.loads(l) for l in (run/'grasp-steps.jsonl').read_text().splitlines()]
    times = np.array([s['time'] for s in steps])
    out = {'live': [], 'usd': [], 'hidden': 0, 'frames': 0, 'camera_moved_m': []}
    first = None
    for path in sorted((run/'frames').glob('[0-9]*.npz')):
        with np.load(path) as f:
            frame = {k: f[k] for k in f.files}
        i = int(np.argmin(np.abs(times - float(frame['time']))))
        if abs(times[i] - float(frame['time'])) > .03:
            continue
        top = box_points(steps[i]['box_com'], steps[i]['box_quat'], HALF)[:1]
        out['frames'] += 1
        first = frame['camera_position'] if first is None else first
        out['camera_moved_m'].append(float(np.linalg.norm(frame['camera_position'] - first)))
        uv, z = project(top, frame['camera_position'], frame['camera_rotation'], frame['K'])
        d = depth_at(frame['depth'], uv)
        if not np.isfinite(d[0]) or d[0] < z[0] - .02:
            out['hidden'] += 1
            continue
        out['live'].append(float(np.linalg.norm(backproject(uv, d, frame['camera_position'], frame['camera_rotation'],
                                                            frame['K'])[0] - top[0])))
        if 'camera_position_usd' in frame:
            uv_usd, _ = project(top, frame['camera_position_usd'], frame['camera_rotation_usd'], frame['K'])
            d_usd = depth_at(frame['depth'], uv)   # the image is the true one; only the pose differs
            out['usd'].append(float(np.linalg.norm(backproject(uv, d_usd, frame['camera_position_usd'],
                                                               frame['camera_rotation_usd'], frame['K'])[0] - top[0])))
    summary = lambda v: {'n': len(v), 'median_cm': round(100 * float(np.median(v)), 2), 'max_cm': round(100 * float(np.max(v)), 2)} if v else None
    return {'run': run.name, 'frames': out['frames'], 'hidden': out['hidden'], 'live_pose': summary(out['live']),
            'usd_pose': summary(out['usd']), 'camera_moved_max_cm': round(100 * max(out['camera_moved_m'] or [0]), 2)}


if __name__ == '__main__':
    print(json.dumps([check(RUNS/name) for name in sys.argv[1:]], indent=1))
