"""Where the box is now, from two points tracked in the head camera (experiment four, part two).

The simulator writes one RGB-D frame every few control steps with the camera's live pose
(``agentic/arena/frame_stream.py``), and once, the pixels of two points on the box in the
first frame: the centre of its top face (the contact point) and the middle of the top face's
edge nearest the robot (the pre-grasp point). That first answer stands in for the model's
pointing, which experiment three measured on its own. From then on nothing about the box
comes from the simulator: an off-the-shelf online point tracker (CoTracker3, no training)
follows the two pixels, each is lifted to 3D with the frame's depth, and the box pose is
read off them. The estimate is moved to the current time with the points' recent velocity.
"""
import json
import time
from pathlib import Path

import numpy as np

UP = np.array([0., 0., 1.])


def backproject(uv, depth, position, rotation, K):
    rays = np.c_[np.asarray(uv, dtype=float), np.ones(len(uv))] @ np.linalg.inv(np.asarray(K, dtype=float)).T
    return (rays * np.asarray(depth, dtype=float)[:, None]) @ np.asarray(rotation).T + np.asarray(position)


def depth_at(depth, uv, radius=1):
    """Median depth in a small patch around each pixel; NaN outside the image or without depth."""
    h, w = depth.shape
    out = []
    for u, v in np.asarray(uv, dtype=float):
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < w and 0 <= y < h):
            out.append(np.nan)
            continue
        patch = depth[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1].astype(float)
        patch = patch[np.isfinite(patch) & (patch > 0)]
        out.append(float(np.median(patch)) if patch.size else np.nan)
    return np.array(out)


def box_pose(top, edge, half):
    """Centre of mass and a yaw quaternion (w, x, y, z) from the top-face centre and the middle of
    its edge nearest the robot: the edge lies half a side behind the centre along the face's -x."""
    heading = np.asarray(top, dtype=float) - np.asarray(edge, dtype=float)
    yaw = float(np.arctan2(heading[1], heading[0]))
    com = np.asarray(top, dtype=float) - half * UP
    return com, [float(np.cos(yaw / 2)), 0., 0., float(np.sin(yaw / 2))]


def extrapolate(times, points, now, window):
    """Points [N, 3] moved from the last time to ``now`` by a straight-line fit over the last
    ``window`` samples; the velocity [N, 3] is returned too."""
    t, p = np.asarray(times[-window:], dtype=float), np.asarray(points[-window:], dtype=float)
    if len(t) < 2 or t[-1] - t[0] <= 0:
        return p[-1], np.zeros_like(p[-1])
    centred = t - t.mean()
    velocity = np.einsum('t,tnd->nd', centred, p - p.mean(0)) / (centred @ centred)
    return p[-1] + velocity * (now - t[-1]), velocity


class BoxPointTracker:
    """Frames are files ``frames/<simulation time in ms>.npz``; ``queries.json`` names the first one."""
    def __init__(self, run, half, checkpoint, device='cuda', velocity_window=6, keep_every_ms=1000):
        self.frames = Path(run)/'frames'
        self.log_path = Path(run)/'tracker.jsonl'
        self.half, self.device, self.window, self.keep_every = float(half), device, int(velocity_window), int(keep_every_ms)
        import torch
        from cotracker.predictor import CoTrackerOnlinePredictor
        self.torch = torch
        self.model = CoTrackerOnlinePredictor(checkpoint=str(checkpoint)).to(device)
        self.reset(None)

    def reset(self, seed):
        self.query = None
        self.window_rgb, self.metas, self.seen = [], [], set()
        self.track_times, self.track_points, self.track_uv, self.track_visible = [], [], [], []
        self.processed, self.initialised = 0, False
        self.compute_s, self.calls = 0., 0
        self.last = None

    def observe(self, samples):
        """Frames are read from disk when an estimate is asked for; nothing to do per sample."""

    def load(self, ms):
        path = self.frames/f'{ms:06d}.npz'
        with np.load(path) as f:
            frame = {k: f[k] for k in f.files}
        if (ms - self.query['ms']) % self.keep_every:
            path.unlink(missing_ok=True)   # 1.5 MB each; one per keep_every_ms stays for the record
        return frame

    def catch_up(self, now):
        """Feed every frame taken up to ``now`` to the tracker, one half-window at a time.

        CoTracker3 online takes the last two half-windows each time a half-window has filled;
        its first call only registers the query points, so the newest tracked frame trails the
        newest frame by one to two half-windows.
        """
        torch = self.torch
        if self.query is None:
            self.query = json.loads((self.frames/'queries.json').read_text())
        k = self.model.step
        names = sorted(int(p.stem) for p in self.frames.glob('[0-9]*.npz'))
        for ms in names:
            if ms in self.seen or ms < self.query['ms']:
                continue
            if ms / 1000 > now + 1e-4:
                break
            frame = self.load(ms)
            self.seen.add(ms)
            self.window_rgb.append(frame['rgb'])
            self.metas.append({'time': float(frame['time']), 'depth': frame['depth'], 'K': frame['K'],
                               'position': frame['camera_position'], 'rotation': frame['camera_rotation']})
            i = len(self.metas) - 1
            if i == 0 or i % k:
                continue
            begin = time.perf_counter()
            chunk = torch.tensor(np.stack(self.window_rgb[-2 * k:]), device=self.device).float().permute(0, 3, 1, 2)[None]
            if not self.initialised:
                queries = torch.tensor([[0., *uv] for uv in self.query['points_uv']], device=self.device)[None]
                self.model(video_chunk=chunk, is_first_step=True, queries=queries, grid_size=0)
                self.initialised = True
            else:
                tracks, visible = self.model(video_chunk=chunk)
                tracks, visible = tracks[0].cpu().numpy(), visible[0].cpu().numpy()
                done = min(len(tracks), i)
                for j in range(self.processed, done):
                    self.add_track(j, tracks[j], visible[j])
                self.processed = done
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.compute_s += time.perf_counter() - begin
            self.calls += 1
            del self.window_rgb[:-2 * k]
        return self.processed

    def add_track(self, j, uv, visible):
        m = self.metas[j]
        d = depth_at(m.pop('depth'), uv)
        self.track_times.append(m['time'])
        self.track_points.append(backproject(uv, d, m['position'], m['rotation'], m['K']))
        self.track_uv.append(np.asarray(uv, dtype=float))
        self.track_visible.append(np.asarray(visible, dtype=bool))

    def estimate(self, now):
        """(centre of mass, quaternion, time of the newest frame used), or the last estimate when
        the tracker has nothing new; the record keeps which."""
        self.catch_up(now)
        good = [i for i, (p, v) in enumerate(zip(self.track_points, self.track_visible)) if np.isfinite(p).all() and v.all()]
        row = {'now': now, 'frames': len(self.metas), 'tracked': self.processed, 'compute_s': round(self.compute_s, 4),
               'calls': self.calls}
        if self.track_times:
            row.update(frame_time=self.track_times[-1], uv=self.track_uv[-1].tolist(),
                       visible=self.track_visible[-1].tolist(), points=np.nan_to_num(self.track_points[-1]).tolist())
        if not good:
            row['status'] = 'no_track_yet' if not self.track_times else 'never_seen'
            if self.last is None:
                # Nothing tracked yet: the first answer's pixels, lifted with the first frame's depth.
                m = self.metas[0] if self.metas else None
                if m is None or 'depth' not in m:
                    raise RuntimeError('No frame to start the tracker from')
                points = backproject(self.query['points_uv'], depth_at(m['depth'], self.query['points_uv']),
                                     m['position'], m['rotation'], m['K'])
                com, quat = box_pose(points[0], points[1], self.half)
                self.last = (com.tolist(), quat, m['time'])
                row['status'] = 'first_frame'
            self.write({**row, 'com': self.last[0], 'used_time': self.last[2]})
            return self.last
        row['status'] = 'tracked' if good[-1] == len(self.track_times) - 1 else 'lost'
        times = [self.track_times[i] for i in good]
        # a point that has been lost coasts on its last velocity for at most half a second, then stays put
        current, velocity = extrapolate(times, [self.track_points[i] for i in good], min(now, times[-1] + .5), self.window)
        com, quat = box_pose(current[0], current[1], self.half)
        self.last = (com.tolist(), quat, times[-1])
        self.write({**row, 'used_time': times[-1], 'velocity': velocity.tolist(), 'com': com.tolist(), 'quat': quat})
        return self.last

    def write(self, row):
        with self.log_path.open('a') as f:
            f.write(json.dumps(row) + '\n')
