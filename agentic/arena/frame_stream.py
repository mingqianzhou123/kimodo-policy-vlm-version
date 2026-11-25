"""Head-camera RGB-D frames for an online tracker (experiment four, part two). Off unless
KVLA_FRAME_STREAM is set to a number of control steps between frames.

Each frame is ``<run>/frames/<simulation time in ms>.npz`` with the image, the depth along the
optical axis, the intrinsics and the camera's live pose (``camera_pose.LiveCamera``). The
first frame also gets ``queries.json``: the pixels of the two points the tracker is asked to
follow, projected from the simulator's box pose at that moment. That stands in for the
model's single pointing answer; nothing else about the box leaves the simulator this way.
"""
import json
from pathlib import Path

import numpy as np

from arena.camera_pose import LiveCamera, project
from kimodo_vla.oracle.geometry import rotation_wxyz
from kimodo_vla.oracle.grasp_policy import face_yaw


def box_points(com, quat_wxyz, half):
    """Centre of the top face and the middle of the top face's edge nearest the robot (-x side)."""
    yaw = face_yaw(quat_wxyz)
    com = np.asarray(com, dtype=float)
    top = com + [0., 0., half]
    edge = top + half * np.array([-np.cos(yaw), -np.sin(yaw), 0.])
    return np.stack([top, edge])


class FrameStream:
    def __init__(self, env, run, every_steps, half):
        self.camera, self.half = LiveCamera(env), float(half)
        self.out = Path(run)/'frames'
        self.out.mkdir(exist_ok=True)
        self.every, self.count, self.queried = int(every_steps), 0, False

    def __call__(self, sample):
        """Called once per control step with the measured state before the step."""
        if self.count % self.every == 0:
            self.write(sample)
        self.count += 1

    def write(self, sample):
        f = self.camera.frame()
        ms = int(round(sample['time'] * 1000))
        np.savez(self.out/f'{ms:06d}.npz', rgb=f['rgb'].astype(np.uint8), depth=f['depth'].astype(np.float32), K=f['K'],
                 camera_position=f['camera_position'], camera_rotation=f['camera_rotation'], time=sample['time'])
        if not self.queried:
            points = box_points(sample['box_com'], sample['box_quat'], self.half)
            uv, depth = project(points, f['camera_position'], f['camera_rotation'], f['K'])
            (self.out/'queries.json').write_text(json.dumps({'ms': ms, 'time': sample['time'],
                'points_uv': uv.tolist(), 'points_depth': depth.tolist(), 'points_world': points.tolist(),
                'names': ['top_centre', 'near_edge_middle'], 'camera_link': f['link'],
                'image_shape': list(f['rgb'].shape)}, indent=1))
            self.queried = True
