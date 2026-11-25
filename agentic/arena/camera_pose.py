"""Camera pose that follows the robot.

IsaacLab reads a camera's pose from the scene description (USD), which physics never writes
back: ``camera.data.pos_w`` stays at the spawn pose however the robot moves (found in experiment
three). The camera hangs under a robot link, so its true pose is that link's live physics pose
times the fixed offset between the two, and that offset can be read from the scene description,
where both are stale in the same way.
"""
import numpy as np


def mount(link_world, camera_world):
    """Fixed offset of the camera in its link's frame, from two 4x4 world transforms of one time."""
    link_world, camera_world = np.asarray(link_world, dtype=float), np.asarray(camera_world, dtype=float)
    return np.linalg.inv(link_world) @ camera_world


def compose(position, rotation, offset):
    """World position [3] and rotation [3, 3] of the camera from its link's pose and the offset."""
    link = np.eye(4)
    link[:3, :3], link[:3, 3] = rotation, position
    world = link @ offset
    return world[:3, 3], world[:3, :3]


def transform(position, rotation):
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rotation, position
    return out


def project(points, position, rotation, K):
    """Pixels (u, v) and depth along the optical axis of world points [N, 3], for a camera whose
    rotation maps optical axes (ROS: x right, y down, z forward) to world axes."""
    local = (np.asarray(points, dtype=float) - position) @ rotation
    uv = local @ np.asarray(K, dtype=float).T
    return uv[:, :2] / uv[:, 2:3], local[:, 2]


def backproject(uv, depth, position, rotation, K):
    """World points from pixels [N, 2] and their depth along the optical axis [N]."""
    uv, depth = np.asarray(uv, dtype=float), np.asarray(depth, dtype=float)
    rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(np.asarray(K, dtype=float)).T
    return (rays * depth[:, None]) @ rotation.T + position


class LiveCamera:
    """Simulator side: the camera's live pose and one RGB-D frame. Needs a running IsaacLab scene."""
    def __init__(self, env, camera_name='front_camera'):
        from pxr import Usd, UsdGeom
        from kimodo_vla.oracle.geometry import rotation_wxyz
        self.env, self.camera = env, env.scene[camera_name]
        robot = env.scene['robot']
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        self.camera.cfg.update_latest_camera_pose = True
        self.camera.update(0., force_recompute=True)
        prim = stage.GetPrimAtPath(self.camera.cfg.prim_path.replace('env_.*', 'env_0'))
        names = list(robot.body_names)
        link = prim.GetParent()
        while link and link.GetName() not in names:
            link = link.GetParent()
        if not link:
            raise ValueError(f'No robot link above {prim.GetPath()}')
        self.link_name, self.link_id = link.GetName(), names.index(link.GetName())
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        link_usd = np.array(cache.GetLocalToWorldTransform(link)).T      # USD stores row vectors
        link_usd[:3, :3] /= np.linalg.norm(link_usd[:3, :3], axis=0, keepdims=True)
        data = self.camera.data
        camera_usd = transform(data.pos_w[0].detach().cpu().numpy(),
                               rotation_wxyz(data.quat_w_ros[0].detach().cpu().numpy()))
        self.offset = mount(link_usd, camera_usd)
        self.K = data.intrinsic_matrices[0].detach().cpu().numpy()
        self.rotation_wxyz = rotation_wxyz

    def pose(self):
        data = self.env.scene['robot'].data
        position = data.body_pos_w[0, self.link_id].detach().cpu().numpy()
        rotation = self.rotation_wxyz(data.body_quat_w[0, self.link_id].detach().cpu().numpy())
        return compose(position, rotation, self.offset)

    def frame(self):
        out = self.camera.data.output
        rgb = out['rgb'][0].detach().cpu().numpy()[..., :3]
        depth = out['distance_to_image_plane'][0].detach().cpu().numpy().squeeze(-1)
        position, rotation = self.pose()
        data = self.camera.data
        return {'rgb': rgb, 'depth': depth, 'K': self.K, 'camera_position': position, 'camera_rotation': rotation,
                'link': self.link_name, 'camera_position_usd': data.pos_w[0].detach().cpu().numpy(),
                'camera_rotation_usd': self.rotation_wxyz(data.quat_w_ros[0].detach().cpu().numpy())}
