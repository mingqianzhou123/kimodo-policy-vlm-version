"""Where the hammer and the robot start on DoubleDesk, and how a placed hammer is moved.

Pure logic, no simulator imports. The robot stands in front of the hammer table facing it
(``facing_deg``, world yaw) and never walks. Positions are given in the robot's frame:
``depth`` is how far the grip point on the handle lies behind the table's near edge,
``lateral`` how far it lies to the robot's left of the robot's root (negative: to the right),
``yaw`` the handle direction relative to the robot's left, degrees, positive turning the head
towards the robot. The robot stands at the stance the reach rule gives for that grip point,
plus ``extra`` further back.
"""
import numpy as np

from kimodo_vla.oracle.hammer_policy import grasp_pose


def axes(facing_deg):
    """Forward and left unit vectors (world x, y) of a robot facing ``facing_deg``."""
    f = np.radians(facing_deg)
    return np.array([np.cos(f), np.sin(f)]), np.array([-np.sin(f), np.cos(f)])


def yaw_quat(yaw_rad):
    return [float(np.cos(yaw_rad / 2)), 0., 0., float(np.sin(yaw_rad / 2))]


def heading_matrix(facing_deg):
    f = np.radians(facing_deg)
    c, s = np.cos(f), np.sin(f)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def hammer_pose(grip_xy, handle_yaw_deg, config):
    """Hammer root (x, y, z) and world yaw that put the grip point at ``grip_xy`` with the
    handle at ``handle_yaw_deg`` from the robot's left."""
    scene, policy = config['scene'], config['policy']
    forward, left = axes(scene['facing_deg'])
    a = np.radians(handle_yaw_deg)
    axis = np.cos(a) * left - np.sin(a) * forward
    # The handle runs along the hammer's local +y.
    yaw = float(np.arctan2(axis[1], axis[0]) - np.pi / 2)
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.asarray(policy['grip_point_local_m'][:2], dtype=float)
    root = np.asarray(grip_xy, dtype=float) - np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])
    return [float(root[0]), float(root[1]), float(scene['hammer_rest_z_m'])], yaw


def place(depth, lateral, yaw_deg, extra, config, robot_lateral=0.):
    """Hammer and robot start poses for one placement, with the quantities the analysis needs.

    The robot's root sits ``robot_lateral`` along the table edge from the edge reference point.
    Its distance from the edge follows the reach rule: the grasping wrist ends ``stance_reach_m``
    ahead of the root, but the root never comes nearer the edge than ``stance_clearance_m``.
    """
    scene, policy, stand = config['scene'], config['policy'], config['stand']
    forward, left = axes(scene['facing_deg'])
    edge = np.asarray(scene['table_edge_point_m'], dtype=float) + robot_lateral * left
    grip = edge + depth * forward + lateral * left
    root_xyz, yaw = hammer_pose(grip, yaw_deg, config)
    pose = grasp_pose(root_xyz, yaw_quat(yaw), heading_matrix(scene['facing_deg']), policy)
    wrist_ahead = float((pose['grip'][:2] - edge) @ forward)
    root_ahead = min(wrist_ahead - stand['stance_reach_m'], -stand['stance_clearance_m']) - extra
    robot = edge + root_ahead * forward
    return {'depth_m': float(depth), 'lateral_m': float(lateral), 'yaw_deg': float(yaw_deg), 'stance_extra_m': float(extra),
            'grip_xy': grip.tolist(), 'hammer_pos': root_xyz, 'hammer_yaw_rad': yaw,
            'robot_xy': robot.tolist(), 'robot_yaw_rad': float(np.radians(scene['facing_deg'])),
            'clearance_m': float(-root_ahead), 'wrist_ahead_of_root_m': wrist_ahead - root_ahead}


def map_cells(config):
    """Every cell of the reach map, in a fixed order."""
    m = config['stand']['map']
    return [dict(depth=d, lateral=l, yaw=y, extra=e) for e in m['stance_extra_m'] for y in m['yaw_deg']
            for d in m['depth_m'] for l in m['lateral_m']]


def cell_name(cell):
    """Run-name suffix of a map cell, e.g. d12-l-20-y+20-e00 (centimetres and degrees)."""
    cm = lambda v: int(round(v * 100))
    return f"d{cm(cell['depth']):02d}-l{cm(cell['lateral']):+03d}-y{int(round(cell['yaw'])):+03d}-e{cm(cell['extra']):02d}"


def in_region(seed, config):
    """A seeded placement inside the fixed region; draws do not depend on any condition."""
    stand = config['stand']
    region = stand.get('region')
    if not region:
        raise ValueError('The region has not been fixed in the configuration yet')
    rng = np.random.default_rng(seed)
    depth = rng.uniform(*region['depth_m'])
    lateral = rng.uniform(*region['lateral_m'])
    yaw = rng.uniform(*region['yaw_deg'])
    direction = int(rng.choice([-1, 1]))
    return {**place(depth, lateral, yaw, region['stance_extra_m'], config), 'direction': direction}


def footprint(hammer_xyz, yaw, config):
    """World (x, y) corners of the hammer's outline from above: the handle and the head, each a
    rectangle in the hammer's own frame (``hammer_footprint_local_m``)."""
    c, s = np.cos(yaw), np.sin(yaw)
    corners = np.array([[a, b] for (x0, y0), (x1, y1) in config['scene']['hammer_footprint_local_m']
                        for a in (x0, x1) for b in (y0, y1)])
    return corners @ np.array([[c, s], [-s, c]]) + np.asarray(hammer_xyz[:2])


def supported(hammer_xyz, yaw, config, table, margin=0.):
    """Every corner of the hammer's outline lies over the table top, ``margin`` inside its edges."""
    (x0, y0), (x1, y1) = table
    pts = footprint(hammer_xyz, yaw, config)
    return bool(np.all((pts[:, 0] >= x0 + margin) & (pts[:, 0] <= x1 - margin)
                       & (pts[:, 1] >= y0 + margin) & (pts[:, 1] <= y1 - margin)))


def placeable(cell, config):
    """Whether a map cell puts the whole hammer over the table (the table rectangle is in the config)."""
    lay = place(cell['depth'], cell['lateral'], cell['yaw'], cell['extra'], config)
    return supported(lay['hammer_pos'], lay['hammer_yaw_rad'], config, config['scene']['table_rect_m'])


def displaced(root_xyz, yaw, grip_xyz, axis, along=0., across=0., turn_deg=0.):
    """A hammer pose moved by a known amount: ``along`` the handle axis, ``across`` it
    (horizontal, positive away from the robot when the head points to the robot's left), and
    turned ``turn_deg`` about the vertical through the grip point. Returns (root xyz, yaw)."""
    axis = np.asarray(axis[:2], dtype=float) / np.linalg.norm(axis[:2])
    across_dir = np.array([axis[1], -axis[0]])
    t = np.radians(turn_deg)
    c, s = np.cos(t), np.sin(t)
    offset = np.asarray(root_xyz[:2], dtype=float) - np.asarray(grip_xyz[:2], dtype=float)
    turned = np.array([c * offset[0] - s * offset[1], s * offset[0] + c * offset[1]])
    xy = np.asarray(grip_xyz[:2]) + turned + along * axis + across * across_dir
    return [float(xy[0]), float(xy[1]), float(root_xyz[2])], float(yaw + t)


class LinearMotion:
    """Constant velocity along a horizontal unit direction, stopping at ``limit`` metres."""
    def __init__(self, start_xy, direction, speed, limit):
        self.start = np.asarray(start_xy, dtype=float)
        d = np.asarray(direction, dtype=float)
        self.direction = d / np.linalg.norm(d)
        if speed < 0 or limit < 0:
            raise ValueError('Speed and travel limit must be non-negative')
        self.speed, self.limit = float(speed), float(limit)

    def travelled(self, elapsed):
        return min(self.speed * max(float(elapsed), 0.), self.limit)

    def position(self, elapsed):
        return self.start + self.travelled(elapsed) * self.direction

    def velocity(self, elapsed):
        moving = self.speed > 0 and self.speed * max(float(elapsed), 0.) < self.limit
        return self.direction * (self.speed if moving else 0.)
