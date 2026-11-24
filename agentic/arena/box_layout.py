"""Where the box and the robot start when the robot never walks, and how a driven box moves.

Pure logic, no simulator imports. Depth is the box centre's distance behind the table's near
edge, lateral its offset to the robot's left (+y, the robot faces +x). The robot stands at the
grasp stance the policy would walk to for that depth, plus ``stance_extra_m`` further back.
"""
import numpy as np

from kimodo_vla.oracle.grasp_policy import stance


def place(depth, lateral, yaw_deg, extra, config):
    """Box centre (x, y), its yaw and the robot root (x, y) for one placement."""
    policy, stand = config['policy'], config['stand']
    box = np.array([config['scene']['table_edge_x_m'] + depth, stand['robot_y_m'] + lateral])
    robot = np.array([stance(box, policy)[0] - extra, stand['robot_y_m']])
    return {'depth_m': float(depth), 'lateral_m': float(lateral), 'yaw_deg': float(yaw_deg), 'stance_extra_m': float(extra),
            'box_xy': box.tolist(), 'robot_xy': robot.tolist(),
            'clearance_m': float(config['scene']['table_edge_x_m'] - robot[0])}


def map_cells(config):
    """Every cell of the workspace map, in a fixed order."""
    m = config['stand']['map']
    return [dict(depth=d, lateral=l, yaw=y, extra=e) for e in m['stance_extra_m'] for y in m['yaw_deg']
            for d in m['depth_m'] for l in m['lateral_m']]


def cell_name(cell):
    """Run-name suffix of a map cell, e.g. d16-l+10-y40-e05 (centimetres and degrees)."""
    cm = lambda v: int(round(v * 100))
    return f"d{cm(cell['depth']):02d}-l{cm(cell['lateral']):+03d}-y{int(round(cell['yaw'])):02d}-e{cm(cell['extra']):02d}"


def in_region(seed, config):
    """A seeded placement inside the fixed region, turned within the full yaw range.

    The draws do not depend on the speed or the delay, so every condition gets the same box
    start, yaw and direction of motion for the same seed.
    """
    stand = config['stand']
    region = stand.get('region')
    if not region:
        raise ValueError('The region has not been fixed in the configuration yet')
    rng = np.random.default_rng(seed)
    depth = rng.uniform(*region['depth_m'])
    lateral = rng.uniform(*region['lateral_m'])
    yaw = rng.uniform(-stand['max_yaw_deg'], stand['max_yaw_deg'])
    direction = int(rng.choice([-1, 1]))
    return {**place(depth, lateral, yaw, stand['stance_extra_m'], config), 'direction': direction,
            'segment_y': [stand['robot_y_m'] + v for v in region['lateral_m']]}


def supported(box_xy, yaw_deg, half, table):
    """The centre of mass is over the table top and every corner stays above it sideways."""
    c, s = abs(np.cos(np.radians(yaw_deg))), abs(np.sin(np.radians(yaw_deg)))
    reach = half * (c + s)
    (x0, y0), (x1, y1) = table
    return bool(x0 < box_xy[0] < x1 and y0 + reach <= box_xy[1] <= y1 - reach)


class BoxMotion:
    """Back and forth along y between two ends at constant speed, turning at each end.

    ``u`` unfolds the motion: it grows at ``speed`` and the box is at ``lo + u`` going up for
    ``u`` in [0, L), at ``lo + 2L - u`` going down for ``u`` in [L, 2L), with L the segment length.
    """
    def __init__(self, segment, speed, start, direction):
        self.lo, self.hi = map(float, segment)
        if not self.lo < self.hi:
            raise ValueError('Empty segment')
        if speed < 0 or direction not in (-1, 1):
            raise ValueError('Speed must be non-negative and direction +1 or -1')
        self.speed, self.length = float(speed), self.hi - self.lo
        start = float(np.clip(start, self.lo, self.hi))
        self.u0 = start - self.lo if direction > 0 else 2 * self.length - (start - self.lo)

    def _u(self, elapsed):
        return self.u0 + self.speed * max(float(elapsed), 0.)

    def position(self, elapsed):
        u = self._u(elapsed) % (2 * self.length)
        return self.lo + u if u < self.length else self.lo + 2 * self.length - u

    def velocity(self, elapsed):
        if self.speed == 0:
            return 0.
        u = self._u(elapsed) % (2 * self.length)
        return self.speed if u < self.length else -self.speed

    def turns(self, start, end):
        """How many times the box turned around between two elapsed times."""
        if self.speed == 0 or end <= start:
            return 0
        return int(np.floor(self._u(end) / self.length) - np.floor(self._u(start) / self.length))
