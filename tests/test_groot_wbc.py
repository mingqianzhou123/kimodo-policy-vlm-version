"""GR00T base commands and pelvis paths invert each other (no simulator)."""

import numpy as np

from kimodo_vla.controllers.groot_wbc import BaseCommand, BaseCommander, pelvis_path
from kimodo_vla.core.types import ActionChunk, yaw_matrix, yaw_of

FPS = 50.0


def random_commands(steps=400, seed=0):
    """Smooth walking and turning; the heading crosses +-pi on the way."""
    t = np.arange(steps) / FPS
    rng = np.random.default_rng(seed)
    a, b, c = rng.uniform(0, 2 * np.pi, 3)
    return BaseCommand(
        vx=(0.3 * np.sin(0.7 * t + a)).astype(np.float32),
        vy=(0.1 * np.sin(1.1 * t + b)).astype(np.float32),
        yaw_rate=(0.8 + 0.5 * np.sin(0.9 * t + c)).astype(np.float32),
        height=(0.75 + 0.03 * np.sin(0.5 * t)).astype(np.float32),
    )


def chunk_at(root_pos, root_rot, step):
    """Frame 0 = pose ``step``, its delta from pose ``step - 1`` in its own pelvis frame (as planners emit)."""
    delta = root_rot[step].T @ (root_pos[step] - root_pos[step - 1])
    return ActionChunk(
        root_delta_xy=delta[None, :2].astype(np.float32),
        root_height=root_pos[step : step + 1, 2],
        root_rot=root_rot[step : step + 1],
        joint_pos=np.zeros((1, 29), np.float32),
        hand=np.zeros((1, 2), np.float32),
        fps=FPS,
    )


def commands_along(root_pos, root_rot, yaw_gain=0.0):
    """Run the online commander along a path, the robot tracking it perfectly."""
    commander = BaseCommander(yaw_gain=yaw_gain)
    commander.reset()
    rows = [commander(chunk_at(root_pos, root_rot, t), root_rot[t - 1]) for t in range(1, len(root_pos))]
    return BaseCommand(*(np.asarray(values, np.float64) for values in zip(*rows)))


def test_commands_to_path_and_back():
    commands = random_commands()
    start_yaw = 2.9
    root_pos, root_rot = pelvis_path(commands, np.array([0.5, -1.0, 0.0], np.float32), start_yaw, FPS)
    assert abs(yaw_of(root_rot[0]) - start_yaw) < 1e-6
    assert np.ptp(np.unwrap(yaw_of(root_rot))) > np.pi  # the heading wraps around
    recovered = commands_along(root_pos, root_rot)
    for name in BaseCommand._fields:
        np.testing.assert_allclose(getattr(recovered, name), getattr(commands, name)[1:], atol=2e-4, err_msg=name)


def test_path_to_commands_and_back():
    root_pos, root_rot = pelvis_path(random_commands(seed=1), np.zeros(3, np.float32), -0.4, FPS)
    commands = commands_along(root_pos, root_rot)
    # Row 0 of a command sequence is the start pose; any value works there.
    padded = BaseCommand(*(np.concatenate([[0.0], c]).astype(np.float32) for c in commands))
    padded = padded._replace(height=np.concatenate([[root_pos[0, 2]], commands.height]).astype(np.float32))
    pos, rot = pelvis_path(padded, root_pos[0], float(yaw_of(root_rot[0])), FPS)
    np.testing.assert_allclose(pos, root_pos, atol=1e-4)
    np.testing.assert_allclose(rot, root_rot, atol=1e-5)


def test_heading_is_feed_forward():
    """With the default gain, a measured heading error does not change the command after the first step."""
    root_pos, root_rot = pelvis_path(random_commands(steps=20), np.zeros(3, np.float32), 0.0, FPS)
    commander = BaseCommander()
    commander(chunk_at(root_pos, root_rot, 1), root_rot[0])
    on_track = commander(chunk_at(root_pos, root_rot, 2), root_rot[1])
    commander.reset()
    commander(chunk_at(root_pos, root_rot, 1), root_rot[0])
    drifted = commander(chunk_at(root_pos, root_rot, 2), yaw_matrix(0.3) @ root_rot[1])
    assert drifted.yaw_rate == on_track.yaw_rate
