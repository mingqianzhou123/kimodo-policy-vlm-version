"""Clock-fixed, feasible oracle path; only supplies goals, never policy actions."""
import numpy as np


def joint_goal_path(initial, goal, elapsed, duration=1.2):
    u = np.clip(np.asarray(elapsed, dtype=float) / duration, 0., 1.)
    w = u**3 * (10 + u * (-15 + 6*u))
    return np.asarray(initial) + w[..., None] * (np.asarray(goal)-initial)


def execution_frames(step, cut, actions=25):
    """30 Hz canvas samples covering the upcoming 50 Hz command interval.

    Frame zero belongs to measured history and must not be overwritten.
    Include the right interpolation endpoint if the grids are out of phase.
    """
    end = step * .6 - cut + actions * .6
    return list(range(1, int(np.ceil(end - 1e-8)) + 1))
