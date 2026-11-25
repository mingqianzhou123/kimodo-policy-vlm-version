"""The walk of a grasp round done the official Kimodo way (``LocomotionPolicy(generator='native')``).

Off by default: ``BoxGraspPolicy`` walks with its own whole clips unless it is given
``native_walk`` settings. With them, the walk of every round is handed to a
``LocomotionPolicy`` with the native generator, the setting experiment two validated on
2026-11-25 (1 s replans of 5 s segments from the measured position, 5 transition frames,
approach gain 2, no text, 100 DDIM steps): walk to within the goal tolerance, then one stop
plan into the standing pose, then hand back to the grasp, which settles and reaches.

The walker shares the grasp policy's loaded model. Its logs (``plans.jsonl``,
``commands.jsonl``, ``plan-*.npz``, ``target.json``) go to a folder of their own per round,
``walk-r<round>/`` in the run folder, so the grasp's own records keep their format.
"""
from pathlib import Path

import numpy as np

from .locomotion import LocomotionPolicy, load_goal
from .policy import OraclePolicy

# Experiment two's validated native setting (native-generation-20261125.md); a config may override.
DEFAULTS = {'walk_steps': 50, 'segment_seconds': 5., 'segment_steps': 100, 'transition_frames': 5,
            'approach_gain': 2., 'speed_m_s': .35, 'tolerance_m': .05, 'prime_samples': 30}


class _EpisodeHeading(OraclePolicy):
    """Arena decodes root rotations relative to the episode's first heading, so a walker that starts
    mid-episode must use that heading for its frame, not the one measured when it starts."""
    fixed_heading = None

    def initialize(self, sample, heading=None):
        super().initialize(sample, heading=self.fixed_heading if heading is None else heading)


class NativeApproach(LocomotionPolicy, _EpisodeHeading):
    """``LocomotionPolicy`` whose frame keeps the grasp episode's heading."""


def walk_goal_xy(start_root, goal_xy, heading):
    """Root goal [forward, left] of the walker: ``goal_xy`` (world) seen from ``start_root`` in the
    frame of ``heading`` (3x3, the episode heading)."""
    offset = np.r_[np.asarray(goal_xy, dtype=float) - np.asarray(start_root, dtype=float)[:2], 0.]
    return (np.asarray(heading).T @ offset)[:2]


def settings(overrides=None):
    out = dict(DEFAULTS)
    unknown = set(overrides or {}) - set(DEFAULTS) - {'_about'}
    if unknown:
        raise ValueError(f'Unknown native walk settings: {sorted(unknown)}')
    out.update({k: v for k, v in (overrides or {}).items() if k != '_about'})
    return out


def make_walker(host, text_cache, overrides=None):
    """One walker for a grasp policy ``host``, sharing its model; goals are set per round by ``start``."""
    s = settings(overrides)
    denoiser = host.model.denoiser   # the walker re-wraps the shared model's guidance; keep the host's
    try:
        walker = NativeApproach(None, text_cache, host.run, 'hold', model=host.model, generator='native',
            motion_goal={'name': 'grasp-approach', 'root': {'xy': [.1, 0.]}},   # replaced by start() every round
            walk_steps=s['walk_steps'], segment_seconds=s['segment_seconds'], segment_steps=s['segment_steps'],
            transition_frames=s['transition_frames'], approach_gain=s['approach_gain'],
            constraint_cfg=host.constraint_cfg, variant='bounded', handoff_seconds=host.handoff_seconds,
            joint_speed=host.joint_speed)
    finally:
        host.model.denoiser = denoiser
    walker.native_settings = s
    return walker


def start(walker, *, run, goal_world, heading, history, seed):
    """Point ``walker`` at ``goal_world`` (x, y) for one round and prime it with the recent measured
    ``history`` (it continues the reference SONIC is on). Returns the goal in the walker's frame."""
    s = walker.native_settings
    primed = list(history)[-s['prime_samples']:]
    xy = walk_goal_xy(primed[0]['root_pos'], goal_world, heading)
    walker.goal = load_goal({'name': 'grasp-approach', 'root': {'xy': xy.tolist(), 'speed_m_s': s['speed_m_s'],
                                                                'tolerance_m': s['tolerance_m']}})
    walker.run = Path(run)
    walker.run.mkdir(parents=True, exist_ok=True)
    walker.fixed_heading = np.asarray(heading, dtype=float)
    walker.noise_seed = seed
    walker.reset(seed)
    walker.primed = primed
    return xy
