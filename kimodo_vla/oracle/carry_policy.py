"""Carry the box after the blocking grasp: the carrying condition of experiment 2's command test (2026-11-26).

Stages, in order, on the measured clock:

  grasp    ``BoxGraspPolicy`` unchanged, until a round's hold phase has run its full time with the box
           at least ``lift_m`` above where it was latched. A round that ends without that lift goes on
           as the grasp policy would (release and retry).
  clear    a ``LocomotionPolicy`` walk ``clear_m`` straight back, away from the table, so that the
           measured command has room. Not measured.
  command  once the clearing walk has reached its hold phase and the root has stood still for
           ``still_steps`` control steps, a second ``LocomotionPolicy`` runs the measured goal.

In ``clear`` and ``command`` every sent frame keeps the arm joints of the reference SONIC was
consuming when the grasp ended (the squeezing pose), and the response tells the simulator side to
keep the hands as the gate left them (closed). Switch times go to ``carry.json`` in the run folder;
each locomotion stage keeps its own files in ``carry-clear/`` and ``carry-command/``.
"""
import json
from pathlib import Path

import numpy as np

from kimodo_vla.core.types import G1_JOINT_NAMES

ARM = [i for i, n in enumerate(G1_JOINT_NAMES) if any(k in n for k in ('shoulder', 'elbow', 'wrist'))]
STILL_SPEED_M_S, STILL_YAW_RAD_S = .05, np.radians(5.)


def yaw(quat_wxyz):
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def still(samples, steps):
    """Whether the root stood still (speed and yaw rate) over the last ``steps`` samples."""
    if len(samples) < steps+1:
        return False
    s = samples[-(steps+1):]
    dt = s[-1]['time']-s[0]['time']
    if dt <= 0:
        return False
    speed = np.linalg.norm(np.diff([x['root_pos'][:2] for x in s], axis=0), axis=1)/np.diff([x['time'] for x in s])
    turn = np.abs(np.diff(np.unwrap([yaw(x['root_quat']) for x in s])))/np.diff([x['time'] for x in s])
    return bool((speed < STILL_SPEED_M_S).all() and (turn < STILL_YAW_RAD_S).all())


class CarryPolicy:
    def __init__(self, grasp, make_locomotion, goal, run, clear_m=.6, lift_m=.1, still_steps=25):
        self.grasp, self.run = grasp, Path(run)
        self.lift_m, self.still_steps = float(lift_m), int(still_steps)
        clear = {'name': 'carry-clear', 'root': {'xy': [-float(clear_m), 0.], 'speed_m_s': .35}}
        self.make = lambda name, g: make_locomotion(g, self.run/name)
        self.clear_goal, self.command_goal = clear, goal
        self.reset()

    def reset(self, seed=None):
        self.stage, self.arms, self.samples, self.log = 'grasp', None, [], {'stages': []}
        self.clear = self.make('carry-clear', self.clear_goal)
        self.command = self.make('carry-command', self.command_goal)
        self.clear.reset(seed)
        self.command.reset(seed)
        return self.grasp.reset(seed)

    def switch(self, stage, obs, **extra):
        self.stage = stage
        self.log['stages'].append({'stage': stage, 'time': obs['time'], **extra})
        (self.run/'carry.json').write_text(json.dumps(self.log, indent=1))

    def grasp_done(self, obs):
        r = self.grasp.round
        if r is None or r.phase != 'hold':
            return False
        ended = obs['time'] >= r.phase_time+self.grasp.cfg['hold_s']-1e-6
        return ended and obs['box_com'][2]-r.box_pos[2] >= self.lift_m

    def __call__(self, request):
        samples = request['observation']['oracle_history']
        obs = samples[-1]
        self.samples = (self.samples+samples)[-200:]
        if self.stage == 'grasp' and self.grasp_done(obs):
            self.arms = np.asarray(obs['reference_joints'], dtype=float)[ARM]
            self.info = {**self.grasp.grasp_info(), 'hands': 'gate', 'event': None}
            self.switch('clear', obs, lift_m=obs['box_com'][2]-self.grasp.round.box_pos[2], round=self.grasp.round.index,
                        arms=self.arms.tolist())
        if self.stage == 'grasp':
            return self.grasp(request)
        if self.stage == 'clear' and getattr(self.clear, 'phase', None) == 'hold' and still(self.samples, self.still_steps):
            self.switch('command', obs, box_com=obs['box_com'])
        policy = self.clear if self.stage == 'clear' else self.command
        response = policy(request)
        actions = np.asarray(response['action_chunk'], dtype=float)
        actions[:, [9+i for i in ARM]] = self.arms
        response['action_chunk'] = actions.tolist()
        response['grasp'] = {**self.info, 'phase': f'carry-{self.stage}'}
        return response
