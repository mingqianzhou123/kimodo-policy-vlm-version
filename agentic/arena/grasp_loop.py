"""Simulator-side half of the blocking grasp loop: the hand gate, the judge and the records.

The policy only says which round and phase it is in and where the wrists should be when the
hands close. Whether they close, and whether the box counts as grasped, is decided here on
every control step from measured wrist poses and measured contact forces.
"""
import json

import numpy as np

from arena.grasp_judge import Attempt, Criteria, GraspJudge, summarize
from kimodo_vla.controllers.gripper_gate import SynchronizedGripperGate


def up_z(quat_wxyz):
    """World-z component of the body's own z axis."""
    _, x, y, _ = quat_wxyz
    return 1. - 2. * (x*x + y*y)


class GraspLoop:
    def __init__(self, run, config, stop, post_success_steps=25):
        self.run, self.stop = run, stop
        self.gate = SynchronizedGripperGate(**config['gate'])
        self.criteria = Criteria(**config['criteria'])
        self.judge = self.info = self.clock_start = None
        self.attempts, self.post_success = [], post_success_steps
        self.steps = (run / 'grasp-steps.jsonl').open('a', buffering=1)

    @property
    def closed(self):
        return self.gate.closed

    def on_plan(self, info, sample):
        """Take the policy's round, phase and close event from one plan response."""
        now = sample['time']
        if self.judge is None:
            self.judge = GraspJudge(sample['box_com'][2], self.criteria)
            self.clock_start = info['latch_time']
        if not self.attempts or info['round'] != self.attempts[-1].index:
            if self.attempts:
                self.attempts[-1].close(now)
            self.attempts.append(Attempt(info['round'], info['latch_time'], tuple(info['latched_box_pos']),
                                         self.judge.rest_height, self.criteria))
        if info['hands'] == 'open' and (self.gate.closed or self.gate.status in ('pending', 'timed_out')):
            self.gate.release()
        if info['event']:
            event = info['event']
            self.gate.submit(event['id'], event['positions'], event['rotations'], event['deadline'])
        self.info = info

    def before_step(self, sample):
        """Decide the hand command for this control step from the wrists as measured now."""
        return self.gate.update(now_s=sample['time'], observation_time_s=sample['time'],
            positions=sample['wrist_pos'], rotations=sample['wrist_rot'],
            speeds_mps=np.linalg.norm(sample['wrist_velocity'], axis=1).tolist())

    def after_step(self, sample, forces):
        """Judge the step that just ran; returns the row that was recorded."""
        if self.judge is None:
            return None
        now, com = sample['time'], sample['box_com']
        already = self.judge.success_time is not None
        success = self.judge.update(now, com[2], forces)
        self.attempts[-1].update(now, com, up_z(sample['box_quat']), forces, self.gate.closed, success)
        row = {'time': now, 'round': self.info['round'], 'phase': self.info['phase'], 'box_com': com,
               'box_quat': sample['box_quat'], 'wrist_pos': sample['wrist_pos'], 'hands_closed': self.gate.closed,
               'gate': self.gate.status, 'lift_m': com[2] - self.judge.rest_height, 'success': success, **forces}
        self.steps.write(json.dumps(row) + '\n')
        if success and not already:
            self.attempts[-1].close(now)
        result = summarize(self.attempts, 'success' if success else None, self.judge.success_time, self.clock_start)
        result.update(rest_height=self.judge.rest_height, clock_start=self.clock_start, last_time=now,
                      criteria=vars(self.criteria), gate_fired_at=self.gate.fired_at)
        (self.run / 'grasp-result.json').write_text(json.dumps(result, indent=1))
        if success:
            self.post_success -= 1
            if self.post_success <= 0:
                self.stop()
        return row
