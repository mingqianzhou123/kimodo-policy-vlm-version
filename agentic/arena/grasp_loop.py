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
    """``layout`` (optional) is where the box was placed and how it moves; see ``box_layout``.

    A moving box is driven from the first latch until a hand first touches it. From then on it
    is left to physics for the rest of the episode: its velocity is not zeroed, it is not held
    in place and no later round drives it again.
    """
    def __init__(self, run, config, stop, post_success_steps=25, layout=None):
        self.run, self.stop = run, stop
        self.gate = SynchronizedGripperGate(**config['gate'])
        self.criteria = Criteria(**config['criteria'])
        self.edge = config['scene']['table_edge_x_m']
        self.judge = self.info = self.clock_start = self.targets = None
        self.attempts, self.post_success = [], post_success_steps
        self.steps = (run / 'grasp-steps.jsonl').open('a', buffering=1)
        self.layout, self.motion = layout or {}, None
        self.drive_released_at = self.drive_release_box = None

    def start_motion(self, box_y):
        speed = self.layout.get('speed_mps', 0.)
        if speed > 0:
            from arena.box_layout import BoxMotion
            self.motion = BoxMotion(self.layout['segment_y'], speed, box_y, self.layout['direction'])

    def drive(self, now):
        """Box y and y-velocity to impose before this control step, or None when not driven."""
        if self.motion is None or self.clock_start is None or self.drive_released_at is not None:
            return None
        elapsed = now - self.clock_start
        return self.motion.position(elapsed), self.motion.velocity(elapsed)

    @property
    def closed(self):
        return self.gate.closed

    def on_plan(self, info, sample):
        """Take the policy's round, phase and close event from one plan response."""
        now = sample['time']
        if self.judge is None:
            self.judge = GraspJudge(sample['box_com'][2], self.criteria)
            self.clock_start = info['latch_time']
            self.start_motion(sample['box_com'][1])
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
            self.targets = np.asarray(event['positions'])
        self.info = info

    def measure(self, sample, forces):
        """Per-round reach and box-motion notes: what the attribution needs, kept with the attempt."""
        n, now, com = self.attempts[-1].notes, sample['time'], sample['box_com']
        phase = self.info['phase']
        if phase in ('raise', 'advance'):
            x = sample['root_pos'][0]
            if 'anchor_root_x' not in n:
                n.update(anchor_root_x=x, anchor_clearance_m=self.edge - x,
                         anchor_lateral_m=self.attempts[-1].latched_box_pos[1] - sample['root_pos'][1])
            # How far the pelvis backs away from the table while the hands go up and out.
            n['backoff_m'] = max(n.get('backoff_m', 0.), n['anchor_root_x'] - x)
            if phase == 'advance' and self.targets is not None and not self.gate.closed:
                n['wrist_gap_m'] = np.linalg.norm(np.asarray(sample['wrist_pos']) - self.targets, axis=1).tolist()
        if self.gate.closed and 'close_box_pos' not in n:
            latched = np.asarray(self.attempts[-1].latched_box_pos)
            n.update(close_time=now, close_box_pos=list(com),
                     latch_to_close_m=float(np.linalg.norm(np.asarray(com[:2]) - latched[:2])))
            if self.motion is not None:
                start, end = self.attempts[-1].start_time - self.clock_start, now - self.clock_start
                if self.drive_released_at is not None:
                    end = min(end, self.drive_released_at - self.clock_start)
                n['turned'] = self.motion.turns(start, end) > 0
        if self.drive_released_at is None and self.motion is not None and (
                forces['left_n'] >= self.criteria.hand_force_n or forces['right_n'] >= self.criteria.hand_force_n):
            self.drive_released_at, self.drive_release_box = now, list(com)
            n['drive_released_at'] = now

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
        self.measure(sample, forces)
        self.attempts[-1].update(now, com, up_z(sample['box_quat']), forces, self.gate.closed, success)
        row = {'time': now, 'round': self.info['round'], 'phase': self.info['phase'], 'box_com': com,
               'box_quat': sample['box_quat'], 'wrist_pos': sample['wrist_pos'], 'hands_closed': self.gate.closed,
               'gate': self.gate.status, 'lift_m': com[2] - self.judge.rest_height, 'success': success, **forces}
        self.steps.write(json.dumps(row) + '\n')
        if success and not already:
            self.attempts[-1].close(now)
        result = summarize(self.attempts, 'success' if success else None, self.judge.success_time, self.clock_start)
        result.update(rest_height=self.judge.rest_height, clock_start=self.clock_start, last_time=now,
                      criteria=vars(self.criteria), gate_fired_at=self.gate.fired_at, layout=self.layout or None,
                      drive_released_at=self.drive_released_at, drive_release_box=self.drive_release_box,
                      drive_released_round=next((a.index for a in self.attempts if 'drive_released_at' in a.notes), None))
        (self.run / 'grasp-result.json').write_text(json.dumps(result, indent=1))
        if success:
            self.post_success -= 1
            if self.post_success <= 0:
                self.stop()
        return row
