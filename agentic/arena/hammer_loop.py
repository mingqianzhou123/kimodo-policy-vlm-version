"""Simulator-side half of the blocking hammer grasp: the right-hand gate, the judge and the records.

As for the box (``grasp_loop``), the policy only says which round and phase it is in and where
the right wrist should be when the hand closes. Whether it closes, and whether the hammer
counts as grasped, is decided here every control step from the measured wrist pose and the
measured contact forces. The left hand is never commanded to close.

Two optional perturbations of the first round, both requested through the layout:

* ``inject`` {"along_m", "across_m", "turn_deg"}: one control step after the first latch the
  hammer is moved by this known amount (along / across the handle, turned about the vertical
  through the grip point) and left at rest. Everything else stays as it was latched.
* ``motion`` {"speed_mps", "direction"}: from the first latch the hammer is driven at constant
  speed along the table edge (``direction`` +1 to the robot's left, -1 to its right) until a
  hand first touches it or it has travelled ``max_travel_m``; afterwards physics alone.
"""
import json
import math

import numpy as np

from arena.grasp_judge import Attempt, Criteria, GraspJudge, Sustained, summarize
from arena.grasp_loop import up_z
from kimodo_vla.controllers.gripper_gate import ArrivalGripperGate, GripperEvent

FINGERS = ('thumb_0', 'thumb_1', 'thumb_2', 'middle_0', 'middle_1', 'index_0', 'index_1')


def gripped_right(forces, c):
    """The right hand presses on the hammer and nothing except the robot touches it."""
    return forces['right_n'] >= c.hand_force_n and forces['external_n'] <= c.support_force_n


class HammerJudge(GraspJudge):
    def update(self, time, box_height, forces):
        c = self.criteria
        lifted = box_height - self.rest_height >= c.lift_m
        if self._run.update(time, lifted and gripped_right(forces, c)) and self.success_time is None:
            self.success_time = time
        return self.success_time is not None


class HammerAttempt(Attempt):
    """One round, labelled as for the box but with a one-handed grip."""
    def update(self, time, box_pos, box_up_z, forces, hands_closed, success=False, driven=False):
        c = self.criteria
        for side in ('left', 'right'):
            if forces[side + '_n'] >= c.hand_force_n:
                self._mark(side + '_contact', time)
        if forces['robot_other_n'] >= c.hand_force_n:
            self._mark('other_contact', time)
        if hands_closed:
            self._mark('closed', time)
        clear = box_pos[2] - self.rest_height >= c.clear_m
        grip = gripped_right(forces, c) and clear
        if forces['external_n'] <= c.support_force_n and clear:
            self._mark('off_support', time)
        if self._grip.update(time, grip):
            self._mark('held', self._grip.since)
            self.holding = True
        if self.holding and self._loss.update(time, not grip) and not success:
            self._mark('slipped', self._loss.since)
            self.holding = False
        if success:
            self._mark('success', time)
        if not self.holding:
            tilt = math.degrees(math.acos(max(-1., min(1., box_up_z))))
            origin = self.latched_box_pos if self.shove_from is None else self.shove_from
            shove = 0. if driven else float(np.linalg.norm(np.asarray(box_pos[:2]) - np.asarray(origin[:2])))
            if tilt > c.tilt_deg or shove > c.shove_m or self.rest_height - box_pos[2] > c.drop_m:
                self._mark('knocked_over', time)


def yaw_of_quat(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class HammerGraspLoop:
    def __init__(self, run, config, stop, env=None, provider=None, layout=None, post_success_steps=25):
        self.run, self.stop, self.env, self.provider = run, stop, env, provider
        self.cfg = config
        self.gate = ArrivalGripperGate(**config['gate'])
        self.criteria = Criteria(**config['criteria'])
        scene = config['scene']
        f = math.radians(scene['facing_deg'])
        self.forward, self.left = np.array([math.cos(f), math.sin(f)]), np.array([-math.sin(f), math.cos(f)])
        self.edge = np.asarray(scene['table_edge_point_m'], dtype=float)
        self.judge = self.info = self.clock_start = self.target = None
        self.attempts, self.post_success = [], post_success_steps
        self.steps = (run / 'grasp-steps.jsonl').open('a', buffering=1)
        self.layout = layout or {}
        self.inject = self.layout.get('spec', {}).get('inject')
        self.injected = None
        self.motion = None
        self.drive_released_at = self.drive_release_box = None
        self.finger_ids = None
        if env is not None:
            names = list(env.scene['robot'].joint_names)
            self.finger_ids = [names.index(f'right_hand_{n}_joint') for n in FINGERS]

    # ---- hand command: right follows the gate, left stays open ---------------------------
    @property
    def closed(self):
        return self.gate.closed

    def hands(self):
        return False, self.gate.closed

    def release(self):
        self.gate.cancel()
        self.gate.closed = False

    # ---- policy side ---------------------------------------------------------------------
    def on_plan(self, info, sample):
        now = sample['time']
        if self.judge is None:
            self.judge = HammerJudge(sample['box_com'][2], self.criteria)
            self.clock_start = info['latch_time']
            self.start_motion(sample)
        if not self.attempts or info['round'] != self.attempts[-1].index:
            if self.attempts:
                self.attempts[-1].close(now)
            self.attempts.append(HammerAttempt(info['round'], info['latch_time'], tuple(info['latched_box_pos']),
                                               self.judge.rest_height, self.criteria))
            self.attempts[-1].notes['latched_yaw_rad'] = yaw_of_quat(info['latched_box_quat'])
        if info['hands'] == 'open' and (self.gate.closed or self.gate.status in ('pending', 'timed_out')):
            self.release()
        if info['event']:
            e = info['event']
            if self.gate.submit(GripperEvent(f"{e['id']}:right", True, tuple(map(float, e['positions'][1])),
                                             tuple(tuple(map(float, row)) for row in e['rotations'][1]), float(e['deadline']))):
                self.target = np.asarray(e['positions'][1])
        self.info = info

    # ---- perturbations of the first round ------------------------------------------------
    def start_motion(self, sample):
        m = self.layout.get('spec', {}).get('motion')
        if m and m.get('speed_mps', 0) > 0:
            from arena.hammer_layout import LinearMotion
            direction = m.get('direction') or self.layout.get('direction', 1)
            self.motion = LinearMotion(sample['box_pos'][:2], self.left * direction, m['speed_mps'],
                                       self.cfg['stand']['motion_max_travel_m'])

    def act_on_scene(self, env, now):
        """Apply a pending injection, or the drive for this control step, before it runs."""
        from arena.hammer_scene import drive_hammer, set_hammer
        if self.inject and self.injected is None and self.clock_start is not None and self.attempts:
            from arena.hammer_layout import displaced
            from kimodo_vla.oracle.hammer_policy import handle_frame
            hammer = env.scene['object_l'].data
            root = hammer.root_state_w[0, :3].detach().cpu().numpy()
            quat = hammer.root_state_w[0, 3:7].detach().cpu().numpy()
            grip, axis = handle_frame(root, quat, self.cfg['policy'])
            xyz, yaw = displaced(root, yaw_of_quat(quat), grip, axis, self.inject.get('along_m', 0.),
                                 self.inject.get('across_m', 0.), self.inject.get('turn_deg', 0.))
            set_hammer(env, xyz, yaw)
            self.injected = {'time': now, 'from': root.tolist(), 'from_yaw_rad': yaw_of_quat(quat), 'to': xyz, 'to_yaw_rad': yaw,
                             'request': self.inject}
            self.attempts[0].notes['injected'] = self.injected
        if self.motion is not None and self.clock_start is not None and self.drive_released_at is None:
            elapsed = now - self.clock_start
            drive_hammer(env, self.motion.position(elapsed), self.motion.velocity(elapsed))

    # ---- per step -------------------------------------------------------------------------
    def measure(self, sample, forces):
        """Per-round reach and hammer-motion notes, kept with the attempt."""
        a = self.attempts[-1]
        n, now, com = a.notes, sample['time'], sample['box_com']
        phase = self.info['phase']
        root = np.asarray(sample['root_pos'][:2])
        if phase in ('raise', 'advance'):
            ahead = float((root - self.edge) @ self.forward)
            if 'anchor_root_ahead' not in n:
                n.update(anchor_root_ahead=ahead, anchor_clearance_m=-ahead)
            # How far the pelvis backs away from the table while the hand goes up and out.
            n['backoff_m'] = max(n.get('backoff_m', 0.), n['anchor_root_ahead'] - ahead)
            if phase == 'advance' and self.target is not None and not self.gate.closed:
                n['wrist_gap_m'] = float(np.linalg.norm(np.asarray(sample['wrist_pos'][1]) - self.target))
                n['wrist_gap_vec_m'] = (np.asarray(sample['wrist_pos'][1]) - self.target).tolist()
                n.update(reach_time=now, reach_box_pos=list(com), reach_box_yaw_rad=yaw_of_quat(sample['box_quat']))
        if self.gate.closed and 'close_box_pos' not in n:
            latched = np.asarray(a.latched_box_pos)
            d = np.asarray(com[:2]) - latched[:2]
            yaw0 = n.get('latched_yaw_rad', 0.)
            axis = np.array([-math.sin(yaw0), math.cos(yaw0)])   # handle axis (hammer local +y) when latched
            n.update(close_time=now, close_box_pos=list(com), latch_to_close_m=float(np.linalg.norm(d)),
                     latch_to_close_along_m=float(d @ axis), latch_to_close_across_m=float(d @ np.array([axis[1], -axis[0]])),
                     latch_to_close_turn_deg=math.degrees((yaw_of_quat(sample['box_quat']) - yaw0 + math.pi) % (2 * math.pi) - math.pi))
        if self.drive_released_at is None and self.motion is not None and (
                forces['left_n'] >= self.criteria.hand_force_n or forces['right_n'] >= self.criteria.hand_force_n
                or forces['robot_other_n'] >= self.criteria.hand_force_n):
            self.drive_released_at, self.drive_release_box = now, list(com)
            n['drive_released_at'] = now

    def before_step(self, sample):
        """Decide the right-hand command for this control step from the wrist as measured now."""
        return self.gate.update(now_s=sample['time'], observation_time_s=sample['time'],
                                position=sample['wrist_pos'][1], rotation=sample['wrist_rot'][1],
                                speed_mps=float(np.linalg.norm(sample['wrist_velocity'][1])))

    def fingers(self):
        """Right finger joint positions, and the targets the hand is driven to, in FINGERS order."""
        if self.env is None:
            return None, None
        q = self.env.scene['robot'].data.joint_pos[0, self.finger_ids].detach().cpu().numpy().tolist()
        target = None if self.provider is None else np.asarray(self.provider._right_hand_target).tolist()
        return q, target

    def after_step(self, sample, forces):
        if self.judge is None:
            return None
        now, com = sample['time'], sample['box_com']
        already = self.judge.success_time is not None
        success = self.judge.update(now, com[2], forces)
        driven = self.motion is not None and self.drive_released_at is None
        self.measure(sample, forces)
        if driven and self.drive_released_at is not None:
            self.attempts[-1].shove_from = tuple(com[:2])
        self.attempts[-1].update(now, com, up_z(sample['box_quat']), forces, self.gate.closed, success,
                                 driven=driven and self.drive_released_at is None)
        q, target = self.fingers()
        row = {'time': now, 'round': self.info['round'], 'phase': self.info['phase'], 'box_com': com,
               'box_quat': sample['box_quat'], 'wrist_pos': sample['wrist_pos'], 'hands_closed': self.gate.closed,
               'gate': self.gate.status, 'lift_m': com[2] - self.judge.rest_height, 'success': success,
               'right_fingers': q, 'right_finger_target': target, **forces}
        self.steps.write(json.dumps(row) + '\n')
        if success and not already:
            self.attempts[-1].close(now)
        result = summarize(self.attempts, 'success' if success else None, self.judge.success_time, self.clock_start)
        result.update(task='doubledesk_hammer', rest_height=self.judge.rest_height, clock_start=self.clock_start,
                      last_time=now, criteria=vars(self.criteria), gate_fired_at=self.gate_fired_at(),
                      layout=self.layout or None, injected=self.injected,
                      drive_released_at=self.drive_released_at, drive_release_box=self.drive_release_box)
        (self.run / 'grasp-result.json').write_text(json.dumps(result, indent=1))
        if success:
            self.post_success -= 1
            if self.post_success <= 0:
                self.stop()
        return row

    def gate_fired_at(self):
        return [a.events.get('closed') for a in self.attempts]
