"""The step loop: observation -> commander -> one command -> planner -> 40-d reference -> tracker.

``CommandLoopPolicy`` is served to the simulator like every other policy here
(``agentic/scripts/serve_oracle.py --command-loop``): each request carries the measured samples
since the last one, the ground-truth objects (``scene_truth``, world frame) and, when the loop
asked for one, a camera capture (``vision``). While a command runs, the loop only forwards the
planner's chunks. When it is complete the loop logs the end state; if the commander needs an
image, it first holds the body for a few steps and asks the simulator for a capture, so the image,
the proprioception and the objects the commander sees are of one instant. The simulator is paused
while the commander thinks (it waits for the reply).

The episode ends when the commander says ``done``, when a budget runs out (commands, commander
calls, refused replies in a row, simulated time), or when Arena ends it (its own success check,
a fall, or the step limit). The response's ``command_loop.end`` asks the simulator to stop.
"""
import time
from pathlib import Path

import numpy as np

from kimodo_vla.core.types import yaw_matrix, yaw_of
from kimodo_vla.oracle.geometry import SIDES, angle_deg, rotation_wxyz
from .episode_log import EpisodeLog
from .planner import end_states, hold_rows
from .state import body_state, hands_closed

BUDGETS = {'max_commands': 40, 'max_commander_calls': 60, 'max_refused_in_row': 3, 'max_episode_s': None,
           'start_hold_s': .6, 'capture_hold_steps': 5}


def errors(command, target, state):
    """How far one end state (measured or reference) is from the commanded target."""
    out = {}
    if 'root_xy' in target:
        out['root_xy_m'] = float(np.linalg.norm(np.subtract(state['root'][:2], target['root_xy'])))
    if 'yaw_deg' in target:
        out['yaw_deg'] = float(abs((state['yaw_deg']-target['yaw_deg']+180.) % 360.-180.))
    if 'pelvis_height_m' in target:
        out['pelvis_height_m'] = float(abs(state['pelvis_height_m']-target['pelvis_height_m']))
    if 'torso_pitch_deg' in target:
        out['torso_pitch_deg'] = float(abs(state['torso_pitch_deg']-target['torso_pitch_deg']))
    for side, t in target.get('wrists', {}).items():
        w = state['wrists'][side]
        out[f'{side}_wrist_m'] = float(np.linalg.norm(np.subtract(w['position'], t['position'])))
        out[f'{side}_wrist_deg'] = angle_deg(np.asarray(w['rotation']), np.asarray(t['rotation']))
    return out


class CommandLoopPolicy:
    def __init__(self, checkpoint, text_cache, run, *, config, task_text, seed=None, planner=None, commander=None,
                 meta=None, fk=None):
        self.config = config
        self.run = Path(run)
        self.log = EpisodeLog(self.run/'loop')
        self.budgets = {**BUDGETS, **config.get('budgets', {})}
        self.task_text = task_text
        if planner is None:
            from .planner import KimodoPlanner
            planner = KimodoPlanner(checkpoint, text_cache, self.run/'loop', config.get('planner'))
        self.planner, self.commander = planner, commander
        self.fk = fk if fk is not None else planner.fk
        self.meta = meta or {}
        self.reset(seed)

    def reset(self, seed=None):
        self.seed = 1000 if seed is None else int(seed)
        self.history, self.heading, self.start_time = [], None, None
        self.hands = np.zeros(2)
        self.active = self.ended = None
        self.index = self.calls = self.refused = 0
        self.last = self.last_error = None
        self.recent = []
        self.capture_asked = False
        return {'status': 'ok'}

    # ---- helpers ------------------------------------------------------------------------
    def begin(self, first, truth):
        rotation = rotation_wxyz(first['root_quat'])
        self.heading = yaw_matrix(float(yaw_of(rotation)))
        self.start_time = first['time']
        self.hold_until = first['time']+self.budgets['start_hold_s']
        self.log.meta({**self.meta, 'seed': self.seed, 'task_text': self.task_text, 'budgets': self.budgets,
                       'planner': getattr(self.planner, 'name', type(self.planner).__name__),
                       'planner_settings': getattr(self.planner, 'settings', None),
                       'commander': type(self.commander).__name__, 'episode_heading': self.heading,
                       'start_time': first['time'], 'scene_truth_at_start': truth})

    def target_info(self, obs):
        if self.active is not None and self.active['command']['type'] == 'hand_pose':
            return self.planner.reacher.target_info
        wrists = np.asarray(obs['wrist_pos'])
        return {'mode': 'command-loop', 'targets_world': np.repeat(wrists[:, None], 2, 1).tolist(),
                'wrist_rotation_world': obs['wrist_rot'], 'arrival_time': obs['time']}

    def respond(self, obs, rows, capture=False):
        rows = np.asarray(rows, dtype=float)
        self.log.sent(obs['time'], self.active['index'] if self.active else None, rows)
        return {'action_chunk': rows.tolist(), 'oracle_target': self.target_info(obs),
                'command_loop': {'capture': bool(capture), 'end': bool(self.ended),
                                 'command': self.active['index'] if self.active else None}}

    def observation(self, obs, truth):
        state = body_state(obs, self.fk)
        closed = hands_closed(obs)
        used = obs['time']-self.start_time
        left = None if self.budgets['max_episode_s'] is None else self.budgets['max_episode_s']-used
        return {'task': self.task_text, 'time_s': used,
                'budget': {'commands_left': self.budgets['max_commands']-self.index, 'time_left_s': left},
                'robot': {'pelvis': state['root'], 'heading_deg': state['yaw_deg'], 'torso_pitch_deg': state['torso_pitch_deg'],
                          'wrists': {s: {k: state['wrists'][s][k] for k in ('position', 'palm', 'fingers')} for s in SIDES},
                          'hands': {s: 'closed' if c else 'open' for s, c in zip(SIDES, closed)}},
                'objects': (truth or {}).get('objects', []),
                'last_command': self.last, 'recent_commands': self.recent[-6:]}

    def image(self, vision):
        if not vision:
            return None
        from PIL import Image
        data = np.load(vision['capture_path'])
        return Image.fromarray(np.asarray(data['rgb'], dtype=np.uint8))

    def end(self, reason, obs):
        self.ended = reason
        self.log.summary({'end_reason': reason, 'time': obs['time'], 'commands': self.index, 'commander_calls': self.calls,
                          'budgets': self.budgets, 'hands_closed': hands_closed(obs)})

    # ---- one command --------------------------------------------------------------------
    def finish(self, obs):
        a = self.active
        states = end_states(obs, self.fk)
        row = {'kind': 'end', 'index': a['index'], 'type': a['command']['type'], 'time': obs['time'],
               'duration_s': obs['time']-a['time'], 'status': 'completed', 'hands_closed': hands_closed(obs),
               'measured': states['measured'], 'reference': states['reference'],
               'measured_error': errors(a['command'], a['target'], states['measured']),
               'reference_error': errors(a['command'], a['target'], states['reference'])}
        self.log.command(row)
        self.last = {'command': a['command'], 'status': 'completed', 'duration_s': row['duration_s'],
                     'target': {k: v for k, v in a['target'].items() if k != 'wrists'},
                     'measured_error': row['measured_error']}
        self.active = None

    def launch(self, command, source, samples, obs):
        index = self.index
        self.index += 1
        context = {'history': self.history, 'hands': self.hands, 'heading': self.heading,
                   'folder': self.run/'loop'/f'cmd-{index:03d}', 'seed': self.seed*1000+index, 'new_samples': len(samples)}
        states = end_states(obs, self.fk)
        info = self.planner.start(command, context)
        self.hands = np.asarray([float(h) for h in info['hands_closed']])
        self.active = {'index': index, 'command': command, 'time': obs['time'], 'target': info['target']}
        self.log.command({'kind': 'start', 'index': index, 'type': command['type'], 'time': obs['time'], 'command': command,
                          'source_call': source, 'target': info['target'], 'plan': info.get('plan'),
                          'hands_closed_before': hands_closed(obs), 'measured': states['measured'],
                          'reference': states['reference']})
        self.recent.append(command['type'])

    def decide(self, obs, truth, vision):
        """Ask the commander until it gives a valid command or a budget runs out; returns the command or None."""
        observation = self.observation(obs, truth)
        image = self.image(vision) if self.commander.needs_image else None
        while True:
            if self.calls >= self.budgets['max_commander_calls']:
                self.end('commander_call_budget', obs)
                return None, None
            begin = time.perf_counter()
            record = self.commander.next(observation, image, self.last_error)
            call = self.calls
            self.calls += 1
            self.log.vlm({'call': call, 'time': obs['time'], 'wall_s': time.perf_counter()-begin,
                          'image': vision['capture_path'] if vision else None, 'observation': observation,
                          'refusal_note': self.last_error, **record})
            if record['error'] is None:
                self.refused, self.last_error = 0, None
                return record['command'], call
            self.refused += 1
            self.last_error = record['error']
            if self.refused >= self.budgets['max_refused_in_row']:
                self.end('refused_replies', obs)
                return None, None

    # ---- the request --------------------------------------------------------------------
    def __call__(self, request):
        samples = request['observation']['oracle_history']
        if not samples:
            raise ValueError('Measured history is required')
        truth = request['observation'].get('scene_truth')
        vision = request['observation'].get('vision')
        if self.heading is None:
            self.begin(samples[0], truth)
        if self.history and samples[0]['time'] <= self.history[-1]['time']:
            raise ValueError('Non-monotonic history')
        self.history.extend(samples)
        self.history = self.history[-400:]
        obs = samples[-1]
        if self.active is not None:
            rows = self.planner.chunk(request)
            if rows is not None:
                return self.respond(obs, rows)
            self.finish(obs)
        if self.ended:
            return self.respond(obs, hold_rows(obs, self.heading, 25, self.hands))
        if obs['time'] < self.hold_until-1e-6:
            count = int(np.clip(round((self.hold_until-obs['time'])*50), 1, 25))
            return self.respond(obs, hold_rows(obs, self.heading, count, self.hands))
        used = obs['time']-self.start_time
        if self.index >= self.budgets['max_commands']:
            self.end('command_budget', obs)
        elif self.budgets['max_episode_s'] is not None and used >= self.budgets['max_episode_s']:
            self.end('time_budget', obs)
        if self.ended:
            return self.respond(obs, hold_rows(obs, self.heading, 25, self.hands))
        fresh = vision is not None and abs(vision['time']-obs['time']) < 1e-6
        if self.commander.needs_image and not fresh:
            return self.respond(obs, hold_rows(obs, self.heading, self.budgets['capture_hold_steps'], self.hands), capture=True)
        command, call = self.decide(obs, truth, vision if fresh else None)
        if command is None:
            return self.respond(obs, hold_rows(obs, self.heading, 25, self.hands))
        if command['type'] == 'done':
            self.end('commander_done', obs)
            return self.respond(obs, hold_rows(obs, self.heading, 25, self.hands))
        self.launch(command, call, samples, obs)
        rows = self.planner.chunk(request)
        if rows is None:      # nothing to do (a toggle that is already over): hold a moment, decide again
            rows = hold_rows(obs, self.heading, 5, self.hands)
        return self.respond(obs, rows)
