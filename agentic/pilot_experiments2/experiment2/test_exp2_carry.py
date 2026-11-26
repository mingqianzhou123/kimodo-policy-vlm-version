"""Carrying condition (kimodo_vla/oracle/carry_policy.py): stage switches and the locked arms, with stand-in policies."""
import json
from types import SimpleNamespace

import numpy as np

from kimodo_vla.oracle.carry_policy import ARM, CarryPolicy, still


def sample(t, x=0., box_z=.80, ref=0.):
    return {'time': t, 'root_pos': [x, 0., .78], 'root_quat': [1., 0., 0., 0.], 'box_com': [0., 0., box_z],
            'reference_joints': [ref]*29}


class Grasp:
    cfg = {'hold_s': .5}

    def __init__(self):
        self.round = SimpleNamespace(phase='lift', phase_time=0., box_pos=[0., 0., .75], index=0)
        self.calls = 0

    def reset(self, seed=None):
        return {'status': 'ok'}

    def grasp_info(self):
        return {'round': 0, 'phase': self.round.phase, 'hands': 'gate', 'event': None}

    def __call__(self, request):
        self.calls += 1
        return {'action_chunk': np.zeros((25, 40)).tolist(), 'grasp': self.grasp_info()}


class Walk:
    def __init__(self, goal, folder):
        self.goal, self.phase = goal, None

    def reset(self, seed=None):
        return {}

    def __call__(self, request):
        self.phase = 'walk' if self.phase is None else self.phase
        return {'action_chunk': np.full((25, 40), 7.).tolist(), 'oracle_target': {}}


def ask(policy, samples):
    return policy({'observation': {'oracle_history': samples}})


def test_still_needs_a_quiet_window():
    quiet = [sample(t) for t in np.arange(0, .6, .02)]
    assert still(quiet, 25)
    moving = [sample(t, x=.2*t) for t in np.arange(0, .6, .02)]
    assert not still(moving, 25)
    assert not still(quiet[:10], 25)


def test_grasp_then_clear_then_command_with_arms_locked(tmp_path):
    grasp = Grasp()
    policy = CarryPolicy(grasp, Walk, {'name': 'p2e2-forward'}, tmp_path)
    ask(policy, [sample(.1)])
    assert policy.stage == 'grasp' and grasp.calls == 1
    grasp.round.phase, grasp.round.phase_time = 'hold', 1.
    ask(policy, [sample(1.2, box_z=.90)])                     # hold not over yet
    assert policy.stage == 'grasp'
    out = ask(policy, [sample(1.6, box_z=.90, ref=.3)])        # hold over, lifted 15 cm
    assert policy.stage == 'clear' and policy.clear.goal['root']['xy'] == [-.6, 0.]
    actions = np.asarray(out['action_chunk'])
    assert np.all(actions[:, [9+i for i in ARM]] == .3) and np.all(actions[:, 0] == 7.)
    assert out['grasp']['hands'] == 'gate' and out['grasp']['event'] is None
    policy.clear.phase = 'hold'
    ask(policy, [sample(t, box_z=.90) for t in np.arange(1.62, 2.3, .02)])
    assert policy.stage == 'command'
    stages = json.loads((tmp_path/'carry.json').read_text())['stages']
    assert [s['stage'] for s in stages] == ['clear', 'command']


def test_an_unlifted_round_stays_with_the_grasp_policy(tmp_path):
    grasp = Grasp()
    policy = CarryPolicy(grasp, Walk, {'name': 'p2e2-forward'}, tmp_path)
    grasp.round.phase, grasp.round.phase_time = 'hold', 1.
    ask(policy, [sample(1.6, box_z=.80)])                     # 5 cm: not lifted
    assert policy.stage == 'grasp' and grasp.calls == 1
