"""The deviation analysis on synthetic triplets with a known answer."""
import json
from types import SimpleNamespace

import numpy as np

import deviation


def walk(direction, size, seed, actual, command, yaw=False):
    xy = [0., 0.] if yaw else list(command)
    return {'run': f'x-open-{direction}-{seed}', 'group': 'open', 'seed': seed, 'command': f'{direction}-{size}',
            'direction': direction, 'size': size, 'command_xy': xy, 'command_yaw_deg': command if yaw else 0.,
            'actual_xy': [0., 0.] if yaw else list(actual), 'actual_yaw_deg': actual if yaw else 0., 'fell': False}


def test_systematic_gain_is_calibration_and_noise_is_random(tmp_path):
    rng = np.random.default_rng(0)
    walks = []
    for seed in range(1000, 1005):
        for size in (.25, .5, 1., 1.5):
            walks.append(walk('fwd', size, seed, [.5*size+rng.normal(0, .005), rng.normal(0, .005)], [size, 0.]))
            walks.append(walk('left', size, seed, [rng.normal(0, .5), size+rng.normal(0, .5)], [0., size]))
        for turn in (45, 90, 180):
            walks.append(walk('turn', turn, seed, .9*turn+rng.normal(0, 1.), turn, yaw=True))
    path = tmp_path/'triplets.json'
    path.write_text(json.dumps({'walks': walks, 'reaches': []}))
    deviation.analyze(SimpleNamespace(triplets=str(path)))
    out = json.loads((tmp_path/'analysis.json').read_text())['directions']
    assert abs(out['fwd']['coefficient']-.5) < .02 and out['fwd']['verdict'] == 'calibration is enough'
    assert out['left']['verdict'] == 'mostly random: only closed loop'
    assert abs(out['turn']['coefficient']-.9) < .02 and out['turn']['verdict'] == 'calibration is enough'
