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


def test_walk_row_reads_a_run(tmp_path):
    """One finished open-loop run on disk -> one triplet row (command, reference, actual in the start frame)."""
    run = tmp_path/'exp2dev-open-20261125-r1-fl-050-s1000'
    run.mkdir()
    eye = np.eye(3)
    (run/'target.json').write_text(json.dumps({'motion_goal': {'name': 'fl-050', 'root': {'xy': [.35, .35], 'speed_m_s': .35}},
                                               'heading': eye.tolist(), 'noise_seed': 1000, 'phases': []}))
    quat = [1., 0., 0., 0.]
    feet = [[0., .1, 0.], [0., -.1, 0.]]

    def step(t, x, y):
        sample = {'time': t, 'root_pos': [x, y, .75], 'root_quat': quat}
        return {'before': sample, 'after': sample, 'reference_root_world': [2*x, 2*y, .75],
                'reference_foot_world': [[2*x+f[0], 2*y+f[1], f[2]] for f in feet], 'reference_root_quat': quat}
    (run/'execution.jsonl').write_text('\n'.join(json.dumps(step(t, .1*t, .1*t)) for t in range(4)))
    row = deviation.walk_row(run)
    assert row['group'] == 'open' and row['direction'] == 'fl' and row['size'] == .5
    np.testing.assert_allclose(row['actual_xy'], [.3, .3])
    np.testing.assert_allclose(row['reference_xy'], [.6, .6])
    np.testing.assert_allclose(row['target_xy'], [.5/np.sqrt(2), .5/np.sqrt(2)])
    assert not row['fell']
