import numpy as np

from analyze_prediction import errors, measured_at, summarize


def series():
    t = np.array([0., .02, .04])
    eye = np.eye(3)
    turned = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    return {'time': t, 'root_pos': np.array([[0., 0, 0], [1, 0, 0], [3, 0, 0]]),
            'wrist_pos': np.zeros((3, 2, 3)), 'root_rot': np.array([eye, eye, turned]),
            'wrist_rot': np.tile(eye, (3, 2, 1, 1))}


def test_measured_state_is_interpolated_between_control_steps():
    at = measured_at(series(), .03)
    assert np.allclose(at['root_pos'], [2, 0, 0])
    assert np.allclose(measured_at(series(), .02)['root_pos'], [1, 0, 0])
    # Rotations are not blended: the nearer sample is used.
    assert np.allclose(measured_at(series(), .039)['root_rot'], series()['root_rot'][2])
    assert np.allclose(measured_at(series(), .021)['root_rot'], np.eye(3))


def test_no_measurement_outside_the_record():
    assert measured_at(series(), .05) is None
    assert measured_at(series(), -.01) is None


def test_errors_are_in_centimetres_and_degrees():
    eye = np.eye(3)
    turned = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    pred = {'root_pos': np.array([.03, 0, 0]), 'root_rot': turned,
            'wrist_pos': np.array([[0, .01, 0], [0, 0, 0]]), 'wrist_rot': np.array([eye, turned])}
    actual = {'root_pos': np.zeros(3), 'root_rot': eye, 'wrist_pos': np.zeros((2, 3)), 'wrist_rot': np.array([eye, eye])}
    e = errors(pred, actual)
    assert np.isclose(e['root_pos_cm'], 3) and np.isclose(e['root_rot_deg'], 90)
    assert np.allclose(e['wrist_pos_cm'], [1, 0]) and np.allclose(e['wrist_rot_deg'], [0, 90])


def test_summary_separates_reaching_from_holding_and_pools_both_wrists():
    def row(reaching, horizon, wrist):
        e = {'root_pos_cm': 1., 'root_rot_deg': 2., 'wrist_pos_cm': wrist, 'wrist_rot_deg': [0., 0.]}
        return {'reaching': reaching, 'horizon_s': horizon, 'kimodo': e, 'persistence': e}
    table = summarize([row(True, .5, [1., 3.]), row(False, .5, [5., 7.]), row(False, 1., [9., 9.])])
    assert table['reaching@0.5']['kimodo']['wrist_pos_cm'] == {'mean': 2., 'p90': 2.8, 'max': 3., 'n': 2}
    assert table['holding@0.5']['kimodo']['wrist_pos_cm']['mean'] == 6.
    assert table['all@0.5']['kimodo']['wrist_pos_cm']['n'] == 4
    assert 'reaching@1.0' not in table and table['all@1.0']['kimodo']['root_pos_cm']['n'] == 1


def test_summary_groups_moving_goal_phases_and_keeps_only_covered_horizons():
    def row(phase, horizon, root, wrist, root_covered=True, wrist_covered=True):
        e = {'root_pos_cm': root, 'root_rot_deg': 1., 'wrist_pos_cm': wrist, 'wrist_rot_deg': [0., 0.]}
        return {'phase': phase, 'reaching': True, 'horizon_s': horizon, 'kimodo': e, 'persistence': e,
                'root_covered': root_covered, 'wrist_covered': wrist_covered, 'wrist_sides': [1]}
    table = summarize([row('walk', 1., 10., [50., 4.], wrist_covered=False), row('walk', 1., 20., [50., 6.], wrist_covered=False),
                       row('reach', 1., 2., [50., 1.]), row('reach', 1., 4., [50., 3.], wrist_covered=False)])
    walk, reach = table['walk@1.0']['kimodo'], table['reach@1.0']['kimodo']
    assert list(table) == ['all@1.0', 'walk@1.0', 'reach@1.0']
    assert walk['root_pos_cm']['mean'] == 15. and walk['covered']['root_pos_cm']['n'] == 2
    assert walk['wrist_pos_cm'] == {'mean': 5., 'p90': 5.8, 'max': 6., 'n': 2}    # the conditioned hand only
    assert walk['covered']['wrist_pos_cm'] is None                                # no hand condition while walking
    assert reach['covered']['wrist_pos_cm'] == {'mean': 1., 'p90': 1., 'max': 1., 'n': 1}
