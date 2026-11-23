import copy

import pytest

from report_settled import assert_same_simulator_setup


def setup(cache):
    usd = f'{cache}/sources/arena/assets/robots/g1/g1.usd'
    return {'robot_asset': {'path': usd, 'sha256': 'abc'},
            'robot_cfg': {'spawn': {'usd_path': usd, 'activate_contact_sensors': True}, 'init_state': {'pos': [0, 0, .8]}},
            'sim_cfg': {'dt': .005}, 'runtime_env': {'ENABLE_DEPTH': '1'}, 'provider': 'sonic',
            'body_names': ['pelvis'], 'joint_names': ['waist_yaw_joint'], 'physics_dt': .005, 'control_dt': .02}


def test_same_setup_survives_a_moved_simulator_cache():
    assert_same_simulator_setup(setup('/first/place'), setup('/second/place'))


def test_different_robot_file_or_settings_are_rejected():
    changed = setup('/first/place'); changed['robot_asset']['sha256'] = 'other'
    with pytest.raises(AssertionError, match='robot_asset'):
        assert_same_simulator_setup(changed, setup('/first/place'))
    unhashed = setup('/first/place'); unhashed['robot_asset']['sha256'] = None
    with pytest.raises(AssertionError, match='robot_asset'):
        assert_same_simulator_setup(unhashed, copy.deepcopy(unhashed))
    for key, value in (('physics_dt', .01), ('robot_cfg', {'spawn': {'usd_path': '/x/other.usd'}})):
        changed = setup('/first/place'); changed[key] = value
        with pytest.raises(AssertionError, match=key):
            assert_same_simulator_setup(changed, setup('/first/place'))
