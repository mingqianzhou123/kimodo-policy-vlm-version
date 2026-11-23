import pytest
from arena.settled_start import stable


@pytest.mark.parametrize('change', [
    {'vertical_force_n': [0., 220.]}, {'ankle_height_m': [.0558, .0558]},
    {'root_speed_m_s': .021}, {'root_angular_speed_rad_s': .11},
    {'foot_speed_m_s': [.001, .011]}, {'tilt_deg': 20.},
])
def test_stability_requires_both_contacts_and_low_motion(change):
    standing = dict(vertical_force_n=[175., 182.], ankle_height_m=[.035, .035],
                    root_speed_m_s=.004, root_angular_speed_rad_s=.01,
                    foot_speed_m_s=[.001, .002], root_height_m=.778, tilt_deg=2.)
    assert stable(standing)
    assert not stable({**standing, **change})
