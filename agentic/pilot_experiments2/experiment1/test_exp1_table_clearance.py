"""Experiment 1 follow-up, step 3: hand points, distance to the table box, and the order of touch and back-off."""
import numpy as np
import pytest

import table_clearance as tc


def test_open_hand_points_follow_the_dex3_chain():
    left, right = tc.hand_points('left', {}), tc.hand_points('right', {})
    assert left.shape == (10, 3)
    assert left[3] == pytest.approx([.1192 + .0458 + .03, .0046, .0285])        # index tip, open
    assert right[3] == pytest.approx([.1958, -.0046, .0285])                   # mirrored in y
    assert left[-1][1] < -.08 and right[-1][1] > .08                           # thumbs point at the other hand
    bent = tc.hand_points('left', {'left_hand_index_0_joint': -np.pi / 2})
    assert bent[2] == pytest.approx([.1192, .0046 - .0458, .0285])            # proximal link turned about z


def test_signed_distance_to_the_table_box():
    lo, hi = np.array([0., -1, 0]), np.array([1., 1, .745])
    d = tc.box_distance([[-.1, 0, .5], [.2, 0, .8], [.05, 0, .7], [.5, 0, .3]], lo, hi)
    assert d == pytest.approx([.1, .055, -.045, -.3])


def test_order_of_touch_and_backoff():
    tr = [{'t': t, 'clearance': c, 'backoff': b, 'z_above_top': .1, 'x_beyond_edge': .0}
          for t, c, b in ((0., .2, 0.), (1., .005, .01), (1.5, -.01, .06), (2., -.02, .2))]
    s = tc.summarize_trace(tr, 0.)
    assert s['order'] == 'touch_first' and s['touch_s'] == 1. and s['onset_s'] == 1.5
    assert s['min_clearance_m'] == -.02 and s['clearance_at_onset_m'] == -.01 and s['max_backoff_m'] == .2
    assert tc.summarize_trace([{**x, 'clearance': .3} for x in tr], 0.)['order'] == 'no_touch'
    assert tc.summarize_trace([{**x, 'backoff': 0.} for x in tr], 0.)['order'] == 'no_onset'
