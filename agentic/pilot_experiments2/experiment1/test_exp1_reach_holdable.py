"""Experiment 1 follow-up: reach, holdability and the support polygon."""
import numpy as np
import pytest

import reach_holdable as rh


def test_hull_and_signed_margin():
    square = rh.hull([[0, 0], [1, 0], [1, 1], [0, 1], [.5, .5]])
    assert len(square) == 4
    assert rh.support_margin([.5, .5], square) == pytest.approx(.5)
    assert rh.support_margin([.5, 1.2], square) == pytest.approx(-.2)
    assert rh.support_margin([1.3, 1.4], square) == pytest.approx(-.5)


def test_classes_in_order():
    assert rh.classify(-.01, -1.) == 'out_of_reach'
    assert rh.classify(.01, -.001) == 'not_holdable'
    assert rh.classify(.01, .001) == 'holdable'


@pytest.fixture(scope='module')
def model():
    pytest.importorskip('scipy')
    from paths import G1_XML
    from kimodo_vla.core.types import G1_JOINT_NAMES
    from kimodo_vla.oracle.geometry import XmlFK
    fk = XmlFK(G1_XML, G1_JOINT_NAMES)
    return fk, rh.Body(fk, G1_XML), rh.Body(fk, G1_XML, extra_hand_kg=1.)


def test_standing_pose_is_holdable_and_translation_invariant(model):
    fk, body, heavy = model
    q, root = np.zeros(len(fk.joint_names)), np.array([.3, -2., .79])
    margin = body.com_margin(q, root, np.eye(3))
    assert margin > .02                                     # straight standing: CoM well inside the feet
    assert body.com_margin(q, root + [5, 5, 0], np.eye(3)) == pytest.approx(margin)
    assert body.mass.sum() == pytest.approx(33.34, abs=.01) and heavy.mass.sum() == pytest.approx(35.34, abs=.01)
    tipped = body.com_margin(q, root, np.array([[np.cos(.6), 0, np.sin(.6)], [0, 1, 0], [-np.sin(.6), 0, np.cos(.6)]]))
    assert tipped < 0                                       # the whole body pitched 34 deg forward over pinned feet


def test_arm_length_and_reach(model):
    fk, body, _ = model
    lengths = [rh.arm_length(fk, s, elbows=31, others=10) for s in rh.SIDES]
    assert .35 < lengths[0] < .65 and lengths[0] == pytest.approx(lengths[1], abs=.02)
    q, root = np.zeros(len(fk.joint_names)), np.array([0., 0., .79])
    shoulders = body.shoulders(q, root, np.eye(3))
    near = rh.reach_margin(body, lengths, q, root, np.eye(3), shoulders + [.1, 0, 0], (0, 1))
    far = rh.reach_margin(body, lengths, q, root, np.eye(3), shoulders + [2., 0, 0], (0, 1))
    assert near > .2 and far < -1.
