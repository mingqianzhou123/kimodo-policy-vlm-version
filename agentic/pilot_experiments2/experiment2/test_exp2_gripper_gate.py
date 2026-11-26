"""The two-hand close gate re-checks the hand that fired first on the tick the second hand fires.

Bug (Codex review, 2026-11-26): ``SynchronizedGripperGate.update`` closed both hands as soon as the
second gate fired, without checking whether the first hand was still inside its tolerance on that
same tick. Carrying in experiment 2 starts from this grasp, so it is fixed before any carrying run.
"""
import numpy as np

from kimodo_vla.controllers.gripper_gate import SynchronizedGripperGate

EYE = np.eye(3)
TARGETS = [[0., .2, 1.], [0., -.2, 1.]]
FAR_RIGHT = [0., -.4, 1.]
OUT_LEFT = [0., .4, 1.]


def tick(gate, t, left=TARGETS[0], right=TARGETS[1]):
    return gate.update(now_s=t, observation_time_s=t, positions=[left, right],
                       rotations=[EYE, EYE], speeds_mps=[0., 0.])


def times(start, stop):
    return [round(t, 2) for t in np.arange(start, stop, .02)]


def test_first_hand_out_on_the_tick_the_second_fires_keeps_both_open():
    gate = SynchronizedGripperGate()
    gate.submit('round-0', TARGETS, [EYE, EYE], 5.)
    for t in times(0, .2):                    # left arrives and fires; right is far
        assert not tick(gate, t, right=FAR_RIGHT)
    assert gate.gates[0].status == 'fired'
    for t in times(.2, .3):                   # right arrives at 0.20 and dwells; left still inside
        assert not tick(gate, t)
    assert gate.gates[1].status == 'pending'
    assert not tick(gate, .3, left=OUT_LEFT)  # right's dwell completes on this tick, left is out
    assert gate.gates[1].status == 'fired'
    assert not gate.closed and gate.gates[0].status == 'pending'
    t = .32
    # left has to arrive again and dwell before both close
    states = [tick(gate, s) for s in times(t, t + .2)]
    assert not states[0] and any(states)


def test_both_still_close_together_when_both_stay_inside():
    gate = SynchronizedGripperGate()
    gate.submit('round-0', TARGETS, [EYE, EYE], 5.)
    for t in times(0, .2):
        tick(gate, t, right=FAR_RIGHT)
    states = [tick(gate, t) for t in times(.2, .4)]
    assert any(states) and gate.gates[0].status == gate.gates[1].status == 'fired'
