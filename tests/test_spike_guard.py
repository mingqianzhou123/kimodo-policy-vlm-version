"""train.py's SpikeGuard: which updates it drops."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train import SpikeGuard  # noqa: E402


def test_nothing_is_skipped_until_the_window_is_full():
    guard = SpikeGuard(factor=5, window=4)
    assert [guard.skip(loss) for loss in (1.0, 100.0, 1.0, 1.0)] == [False] * 4


def test_a_spike_is_skipped_and_kept_out_of_the_window():
    guard = SpikeGuard(factor=5, window=4)
    for loss in (1.0, 1.0, 1.0, 1.0):
        guard.skip(loss)
    assert guard.skip(6.0)
    assert not guard.skip(4.0)  # under 5 x the median of the accepted steps
    assert list(guard.recent) == [1.0, 1.0, 1.0, 4.0]
    assert guard.total == 1


def test_a_lasting_jump_is_taken_after_max_consecutive_skips():
    guard = SpikeGuard(factor=5, window=4, max_consecutive=2)
    for loss in (1.0, 1.0, 1.0, 1.0):
        guard.skip(loss)
    assert [guard.skip(10.0) for _ in range(4)] == [True, True, False, True]
    assert guard.total == 3
