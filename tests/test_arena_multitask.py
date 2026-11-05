"""Protect the original seven-task sampling and evaluation contracts."""
import sys
from pathlib import Path

import pytest

from kimodo_vla.benchmarks.arena.training import ArenaOrdinalBatchSampler


@pytest.mark.parametrize("world,micro", [(4, 16), (4, 32), (8, 16)])
def test_rank_batches_cover_each_original_global_ordinal_once_and_resume(world, micro):
    streams = [list(ArenaOrdinalBatchSampler(0, 2, 128, micro, rank, world)) for rank in range(world)]
    accumulation = 128 // (micro * world)
    for step in range(2):
        combined = [ordinal for rank in range(world) for batch in streams[rank][step*accumulation:(step+1)*accumulation] for ordinal in batch]
        assert sorted(combined) == list(range(step*128, (step+1)*128))
    resumed = list(ArenaOrdinalBatchSampler(1, 2, 128, micro, 2, world))
    assert resumed == streams[2][accumulation:]
    late = next(iter(ArenaOrdinalBatchSampler(400000, 500000, 128, micro, 0, world)))
    assert late[0] == 400000*128  # not reduced modulo finite dataset length


def test_invalid_global_batch_is_rejected():
    with pytest.raises(ValueError):
        ArenaOrdinalBatchSampler(0, 10, 128, 24, 0, 4)


def test_hoi_protocol_uses_table2_scratch_and_original_task_limits():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    from eval_arena_doubledesk import TASKS
    assert [TASKS[t]["paper_sr"] for t in ["football", "doubledesk", "pp_box"]] == [.35, .4, .8]
    assert [TASKS[t]["max_steps"] for t in ["football", "doubledesk", "pp_box"]] == [2000, 2000, 1450]
    assert TASKS["pp_box"]["task"].endswith("Wholedoby")  # official registration spelling
