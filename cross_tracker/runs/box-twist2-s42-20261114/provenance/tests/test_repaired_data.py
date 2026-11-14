"""Regression coverage for wrong labels, invalid masks, video offsets and sampling."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent)]
from repaired_data import checksum, read_segment, checked_table
from sampling import BalancedSampler


class RepairedDataTests(unittest.TestCase):
    def test_repair_mask_and_original_video_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "dataset/task/sonic_refpose_v3_1"
            (directory / "meta/episodes/chunk-000").mkdir(parents=True)
            prefix = "videos/observation.images.front/"
            info = {"fps": 50, "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
            (directory / "meta/info.json").write_text(json.dumps(info))
            pq.write_table(pa.Table.from_pylist([{"task": "test"}]), directory / "meta/tasks.parquet")
            meta = {"episode_index": 0, prefix+"chunk_index": 0, prefix+"file_index": 1,
                    prefix+"from_timestamp": 10., prefix+"to_timestamp": 10.12}
            pq.write_table(pa.Table.from_pylist([meta]), directory / "meta/episodes/chunk-000/file-000.parquet")
            keys = {"index": list(range(6)), "episode_index": [0]*6, "frame_index": list(range(6)),
                    "timestamp": [i/50 for i in range(6)]}
            raw = pa.table({**keys, "observation.state": [[0.]*64]*6, "action": [[99.]*40]*6})
            action = np.zeros((6, 40)); action[:, 9:38] = 0.25
            fixed = pa.table({**keys, "candidate.action": action.tolist(), "numeric_eligible": [False, False, True, True, True, True]})
            pq.write_table(raw, root / "source.parquet"); pq.write_table(fixed, root / "overlay.parquet")
            dataset = {"dataset": "task/sonic_refpose_v3_1",
                       "source": {"root": "audit", "path": "source.parquet", "sha256": checksum(root / "source.parquet")},
                       "overlay": {"root": "repair", "path": "overlay.parquet", "sha256": checksum(root / "overlay.parquet")}}
            assets = {"metadata": {dataset["dataset"]: {str(p.relative_to(directory)): checksum(p) for p in directory.rglob('*') if p.is_file()}}}
            roots = {"audit": str(root), "repair": str(root)}
            state, result, mapping = read_segment(dataset, {"episode": 0, "eligible_rows": [2, 6]}, roots, assets)
            np.testing.assert_array_equal(result, action[2:])
            self.assertEqual(state.shape, (4, 64))
            self.assertAlmostEqual(mapping["video_offset_s"], 10.04)
            self.assertEqual(mapping["source_frame_start"], 2)
            with self.assertRaises(ValueError):
                read_segment(dataset, {"episode": 0, "eligible_rows": [1, 6]}, roots, assets)
            with self.assertRaises(ValueError):
                read_segment(dataset, {"episode": 1, "eligible_rows": [2, 6]}, roots, assets)
        checked_table.cache_clear()

    def test_sampler_balances_tasks_sources_and_episode_lengths(self):
        records = []; index = []
        for task in ("a", "b"):
            for tracker in ("s", "m"):
                for count in (100, 4000):
                    e = len(records); records.append({"task": task, "tracker": tracker})
                    index.extend((e, i) for i in range(count))
        class Dataset(SimpleNamespace):
            def __len__(self): return len(self.index)
        dataset = Dataset(episodes=records, index=index)
        one = list(BalancedSampler(dataset, seed=42))
        self.assertEqual(one, list(BalancedSampler(dataset, seed=42)))
        self.assertNotEqual(one, list(BalancedSampler(dataset, seed=43)))
        counts = np.bincount([index[i][0] for i in one])
        np.testing.assert_allclose(counts/len(one), np.ones(8)/8, atol=.015, rtol=0)


if __name__ == "__main__":
    unittest.main()
