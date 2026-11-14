"""Check the actual trainer argument boundary and independent DDP source sampling."""
import argparse
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT.parent)]
from run_box_screen import command, read, preflight
from sampling import BalancedSampler


class BoxScreenTests(unittest.TestCase):
    def test_preflight_does_not_inspect_other_running_jobs(self):
        def fixture(path):
            if path.name == "plan.json":
                return {"run_ids": ["one", "already-running"]}
            if path.name in ("dependency-files.json", "deployment-files.json"):
                return {}
            if path.name == "assets-verified.json":
                return {"passed": True, "prepared_episodes": 108, "manifest_sha256": "hash", "files": {}}
            self.assertEqual(path.parent.name, "one")
            return {"platform": "sensecore", "seed": 42, "status": "planned", "sha256": {}}
        with patch("run_box_screen.read", side_effect=fixture), patch("run_box_screen.digest", return_value="hash"):
            self.assertEqual(preflight("one"), "one")
            with self.assertRaises(AssertionError):
                preflight("unknown")

    def test_three_recipes_keep_task_and_sample_budget(self):
        protocol = read(ROOT / "configs/box-screen.json")
        parser = argparse.ArgumentParser()
        parser.add_argument("--config")
        parser.add_argument("--run-dir")
        parser.add_argument("--init-checkpoint")
        parser.add_argument("--task", nargs="+")
        parser.add_argument("--device")
        parser.add_argument("overrides", nargs="*")
        for sources in protocol["source_groups"]:
            argv = command(ROOT / "runs/test", protocol, sources)
            args = parser.parse_args(argv[6:])
            self.assertEqual(args.task, [f"HOI_pp_box/{s}" for s in sources])
            steps = int(next(s.split("=")[1] for s in args.overrides if s.startswith("main.max_steps=")))
            self.assertEqual(steps * protocol["batch_size"] // len(sources), 80000)
        self.assertEqual(protocol["batch_size"], protocol["world_size"] * protocol["micro_batch_size"])

    def test_ddp_ranks_are_reproducible_and_balance_sources(self):
        # Very unequal episode lengths must not give a source more sampling weight.
        class Dataset:
            episodes = [{"task": "box", "tracker": s} for s in ("sonic", "twist2")]
            index = [(0, i) for i in range(1000)] + [(1, i) for i in range(39000)]
            def __len__(self):
                return len(self.index)
        dataset = Dataset()
        streams = [list(BalancedSampler(dataset, 8, rank, 42)) for rank in range(8)]
        self.assertEqual(streams[0], list(BalancedSampler(dataset, 8, 0, 42)))
        self.assertEqual(len({tuple(s[:20]) for s in streams}), 8)
        for stream in streams:
            fraction = sum(i < 1000 for i in stream) / len(stream)
            self.assertTrue(.45 < fraction < .55, fraction)


if __name__ == "__main__":
    unittest.main()
