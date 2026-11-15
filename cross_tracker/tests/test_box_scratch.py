"""Protect the initialization and budget boundary of scratch jobs."""
import argparse
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_box_scratch import command


class ScratchCommands(unittest.TestCase):
    def test_pilot_and_formal_never_load_policy_checkpoint(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config")
        parser.add_argument("--run-dir")
        parser.add_argument("--task", nargs="+")
        parser.add_argument("--device")
        parser.add_argument("overrides", nargs="*")
        for sources in (["sonic"], ["twist2"], ["sonic", "twist2"]):
            for pilot in (False, True):
                with self.subTest(sources=sources, pilot=pilot):
                    argv = command(ROOT / "runs/test", sources, pilot)
                    self.assertNotIn("--init-checkpoint", argv)
                    self.assertNotIn("--resume", argv)
                    parsed = parser.parse_args(argv[6:])
                    self.assertEqual(parsed.task, [f"HOI_pp_box/{s}" for s in sources])
                    self.assertIn(f"main.max_steps={10000*len(sources)}", parsed.overrides)
                    self.assertEqual("main.stop_after=200" in parsed.overrides, pilot)


if __name__ == "__main__":
    unittest.main()
