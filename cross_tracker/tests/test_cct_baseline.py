"""A formal CCT run must not accept an unrelated or incomplete simulator receipt."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_cct_baseline import command, validate_simulation


class FormalCCTTests(unittest.TestCase):
    def test_simulation_gate_rejects_wrong_evidence(self):
        receipt = dict(controller="sonic", task="box", status="completed", passed=True,
                       checkpoint_sha256="checkpoint", context_manifest_sha256="bank",
                       seed=1000, diffusion_steps=10, episode_steps=50, video_frames=51,
                       successful_inference_requests=2, history_packet_sizes=[1, 25], exit_code=0)
        validate_simulation(receipt, "sonic", "checkpoint", "bank")
        for key, value in dict(controller="twist2", checkpoint_sha256="old", passed=False,
                               context_manifest_sha256="other", diffusion_steps=1,
                               episode_steps=0, video_frames=1, history_packet_sizes=[1, 1]).items():
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_simulation(dict(receipt, **{key: value}), "sonic", "checkpoint", "bank")

    def test_formal_job_starts_from_pretraining_not_pilot(self):
        argv = command(ROOT / "runs/box-cct-sonic-twist2-s42-20261115")
        self.assertIn("checkpoint_1000000", argv[argv.index("--init-checkpoint") + 1])
        self.assertNotIn("main.stop_after=200", argv)
        self.assertNotIn("--resume", argv)
        self.assertIn("--nproc_per_node=8", argv)


if __name__ == "__main__":
    unittest.main()
