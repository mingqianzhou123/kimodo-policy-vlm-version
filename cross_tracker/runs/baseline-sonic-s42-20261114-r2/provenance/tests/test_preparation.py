"""Check split isolation and that registration cannot start/overwrite a run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freeze_data import save_frozen, split_episodes
import register_run


class PreparationTests(unittest.TestCase):
    def test_balanced_disjoint_stable_split(self):
        for total, n_train in ((54, 43), (100, 80), (94, 75)):
            ids = list(range(120))
            first = split_episodes(ids, total, 20261113, "task/source", .2)
            self.assertEqual(first, split_episodes(ids[::-1], total, 20261113, "task/source", .2))
            self.assertEqual(len(first["train"]), n_train)
            self.assertEqual(len(set(first["train"] + first["validation"])), total)

    def test_insufficient_or_duplicate_candidates_rejected(self):
        for ids in ([1, 2], [1, 1, 2, 3]):
            with self.assertRaises(ValueError):
                split_episodes(ids, 3, 1, "task/source", .2)

    def test_manifest_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            save_frozen(path, {"version": 1})
            save_frozen(path, {"version": 1})
            with self.assertRaises(FileExistsError):
                save_frozen(path, {"version": 2})
            self.assertEqual(json.loads(path.read_text()), {"version": 1})

    def test_run_registration_is_inert_and_preserves_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "dependency"
            repo.mkdir()
            def git(*args):
                subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
            git("init")
            (repo / "model.py").write_text("original = True\n")
            git("add", "model.py")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
            (repo / "model.py").write_text("original = False\n")
            for name in ("scripts", "tests"):
                (root / name).mkdir()
            config = {"source_groups": [["sonic"]]}
            (root / "config.json").write_text(json.dumps(config))
            (root / "manifest.json").write_text(json.dumps({"protocol": config}))
            marker = root / "MUST_NOT_EXIST"
            args = argparse.Namespace(run_id="test-run", stage="diagnostic", purpose="test", platform="local",
                                      seed=42, sources=["sonic"], code_repo=repo, config=root / "config.json",
                                      manifest=root / "manifest.json", parent_run=None,
                                      command=[sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"])
            with patch.object(register_run, "ROOT", root):
                run = register_run.register(args)
                with self.assertRaises(FileExistsError):
                    register_run.register(args)
                args.run_id = "../escape"
                with self.assertRaises(ValueError):
                    register_run.register(args)
            self.assertFalse(marker.exists())
            self.assertIn("original = False", (run / "provenance/dependency.patch").read_text())
            record = json.loads((run / "run.json").read_text())
            self.assertFalse(record["training_started"])
            self.assertIn("provenance/dependency.tar", record["sha256"])


if __name__ == "__main__":
    unittest.main()
