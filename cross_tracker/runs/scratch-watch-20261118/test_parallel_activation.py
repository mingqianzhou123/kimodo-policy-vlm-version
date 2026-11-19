"""Check that automatic activation cannot steal training GPUs or pause arbitrary owners."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import await_parallel_handoff as watcher


class ActivationTest(unittest.TestCase):
    def test_completed_receipt_and_final_file_are_both_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'runs/box-aligned-sonic-pt419h-s42-dev-20261118-r1'
            run.mkdir(parents=True)
            receipt = run / 'launch.json'
            receipt.write_text('{"status":"running","stage":"training"}')
            with patch.object(watcher, 'ROOT', root):
                self.assertFalse(watcher.training_complete(['sonic']))
                receipt.write_text('{"status":"completed","stage":"completed"}')
                self.assertFalse(watcher.training_complete(['sonic']))
                final = run / 'training/checkpoint_200000/training_state.pt'
                final.parent.mkdir(parents=True)
                final.write_bytes(b'test')
                self.assertTrue(watcher.training_complete(['sonic']))

    def test_busy_or_incomplete_gpu_query_cannot_pass_idle_check(self):
        with patch.object(watcher.subprocess, 'check_output', return_value='0, 0\n1644, 97\n'):
            self.assertFalse(watcher.gpu_idle((0, 1)))
        with patch.object(watcher.subprocess, 'check_output', return_value='0, 0\n'):
            with self.assertRaises(AssertionError):
                watcher.gpu_idle((0, 1))

    def test_unrelated_gpu_owner_is_never_paused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proof = root / 'idle-utility.json'
            proof.write_text(json.dumps(dict(sha256=hashlib.sha256(b'inspected utility').hexdigest())))
            def read_bytes(path):
                return b'inspected utility' if str(path) == '/root/bin/xiancun' else b'python\0train.py\0'
            with patch.object(watcher, 'WATCH', root), patch.object(Path, 'read_bytes', read_bytes), \
                 patch.object(watcher.subprocess, 'check_output', return_value='12345\n'), \
                 patch.object(watcher.subprocess, 'run') as run:
                self.assertFalse(watcher.yield_idle_utility((0, 1), {}))
                run.assert_not_called()

    def test_unknown_utility_is_never_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(watcher, 'WATCH', Path(directory)), patch.object(watcher.subprocess, 'run') as run:
                self.assertFalse(watcher.yield_idle_utility((0, 1), {}))
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
