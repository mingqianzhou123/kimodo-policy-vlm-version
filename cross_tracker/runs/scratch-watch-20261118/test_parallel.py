"""No-GPU checks for reservations, protocol preservation, and isolated retry handling."""
from pathlib import Path
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('episode_protocol', HERE / 'run_batch.py')
protocol = importlib.util.module_from_spec(spec)
sys.modules['episode_protocol'] = protocol
spec.loader.exec_module(protocol)
import run_parallel as runner
import handoff_r5 as handoff
import monitor_parallel as monitor
from parallel_state import assert_resumable, gpu_pairs, key, pending


class ParallelEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.tasks = list(protocol.tasks())
        self.state = dict(completed=[], failed=[], active=[], current=None)

    def test_all_480_cells_have_unique_reservations(self):
        self.assertEqual(len(self.tasks), 480)
        self.assertEqual(len({key(x) for x in self.tasks}), 480)
        for _ in range(160):
            for pair in ((6, 7), (0, 1), (2, 3)):
                ready, blocked = pending(self.state, self.tasks, {'sonic', 'twist2', 'mix', 'cct'}, lambda _: False)
                self.assertFalse(blocked)
                self.state['active'].append(dict(ready[0], gpu_pair=pair))
            self.state['completed'].extend(reversed(self.state['active']))
            self.state['active'] = []
        self.assertEqual(len({key(x) for x in self.state['completed']}), 480)
        self.assertEqual(pending(self.state, self.tasks, {'sonic', 'twist2', 'mix', 'cct'}, lambda _: False), ([], []))

    def test_real_150_receipts_are_not_rescheduled(self):
        audit = json.loads((HERE / 'progress-audit-20261119-1111.json').read_text())
        self.state['completed'] = audit['batch']['completed']
        ready, _ = pending(self.state, self.tasks, {'sonic', 'twist2', 'mix'}, lambda _: False)
        self.assertEqual(len(ready), 210)
        self.assertTrue(all(key(x) not in {key(y) for y in self.state['completed']} for x in ready))
        self.state['active'] = ready[:3]
        more, _ = pending(self.state, self.tasks, {'sonic', 'twist2', 'mix'}, lambda _: False)
        self.assertEqual(len(more), 207)
        self.assertTrue(set(map(key, more)).isdisjoint(map(key, self.state['active'])))

    def test_infrastructure_retry_does_not_block_other_cells(self):
        self.state['failed'] = [dict(self.tasks[0]) for _ in range(3)]
        ready, blocked = pending(self.state, self.tasks, {'sonic', 'twist2', 'mix'}, lambda _: False)
        self.assertEqual(len(blocked), 1)
        self.assertEqual(key(ready[0]), key(self.tasks[1]))
        ready, blocked = pending(self.state, self.tasks, {'sonic'}, lambda _: True)
        self.assertEqual(ready[0]['attempt'], 3)
        self.assertFalse(blocked)

    def test_overlapping_devices_and_duplicate_or_live_restarts_rejected(self):
        self.assertEqual(gpu_pairs([[6, 7], [0, 1], [2, 3]]), [(6, 7), (0, 1), (2, 3)])
        for value in ([[6, 7], [6, 7]], [[4, 5]], [], [[0, 7]]):
            with self.assertRaises(AssertionError):
                gpu_pairs(value)
        with self.assertRaises(AssertionError):
            assert_resumable(self.state, True)
        self.state['active'] = [self.tasks[0]]
        with self.assertRaises(AssertionError):
            assert_resumable(self.state, False)
        self.state['completed'] = [self.tasks[0]]
        with self.assertRaises(AssertionError):
            pending(self.state, self.tasks, {'sonic'}, lambda _: False)

    def test_only_gpu_flags_change_in_frozen_episode_commands(self):
        for task in self.tasks[:8]:
            item = dict(task, run_id='protocol-check')
            expected = protocol.command(item)
            for pair in ((6, 7), (0, 1), (2, 3)):
                actual = runner.command(dict(item, gpu_pair=pair))
                for flag in ('--sim-gpu', '--policy-gpu'):
                    index = actual.index(flag) + 1
                    actual[index] = expected[index]
                self.assertEqual(actual, expected)

    def test_malformed_receipt_is_retryable_not_a_scheduler_crash(self):
        process = Mock(returncode=0)
        process.poll.return_value = 0
        entry = dict(item={'run_id': 'bad-receipt'}, process=process, log=Mock())
        with patch.object(protocol, 'result_for', side_effect=ValueError('truncated receipt')):
            self.assertEqual(runner.poll(entry), 'failed')
        self.assertIn('Invalid receipt', entry['item']['error'])
        entry['log'].close.assert_called_once()

    def test_stall_signals_only_the_owned_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'code/cross_tracker/runs/stalled'
            folder.mkdir(parents=True)
            outer = root / 'stalled.log'
            outer.write_text('waiting')
            os.utime(outer, (0, 0))
            process = Mock(pid=12345)
            process.poll.return_value = None
            entry = dict(item={'run_id': 'stalled'}, process=process, log=Mock())
            with patch.object(runner, 'RUN', root), patch.object(runner, 'CODE', root / 'code'), \
                 patch.object(runner.time, 'time', return_value=1000), \
                 patch.object(runner.os, 'getpgid', return_value=12345), patch.object(runner.os, 'killpg') as kill:
                self.assertIsNone(runner.poll(entry))
                kill.assert_called_once_with(12345, runner.signal.SIGTERM)
                self.assertEqual(entry['kill_at'], 1020)
            with patch.object(runner.time, 'time', return_value=1021), patch.object(runner.os, 'killpg') as kill:
                self.assertIsNone(runner.poll(entry))
                kill.assert_called_once_with(12345, runner.signal.SIGKILL)

    def test_handoff_refuses_gpu_pair_while_finetune_is_running(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new = root / 'r5'
            new.mkdir()
            (new / 'prepare.json').write_text('{"status":"complete"}')
            (new / 'enabled-pairs.json').write_text('[[6,7],[0,1]]')
            training = root / 'runs/box-aligned-sonic-pt419h-s42-dev-20261118-r1'
            training.mkdir(parents=True)
            (training / 'launch.json').write_text('{"status":"running","stage":"training"}')
            with patch.object(handoff, 'ROOT', root), patch.object(handoff, 'NEW', new), \
                 patch.object(handoff.os, 'kill') as kill:
                with self.assertRaises(AssertionError):
                    handoff.main()
                kill.assert_not_called()

    def test_monitor_reports_stalled_secondary_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = dict(pid=-1, current={'run_id': 'primary'}, active=[
                {'run_id': 'primary'}, {'run_id': 'secondary'}], status='running',
                completed=[], failed=[], total_episodes=480, heartbeat_unix=1000)
            (root / 'batch.json').write_text(json.dumps(state))
            with patch.object(monitor, 'worker', side_effect=[
                dict(run_id='primary', issues=[]),
                dict(run_id='secondary', issues=['no_log_progress_for_10_minutes'])]):
                result = monitor.inspect(root)
            self.assertIn('secondary: no_log_progress_for_10_minutes', result['issues'])
            self.assertEqual(len(result['workers']), 2)


if __name__ == '__main__':
    unittest.main()
