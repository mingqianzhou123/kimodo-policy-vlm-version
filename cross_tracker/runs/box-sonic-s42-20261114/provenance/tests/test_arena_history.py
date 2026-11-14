"""The bridge must preserve every measured frame, packet boundary and reset."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.arena_history import attach_history, fix_twist2_render_step


class HistoryTest(unittest.TestCase):
    def test_render_does_not_add_physics_steps(self):
        sim = SimpleNamespace(current_time=0.0, renders=0)
        def original_step(render=True):
            sim.current_time += 0.02 if render else 0.005
        def render_only():
            sim.renders += 1
        sim.step, sim.render = original_step, render_only
        fix_twist2_render_step(SimpleNamespace(sim=sim))
        for i in range(4):
            sim.step(render=i == 3)
        self.assertAlmostEqual(sim.current_time, 0.02)
        self.assertEqual(sim.renders, 1)

    def test_real_samples_and_reset(self):
        requests = []
        env = SimpleNamespace(sim=SimpleNamespace(current_time=0.0))
        client = SimpleNamespace(_post_json=lambda path, body: requests.append((path, body)) or {})
        state = lambda: np.full(64, env.sim.current_time, dtype=np.float32)
        provider = SimpleNamespace(_lerobot_http_client=client,
                                   _build_lerobot_vla_observation_state=state)
        provider.get_action = lambda env: None
        attach_history(provider)
        for step in range(26):
            env.sim.current_time = step * 0.02
            provider.get_action(env)
            if step in (0, 25):
                client._post_json('/infer', {'observation': {'state': state().tolist()}})
        np.testing.assert_allclose(np.array(requests[1][1]['observation']['state_history'])[:, 0],
                                   np.arange(1, 26) * 0.02)
        self.assertEqual(len(requests[0][1]['observation']['state_history']), 1)
        client._post_json('/reset', {})
        env.sim.current_time = 0.0
        provider.get_action(env)
        client._post_json('/infer', {'observation': {'state': state().tolist()}})
        self.assertEqual(len(requests[-1][1]['observation']['state_history']), 1)
        env.sim.current_time = 0.04
        with self.assertRaisesRegex(ValueError, '50 Hz'):
            provider.get_action(env)


if __name__ == '__main__':
    unittest.main()
