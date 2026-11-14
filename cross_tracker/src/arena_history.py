"""Send real 50 Hz states between Arena requests to the history-conditioned Kimodo."""
import importlib
import json
import os
from pathlib import Path

import numpy as np


def attach_history(provider):
    client = provider._lerobot_http_client
    if client is None:
        raise ValueError('History bridge requires the HTTP evaluation route')
    get_action = provider.get_action
    post_json = client._post_json
    pending = []
    times = []
    last_time = None

    def sample_then_act(env):
        nonlocal last_time
        now = float(env.sim.current_time)
        if last_time is not None and not np.isclose(now - last_time, 0.02, atol=1e-6):
            raise ValueError(f'Expected one real state per 50 Hz step: {last_time} -> {now}')
        state = np.asarray(provider._build_lerobot_vla_observation_state(), dtype=np.float32)
        if state.shape != (64,) or not np.isfinite(state).all():
            raise ValueError('Invalid measured 64-D state')
        pending.append(state.copy())
        times.append(now)
        last_time = now
        return get_action(env)

    def post_with_history(path, payload):
        nonlocal last_time
        if path == '/infer':
            if not pending:
                raise ValueError('No measured state history for inference')
            np.testing.assert_allclose(pending[-1], payload['observation']['state'], atol=1e-6)
            payload['observation']['state_history'] = np.stack(pending).tolist()
        response = post_json(path, payload)
        if path == '/infer':
            trace = os.environ.get('CROSS_TRACKER_HISTORY_TRACE')
            if trace:
                with Path(trace).open('a') as f:
                    f.write(json.dumps({'times': times, 'states': payload['observation']['state_history']}) + '\n')
            pending.clear()
            times.clear()
        elif path == '/reset':
            pending.clear()
            times.clear()
            last_time = None
        return response

    provider.get_action = sample_then_act
    client._post_json = post_with_history
    return provider


def fix_twist2_render_step(env):
    # In Isaac Sim 5, step(render=True) advances a rendering interval rather
    # than one physics dt. TWIST2 calls it inside its physics substep loop.
    original_step = env.sim.step

    def step_once(render=True):
        original_step(render=False)
        if render:
            env.sim.render()

    env.sim.step = step_once


def install_history_bridge():
    factory = importlib.import_module('action_provider.create_action_provider')
    original = factory.create_action_provider

    def create(env, args):
        provider = original(env, args)
        if args.action_source == "twist2_wholebody":
            fix_twist2_render_step(env)
        return attach_history(provider)

    factory.create_action_provider = create
