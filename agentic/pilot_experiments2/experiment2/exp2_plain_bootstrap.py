"""Experiment 2 replay from the reset pose, as the recording started: Arena's stock evaluation, either tracker.

No oracle bridge (it is SONIC-only) and no settled start (it is tuned for the aligned actuators; with
Arena's own actuators it never stabilizes), so Arena's own dynamics. Before the first control step the
objects are put where the recording had them (``exp2_replay.place_recorded``); every step's root pose and
tracked object pose go to ``execution.jsonl``. The replay server runs open loop (no measured history),
from recording frame 0.
"""
import importlib
import json
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENTIC = HERE.parents[1]
sys.path[:0] = [str(AGENTIC), str(HERE)]
from arena import oracle_bridge


def install():
    factory = importlib.import_module('action_provider.create_action_provider')
    original = factory.create_action_provider

    def create(env, args):
        provider = original(env, args)
        get_action = provider.get_action
        run = Path(os.environ['ORACLE_RUN'])
        track = os.environ.get('KVLA_TRACK_OBJECT', 'box')
        log = (run/'execution.jsonl').open('a', buffering=1)
        placed = False

        def state():
            root = env.scene['robot'].data.root_state_w[0].detach().cpu().numpy()
            obj = env.scene[track].data.root_state_w[0].detach().cpu().numpy()
            return {'time': float(env.sim.current_time), 'root_pos': root[:3].tolist(), 'root_quat': root[3:7].tolist(),
                    'box_pos': obj[:3].tolist(), 'box_quat': obj[3:7].tolist()}

        def act(current_env):
            nonlocal placed
            if not placed:
                from exp2_replay import place_recorded
                place_recorded(env, os.environ['KVLA_EXP2_EPISODE'], run)
                placed = True
            before = state()
            action = get_action(current_env)
            log.write(json.dumps({'before': before, 'after': state()})+'\n')
            return action

        provider.get_action = act
        return provider

    factory.create_action_provider = create


oracle_bridge.install_oracle_bridge = install
runpy.run_path(str(AGENTIC/'scripts/oracle_sim_bootstrap.py'), run_name='__main__')
