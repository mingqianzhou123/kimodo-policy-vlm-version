"""Experiment 2 replay: the settled start, with the objects put where the official recording had them.

Before the robot settles, Arena's own seeded object reset (``apply_deterministic_object_resets_with_seed``)
runs with the recording's object seed, which is how Arena's replay restores a recorded layout. What it
applied and where the objects stand afterwards go to ``layout-applied.json``; the comparison with the
recorded initial poses is made from the first control step's measured state (exp2_replay.py analyze).
"""
import json
import os
import runpy
import sys
from pathlib import Path

AGENTIC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENTIC))
from arena import oracle_bridge
from arena.settled_start import install

original = oracle_bridge.install_oracle_bridge


def place_recorded(env):
    from common_env_objects import apply_deterministic_object_resets_with_seed
    episode = json.loads(Path(os.environ['KVLA_EXP2_EPISODE']).read_text())
    applied = apply_deterministic_object_resets_with_seed(env.cfg, env, episode_seed=int(episode['seed']),
                                                          seed_source='recorded')
    placed = {}
    for name in episode['objects']:
        try:
            state = env.scene[name].data.root_state_w[0].detach().cpu().numpy()
        except KeyError:
            continue
        placed[name] = {'position': state[:3].tolist(), 'orientation': state[3:7].tolist()}
    (Path(os.environ['ORACLE_RUN'])/'layout-applied.json').write_text(json.dumps(
        {'seed': episode['seed'], 'applied': applied, 'scene_state_after_write': placed,
         'recorded': episode['objects']}, indent=1))


def any_heading():
    """SitSofa starts turned 90 degrees; the settled start then re-anchors at the placed heading."""
    w = json.loads(Path(os.environ['KVLA_EXP2_EPISODE']).read_text())['robot_start']['orientation'][0]
    return abs(abs(w)-1.) > 1e-3


oracle_bridge.install_oracle_bridge = lambda: install(original, markers=False, audit=False, before=place_recorded,
                                                      any_heading=any_heading())
runpy.run_path(str(AGENTIC/'scripts/oracle_sim_bootstrap.py'), run_name='__main__')
