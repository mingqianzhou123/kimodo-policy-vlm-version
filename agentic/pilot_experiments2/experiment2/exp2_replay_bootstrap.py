"""Experiment 2 SONIC replay: the settled start, with the objects put where the official recording had them.

Before the robot settles, Arena's own seeded object reset runs with the recording's object seed
(``exp2_replay.place_recorded``), which is how Arena's replay restores a recorded layout. The
comparison with the recorded initial poses is made from the first control step's measured state
(``exp2_replay.py analyze``).
"""
import json
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENTIC = HERE.parents[1]
sys.path[:0] = [str(AGENTIC), str(HERE)]
from arena import oracle_bridge
from arena.settled_start import install
from exp2_replay import place_recorded

original = oracle_bridge.install_oracle_bridge


def any_heading():
    """SitSofa starts turned 90 degrees; the settled start then re-anchors at the placed heading."""
    w = json.loads(Path(os.environ['KVLA_EXP2_EPISODE']).read_text())['robot_start']['orientation'][0]
    return abs(abs(w)-1.) > 1e-3


oracle_bridge.install_oracle_bridge = lambda: install(
    original, markers=False, audit=False, any_heading=any_heading(),
    before=lambda env: place_recorded(env, os.environ['KVLA_EXP2_EPISODE'], os.environ['ORACLE_RUN']))
runpy.run_path(str(AGENTIC/'scripts/oracle_sim_bootstrap.py'), run_name='__main__')
