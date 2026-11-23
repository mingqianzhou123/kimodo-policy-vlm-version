"""Where the agentic line keeps things.

Code never names a machine. Everything outside the repository is found through
three environment variables, set once per machine (see agentic/README.md):

  KVLA_SIM_CACHE     simulator environment, HumanoidArena source, SONIC weights  (required to run)
  KVLA_CHECKPOINTS   model weights; defaults to <repo>/checkpoints
  KVLA_AGENTIC_RUNS  experiment outputs; defaults to <repo>/agentic/runs
"""
import os
from pathlib import Path

AGENTIC = Path(__file__).resolve().parents[1]
REPO = AGENTIC.parent
G1_XML = REPO/'kimodo_vla/skeleton/assets/g1skel34/xml/g1.xml'
RUNS = Path(os.environ.get('KVLA_AGENTIC_RUNS') or AGENTIC/'runs')
CHECKPOINTS = Path(os.environ.get('KVLA_CHECKPOINTS') or REPO/'checkpoints')


def sim_cache():
    value = os.environ.get('KVLA_SIM_CACHE')
    if not value:
        raise SystemExit('Set KVLA_SIM_CACHE to the directory that holds envs/, sources/ and sim-assets/')
    return Path(value)
