"""Simulator start-up for the pointing image set: the shared bootstrap with capture in place of control."""
import runpy
from pathlib import Path

from arena import oracle_bridge
from arena.pointing_capture import install

oracle_bridge.install_oracle_bridge = install
runpy.run_path(str(Path(__file__).with_name('oracle_sim_bootstrap.py')), run_name='__main__')
