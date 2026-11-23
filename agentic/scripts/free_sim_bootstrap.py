import runpy
from pathlib import Path
from arena import oracle_bridge
from arena.free_start import install

original = oracle_bridge.install_oracle_bridge
oracle_bridge.install_oracle_bridge = lambda: install(original)
runpy.run_path(str(Path(__file__).with_name('oracle_sim_bootstrap.py')), run_name='__main__')
