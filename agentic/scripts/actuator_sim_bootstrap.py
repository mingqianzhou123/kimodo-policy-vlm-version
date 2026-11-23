"""Add read-only actuator measurements after installing the unchanged baseline bridge."""
import runpy
from pathlib import Path
from arena import oracle_bridge
original_install=oracle_bridge.install_oracle_bridge


def install():
    original_install()
    from arena.actuator_audit import install_actuator_audit
    install_actuator_audit()


oracle_bridge.install_oracle_bridge=install
runpy.run_path(str(Path(__file__).with_name('oracle_sim_bootstrap.py')),run_name='__main__')
