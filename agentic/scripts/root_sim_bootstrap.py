"""Frozen SONIC reference/actuators, independent load and support diagnostics."""
import runpy
from pathlib import Path
from arena import oracle_bridge
original_install=oracle_bridge.install_oracle_bridge

def install():
    original_install()
    from arena.actuator_audit import install_actuator_audit
    from arena.dynamics_override import install_dynamics_override
    from arena.root_support import install_root_support
    install_actuator_audit();install_dynamics_override();install_root_support()

oracle_bridge.install_oracle_bridge=install
runpy.run_path(str(Path(__file__).with_name('oracle_sim_bootstrap.py')),run_name='__main__')
