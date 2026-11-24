"""Grasp loop: the settled start without markers or audit, plus contact reporting on the box."""
import os
import runpy
from pathlib import Path
from arena import oracle_bridge
from arena.grasp_scene import install_grasp_scene, place_at_stance
from arena.settled_start import install

original = oracle_bridge.install_oracle_bridge


def install_all():
    install_grasp_scene()
    install(original, markers=False, audit=False,
            before=place_at_stance if os.environ.get('GRASP_START_AT_STANCE') == '1' else None)


oracle_bridge.install_oracle_bridge = install_all
runpy.run_path(str(Path(__file__).with_name('oracle_sim_bootstrap.py')), run_name='__main__')
