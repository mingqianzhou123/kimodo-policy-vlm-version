"""Hammer grasp (DoubleDesk): the settled start without markers or audit, a contact report on
the hammer, and the robot and hammer placed by the layout before settling."""
import runpy
from pathlib import Path
from arena import oracle_bridge
from arena.hammer_scene import install_hammer_scene, place_hammer_layout
from arena.settled_start import install

original = oracle_bridge.install_oracle_bridge


def install_all():
    install_hammer_scene()
    install(original, markers=False, audit=False, before=place_hammer_layout, any_heading=True)


oracle_bridge.install_oracle_bridge = install_all
runpy.run_path(str(Path(__file__).with_name('oracle_sim_bootstrap.py')), run_name='__main__')
