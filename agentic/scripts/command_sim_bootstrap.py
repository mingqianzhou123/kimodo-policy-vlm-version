"""Command loop: the settled start without markers or audit, the task's contact reporting, and the
loop's simulator hooks (``arena/command_loop.py``). COMMAND_TASK is pp_box or sit_sofa."""
import os
import runpy
from pathlib import Path
from arena import oracle_bridge
from arena.settled_start import install

original = oracle_bridge.install_oracle_bridge


def install_all():
    from arena.command_loop import install_command_hooks
    task = os.environ['COMMAND_TASK']
    if task == 'pp_box':
        from arena.grasp_scene import install_grasp_scene
        install_grasp_scene()

    def bridge():
        original()
        install_command_hooks()
    # SitSofa starts the robot turned about the vertical; P&PBox at identity, as the grasp loop.
    install(bridge, markers=False, audit=False, any_heading=task != 'pp_box')


oracle_bridge.install_oracle_bridge = install_all
runpy.run_path(str(Path(__file__).with_name('oracle_sim_bootstrap.py')), run_name='__main__')
