"""Run the unmodified Arena entry point with the compatible Warp and shutdown settings."""
import runpy
import sys
import warp
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

ChannelFactoryInitialize(1, "lo")
assert warp.__version__ == "1.9.0", warp.__version__
warp.init()
from isaacsim import SimulationApp
from isaaclab.app import AppLauncher

original_close = SimulationApp.close
original_launcher_init = AppLauncher.__init__


def close(self, wait_for_replicator=False):
    from isaaclab.sim import SimulationContext
    context = SimulationContext.instance()
    if context is not None:
        context._disable_app_control_on_stop_handle = True
    return original_close(self, wait_for_replicator=wait_for_replicator)


def launch(self, *args, **kwargs):
    original_launcher_init(self, *args, **kwargs)
    from isaaclab.envs import ManagerBasedEnv
    original_env_close = ManagerBasedEnv.close

    def close_env(env):
        env.sim._disable_app_control_on_stop_handle = True
        return original_env_close(env)

    ManagerBasedEnv.close = close_env


SimulationApp.close = close
AppLauncher.__init__ = launch
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
