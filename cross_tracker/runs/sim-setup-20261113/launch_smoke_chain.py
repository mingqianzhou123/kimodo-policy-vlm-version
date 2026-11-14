from pathlib import Path
import subprocess
r=Path('cross_tracker/runs/sim-setup-20261113')
subprocess.run(['python',str(r/'repair_sim_dependencies_r2.py')],check=True)
subprocess.run(['cross_tracker/data/cache/envs/isaaclab-2.2/bin/python',str(r/'check_controller_weights.py')],check=True)
subprocess.run(['python',str(r/'run_smoke.py')],check=True)
