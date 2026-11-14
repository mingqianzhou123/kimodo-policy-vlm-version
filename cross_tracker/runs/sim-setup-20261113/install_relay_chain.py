from pathlib import Path
import time,subprocess
r=Path('cross_tracker/runs/sim-setup-20261113')
end=time.monotonic()+7200
while not (r/'sim-wheels-import.json').exists():
 if time.monotonic()>end:raise TimeoutError('wheel import')
 time.sleep(10)
for name in ['install_runtime_r6.py','install_lab.py']:
 subprocess.run(['python',str(r/name)],check=True)
