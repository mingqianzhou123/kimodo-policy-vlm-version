"""Run only the remaining short diagnostics after the SONIC box loop passes."""
from pathlib import Path
import json
import subprocess
import sys

setup = Path(__file__).resolve().parent
root = setup.parents[1]
run_id = 'sim-closed-loop-20261113-r8'
assert json.loads((root / 'runs' / run_id / 'sonic/run.json').read_text())['status'] == 'completed'
for tracker, task in [('twist2','box'), ('sonic','boxing'), ('twist2','boxing'),
                      ('sonic','navigation'), ('twist2','navigation')]:
    command = [sys.executable, str(setup / 'check_closed_loop.py'), tracker,
               '--task', task, '--run-id', run_id]
    print('DIAGNOSTIC',tracker,task,flush=True)
    subprocess.run(command,check=True)
print('ALL_SIX_SHORT_DIAGNOSTICS_COMPLETED',flush=True)
