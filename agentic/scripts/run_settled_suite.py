"""Regenerate one or more of the five archived visual reaches on original ground."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from paths import AGENTIC as ROOT, RUNS
CASES = ('base-center', 'base-right', 'far-center', 'far-left', 'far-right')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    p.add_argument('--gpu', required=True, type=int)
    p.add_argument('--suffix', default='r1')
    args = p.parse_args()
    for case in args.cases:
        old = RUNS/('vlm-refined-20261118-r1-'+case)
        truth = json.loads((old/'visual-evaluation-truth.json').read_text())
        run_id = 'settled-20261123-'+args.suffix+'-'+case
        print('START', run_id, flush=True)
        subprocess.run([sys.executable, 'scripts/run_oracle.py', '--run-id', run_id,
            '--settled-goal', str(old/'target.json'),
            '--sim-gpu', str(args.gpu), '--policy-gpu', str(args.gpu), '--share-policy-sim-gpu',
            '--marker-reach-rad', str(truth['reach_rad']),
            '--camera-lateral-m', str(truth['camera_lateral_m'])], cwd=ROOT, check=True)
        print('COMPLETE', run_id, flush=True)
