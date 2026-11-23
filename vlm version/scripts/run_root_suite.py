"""Verify a fixed-actuator baseline replay, then run one five-reference factorial cell."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from kimodo_vla.oracle.recorded_reference import read_rows
from analyze_sonic_replay import audit
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cell',choices=['load','support','both'],required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--suffix',default='r1');a=p.parse_args()
 suite=f'root-{a.cell}-20261120-{a.suffix}';pilot=f'{suite}-control';fixture=ROOT/'runs/dynamics-actuator-20261119-r1/fixtures/base-center'
 cmd=[sys.executable,'scripts/run_oracle.py','--run-id',pilot,'--reference-source','replay','--replay-fixture',str(fixture),'--steps','400','--capture-vision','--visual-markers','--marker-reach-rad','.15','--camera-lateral-m','0','--sim-gpu',str(a.gpu),'--gpu-min-free-mib','16384','--actuator-audit','--dynamics-profile','actuator','--root-support','none']
 print('CONTROL',pilot,flush=True);subprocess.run(cmd,cwd=ROOT,check=True)
 out=audit(ROOT/'runs'/pilot,fixture,ROOT/'runs/vlm-refined-20261118-r1-base-center')
 x=read_rows(ROOT/'runs'/pilot/'execution.jsonl');y=read_rows(ROOT/'runs/dynamics-actuator-20261119-r1-base-center/execution.jsonl');delta={}
 for k in ('root_pos','root_quat','joints','wrist_pos'):
  d=float(np.max(np.abs(np.array([r['after'][k] for r in x])-np.array([r['after'][k] for r in y]))));delta[k]=d
  if d>1e-6:raise ValueError(('Baseline equivalence failed',k,d))
 out['baseline_max_deltas']=delta;out['gpu']=a.gpu
 (ROOT/'runs/root-dynamics-20261120'/f'{a.cell}-{a.suffix}-control-audit.json').write_text(json.dumps(out,indent=2))
 print('CONTROL PASS',delta,flush=True)
 profile='actuator' if a.cell=='support' else 'actuator-load';support='none' if a.cell=='load' else 'raised'
 subprocess.run([sys.executable,'scripts/run_sonic_replay_suite.py','--suite',suite,'--sim-gpu',str(a.gpu),'--gpu-min-free-mib','16384','--actuator-audit','--dynamics-profile',profile,'--root-support',support],cwd=ROOT,check=True)
 print('CELL COMPLETE',suite,flush=True)
if __name__=='__main__':main()
