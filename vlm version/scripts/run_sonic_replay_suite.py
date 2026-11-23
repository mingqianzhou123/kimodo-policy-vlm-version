"""Replay five archived failures, stopping immediately on any reference/environment mismatch."""
import argparse,json,subprocess,sys
from pathlib import Path
from kimodo_vla.oracle.recorded_reference import export_reference
from analyze_sonic_replay import summarize,audit
ROOT=Path(__file__).resolve().parents[1]
CASES=('base-center','base-right','far-center','far-left','far-right')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',default='sonic-replay-20261118-r1')
    p.add_argument('--source-prefix',default='vlm-refined-20261118-r1')
    p.add_argument('--sim-gpu',type=int,default=7)
    p.add_argument('--gpu-min-free-mib',type=int)
    p.add_argument('--actuator-audit',action='store_true')
    p.add_argument('--dynamics-profile',choices=['sensor-load','actuator','actuator-load'])
    p.add_argument('--root-support',choices=['none','raised'])
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    suite=ROOT/'runs'/a.suite
    if suite.resolve().parent!=(ROOT/'runs').resolve():raise ValueError('Invalid suite name')
    suite.mkdir(exist_ok=True)
    results=[]
    for case in CASES:
        source=ROOT/'runs'/f'{a.source_prefix}-{case}'
        fixture=suite/'fixtures'/case
        if not fixture.exists():export_reference(source,fixture)
        original=summarize(source)
        (suite/f'{case}-original-analysis.json').write_text(json.dumps(original,indent=2))
        if a.prepare_only:continue
        m=json.loads((fixture/'replay.json').read_text());run_id=f'{a.suite}-{case}';run=ROOT/'runs'/run_id
        if not run.exists():
            cmd=[sys.executable,str(ROOT/'scripts/run_oracle.py'),'--run-id',run_id,
                 '--reference-source','replay','--replay-fixture',str(fixture),'--sim-gpu',str(a.sim_gpu),
                 '--steps',str(m['consumed_frames']),'--capture-vision','--visual-markers',
                 '--marker-reach-rad',str(m['marker_reach_rad']),'--camera-lateral-m',str(m['camera_lateral_m'])]
            if a.actuator_audit:cmd+=['--actuator-audit']
            if a.dynamics_profile:cmd+=['--dynamics-profile',a.dynamics_profile]
            if a.root_support:cmd+=['--root-support',a.root_support]
            if a.gpu_min_free_mib is not None:cmd+=['--gpu-min-free-mib',str(a.gpu_min_free_mib)]
            print('START',case,flush=True);subprocess.run(cmd,check=True,cwd=ROOT)
        if json.loads((run/'run.json').read_text())['status']!='completed':raise ValueError('Incomplete run; use a fresh suite ID')
        result=summarize(run);result['replay_audit']=audit(run,fixture,source)
        (suite/f'{case}-replay-analysis.json').write_text(json.dumps(result,indent=2))
        results.append(result)
        (suite/'summary.json').write_text(json.dumps(results,indent=2))
        print('PASS',case,'right tracking cm',result['post_arrival']['tracking']['mean_norm'][1],flush=True)
    print('PREPARED' if a.prepare_only else 'COMPLETE',len(CASES),flush=True)

if __name__=='__main__':main()
