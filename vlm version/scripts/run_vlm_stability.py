"""Nine preregistered conditions, frozen backend, three deterministic inferences per image."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from kimodo_vla.vlm.stability import review_suite

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vlm-localizer',choices=['baseline','refined'],default='baseline')
    p.add_argument('--suite-id',required=True)
    p.add_argument('--policy-gpu',type=int,default=6);p.add_argument('--sim-gpu',type=int,default=1)
    a=p.parse_args();folder=ROOT/'runs'/a.suite_id
    if folder.resolve().parent!=(ROOT/'runs').resolve():raise ValueError('Invalid suite ID')
    folder.mkdir(exist_ok=False)
    preserved=json.loads((ROOT/'snapshots/vlm-before-refinement-20261118/manifest.json').read_text())
    # Only the image localizer dispatch and evaluator may differ. All motion
    # generation, projection, FK, conditions, camera capture and fixtures stay frozen.
    frozen=[n for n in preserved if (n.startswith(('kimodo_vla/','arena/')) and n not in
        ('kimodo_vla/vlm/policy.py','kimodo_vla/vlm/stability.py')) or n in
        ('scripts/infer_vlm_points.py','scripts/oracle_sim_bootstrap.py')]
    infer_script='infer_vlm_refined.py' if a.vlm_localizer=='refined' else 'infer_vlm_points.py'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    def check_frozen():
        for n in frozen:
            if sha(ROOT/n)!=preserved[n]:raise ValueError(f'Frozen backend changed: {n}')
    check_frozen()
    cases=[{'run_id':f'{a.suite_id}-{position}-{view}','position':position,'view':view,
            'reach_rad':reach,'lateral_m':lateral}
           for position,reach in [('base',.15),('near',.10),('far',.20)]
           for view,lateral in [('center',0.),('left',-.06),('right',.06)]]
    suite={'status':'running','cases':cases,'started_unix':time.time(),
           'frozen_sha256':{n:preserved[n] for n in frozen},'repeats_per_image':3,
           'vlm_localizer':a.vlm_localizer,
           'visual_source_sha256':{n:sha(ROOT/n) for n in ('scripts/'+infer_script,'kimodo_vla/vlm/refinement.py','kimodo_vla/vlm/policy.py')},
           'steps':400,'seed':1000,'same_image_repeats_not_independent':True,'grippers':'open'}
    def save():(folder/'suite.json').write_text(json.dumps(suite,indent=2))
    save()
    uuid=subprocess.check_output(['nvidia-smi',f'--id={a.policy_gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    repeat_env={**os.environ,'CUDA_VISIBLE_DEVICES':uuid,'PYTHONPATH':str(ROOT),'HF_HUB_OFFLINE':'1'}
    try:
        for case in cases:
            check_frozen();suite['active_case']=case['run_id'];save()
            print('START',case['run_id'],flush=True)
            cmd=[sys.executable,str(ROOT/'scripts/run_oracle.py'),'--run-id',case['run_id'],
                 '--vlm-localizer',a.vlm_localizer,'--mode','hold','--steps','400','--variant','bounded','--joint-speed','4',
                 '--support-constraints','root','--hand-constraints','continuous','--trajectory-projection','ik-hold',
                 '--policy-gpu',str(a.policy_gpu),'--sim-gpu',str(a.sim_gpu),'--gpu-min-free-mib','16384',
                 '--visual-markers','--vlm-model',str(ROOT/'checkpoints/Qwen3-VL-4B-Instruct'),
                 '--marker-reach-rad',str(case['reach_rad']),'--camera-lateral-m',str(case['lateral_m'])]
            with (folder/(case['run_id']+'.log')).open('x') as log:
                run=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)},stdout=log,stderr=subprocess.STDOUT)
            case['returncode']=run.returncode;case['repeats']=[];save()
            runpath=ROOT/'runs'/case['run_id']
            # Failed localization is a measured result, not a reason to omit the condition.
            if (runpath/'vlm-capture.npz').exists():
                for repeat in (2,3):
                    free=int(subprocess.check_output(['nvidia-smi',f'--id={a.policy_gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                    if free<16384:raise RuntimeError('Insufficient memory for inference repeat')
                    with (runpath/f'vlm-repeat-{repeat}.log').open('x') as log:
                        rr=subprocess.run([sys.executable,str(ROOT/'scripts'/infer_script),
                            '--model',str(ROOT/'checkpoints/Qwen3-VL-4B-Instruct'),
                            '--capture',str(runpath/'vlm-capture.npz'),'--out',str(runpath/f'vlm-repeat-{repeat}.json')],
                            env=repeat_env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
                    case['repeats'].append(rr.returncode);save()
            result=review_suite(folder)
            print('DONE',case['run_id'],'exit',run.returncode,'completed',result['completed_cases'],
                  'accepted',result['first_intent_accepted'],'/',result['first_intent_count'],flush=True)
        check_frozen();suite.update(status='completed',ended_unix=time.time());save();review_suite(folder)
    except BaseException as exc:
        suite.update(status='failed',error=repr(exc),ended_unix=time.time());save();raise
    print('SUITE COMPLETED',folder,flush=True)

if __name__=='__main__':main()
