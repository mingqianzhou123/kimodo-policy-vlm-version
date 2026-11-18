"""Three final scratch models x two trackers x 60 fixed official scenes, on GPUs 6/7."""
from pathlib import Path
import hashlib,json,os,subprocess,time
RUN=Path(__file__).resolve().parent
ROOT=Path('/root/workspace/humanoidVLA')
CACHE=ROOT/'cross_tracker/data/cache'
CODE=RUN/'code'
PY=CACHE/'envs/policy/bin/python'
TASK='Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby'
MODELS=[('sonic','sonic'),('twist2','twist2'),('mix','sonic-twist2')]
# Round-robin models/controllers at each scene so partial results stay comparable.
COMBINATIONS=[('sonic','sonic'),('twist2','twist2'),('mix','sonic'),('mix','twist2'),('sonic','twist2'),('twist2','sonic')]

def save(value):
    path=RUN/'batch.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    assert not (RUN/'batch.json').exists()
    assert json.loads((RUN/'prepare.json').read_text())['status']=='complete'
    sources=json.loads((RUN/'source-files.json').read_text())
    assets=json.loads((RUN/'shared-eval-assets.json').read_text())
    aliases=dict(MODELS)
    state=dict(status='running',pid=os.getpid(),started_unix=time.time(),platform='Baidu T2M-PLUS dv-e21ddf3a87',sim_gpu=6,policy_gpu=7,total_episodes=360,completed=[],failed=[],current=None,protocol=dict(groups=[0,1,2],repeats=20,task_seed_name=TASK,fresh_sim=True,deterministic=True,timing='original',env_preset='official',max_steps=1450,checkpoint_step=200000,twist2_physics_timing='recorded',twist2_sim_device='cpu',twist2_scope='adapted tracker evaluation; not a claimed published TWIST2 baseline'))
    save(state)
    try:
        for group in range(3):
            for repeat in range(20):
                seed=int.from_bytes(hashlib.sha256(f'{TASK}|{group}|{repeat}'.encode()).digest()[-4:],'big')&0x7fffffff
                for method,controller in COMBINATIONS:
                    if any(f['method']==method and f['controller']==controller for f in state['failed']):continue
                    for name,digest in sources.items():assert sha(CODE/name)==digest,name
                    for name,digest in assets.items():assert sha(CACHE/name)==digest,name
                    ident=f'scratch200k-{method}-to-{controller}-g{group}-e{repeat}-20261118-r1'
                    cmd=[str(PY),'-u',str(CODE/'cross_tracker/scripts/evaluate_box.py'),aliases[method],controller,'--run-id',ident,'--sim-gpu','6','--policy-gpu','7','--model-manifest',str(RUN/f'{method}-manifest.json'),'--scene-seeds',str(seed),'--group-seed',str(group),'--repeat-index',str(repeat),'--timing','original','--fresh-sim','--deterministic','--env-preset','official','--robot-usd',str(CACHE/'sim-assets/arena/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd'),'--post-termination-steps','10']
                    if controller=='sonic':cmd+=['--refresh-video']
                    else:
                        i=cmd.index('--robot-usd');del cmd[i:i+2]
                        cmd+=['--sim-device','cpu','--twist2-physics-timing','recorded']
                    item=dict(method=method,controller=controller,group=group,repeat=repeat,seed=seed,run_id=ident,command=cmd,started_unix=time.time())
                    state['current']=item;save(state)
                    print('START',ident,flush=True)
                    with (RUN/f'{ident}.log').open('x') as log:
                        p=subprocess.Popen(cmd,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':str(CODE)})
                        item['pid']=p.pid;save(state);item['exit_code']=p.wait()
                    report=CODE/'cross_tracker/runs'/ident/'run.json'
                    result=json.loads(report.read_text()) if report.exists() else {}
                    if item['exit_code']!=0 or result.get('status')!='completed' or not result.get('valid_for_success_rate'):
                        item['error']=result.get('error','Missing or invalid run receipt');state['failed'].append(item)
                        print('FAILED',ident,item['error'],flush=True)
                    else:
                        assert len(result['episodes'])==1 and result['episodes'][0]['seed']==seed
                        item.update(result['episodes'][0]);state['completed'].append(item)
                        print('DONE',ident,item['success'],item['steps'],flush=True)
                    state['current']=None;save(state)
                    time.sleep(5)  # Let CUDA contexts close before the next idle-GPU guard.
        state['status']='completed' if not state['failed'] else 'partial_failure'
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:
        state['finished_unix']=time.time();save(state)
if __name__=='__main__':main()
