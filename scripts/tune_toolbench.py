"""Persistent staged ToolBench search. Capacity/loss fixed; no policy checkpoint initialization.

Selection: primary=macro L1/L2, secondary=macro all18. All tuning uses development
seeds; canonical test seeds are evaluated only after the configuration is locked.
"""
from __future__ import annotations
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import shutil
import subprocess
import sys
import threading
import time
from eval_toolbench import REPO, stop
from run_config import load_yaml
from toolbench_results import write_stage_reports


def read(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,value):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)


def reusable_training_prefix(actual,expected,step):
    """A stop limit may differ only when the LR horizon is explicitly fixed.

    Do not rewrite saved configs to manufacture cache identity. All other fields,
    including data, seed, model, optimizer and sampling must agree exactly.
    """
    import copy
    actual,expected=copy.deepcopy(actual),copy.deepcopy(expected)
    if step<=0 or min(actual['main']['max_steps'],expected['main']['max_steps'])<step:return False
    a=actual['training']['scheduler'];b=expected['training']['scheduler']
    if 'decay_steps' not in a or 'decay_steps' not in b:return actual==expected
    actual['main'].pop('max_steps');expected['main'].pop('max_steps')
    return actual==expected


class AdoptedTraining:
    """Monitor the existing torchrun group without restarting a single update."""
    def __init__(self,pid,checkpoint):self.pid=pid;self.checkpoint=checkpoint;self.returncode=None
    def poll(self):
        if self.returncode is not None:return self.returncode
        try:alive=Path(f'/proc/{self.pid}/stat').read_text().split(') ')[1].split()[0]!='Z'
        except FileNotFoundError:alive=False
        if not alive:
            self.returncode=0 if (self.checkpoint/'training_state.pt').exists() and (self.checkpoint/'optimizer_state.pt').exists() else 1
        return self.returncode
    def wait(self,timeout=None):
        start=time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic()-start>timeout:raise subprocess.TimeoutExpired('adopted training',timeout)
            time.sleep(.1)
        return self.returncode


class Search:
    def __init__(self,root,*,migration_from=None):
        self.root=root.resolve();self.root.mkdir(parents=True,exist_ok=True)
        if migration_from is not None:
            proc=Path(f'/proc/{migration_from}')
            assert proc.joinpath('stat').read_text().split(') ')[1].split()[0]=='T', 'old scheduler must be stopped, not running'
            command=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
            assert 'scripts/tune_toolbench.py' in command and str(self.root) in command
        self.lockfile=(self.root/('migration.lock' if migration_from is not None else 'search.lock')).open('w');fcntl.flock(self.lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.mutex=threading.RLock();self.children=[];self.stopping=False
        self.gpu_condition=threading.Condition();self.gpu_busy=set();self.adoption=None
        self.state=read(self.root/'status.json') if (self.root/'status.json').exists() else dict(started=now(),trials=[],active={})
        self.state.update(pid=os.getpid(),phase='preflight',error=None,active={})
        self.state['validating']={}
        self.state.setdefault('launches',[]).append(dict(pid=os.getpid(),started=now()))
        self.groups=['0,1,2,3','4,5,6,7']
        self.base=load_yaml(REPO/'configs/toolbench_tuning.yaml')
        self.invariants=dict(model=json.loads((REPO/'log/toolbench/joint18_scratch40k_20261027_8gpu/config.json').read_text())['model'],loss=dict(self.base.training.loss))
        self.source_hash={str(p.relative_to(REPO)):digest(p) for p in (REPO/'kimodo_vla').rglob('*.py')}
        self.adapter_sha256=hashlib.sha256(json.dumps(self.source_hash,sort_keys=True).encode()).hexdigest()
        if (self.root/'plan.json').exists():
            assert read(self.root/'plan.json')['core_source_sha256']==self.source_hash,'core source changed since registration'
        self.protocol_hash=read(self.root/'plan.json').get('protocol_source_sha256',{})
        self.provisional_sampling=read(self.root/'plan.json').get('provisional_sampling')
        self.last_telemetry=0
        self.env=os.environ.copy();self.env.update(PYTHONPATH=str(REPO),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1')
        for name in ['training','evaluation','logs','configs']:(self.root/name).mkdir(exist_ok=True)
        self.persist()

    def persist(self):
        with self.mutex:
            self.state['heartbeat']=now();atomic(self.root/'status.json',self.state)
            (self.root/'progress.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>ToolBench search</title><style>body{font:16px system-ui;margin:2rem}pre{white-space:pre-wrap}</style><h1>ToolBench tuning</h1><pre>'+__import__('html').escape(json.dumps(self.state,indent=2))+'</pre>')

    def phase(self,name):
        assert all(digest(REPO/p)==h for p,h in self.source_hash.items()), 'core source changed during search'
        assert all(digest(REPO/p)==h for p,h in self.protocol_hash.items()), 'training/evaluation protocol changed during search'
        self.state['phase']=name;self.persist();print(now(),name,flush=True)

    def acquire_gpus(self,preferred,all_gpus=False):
        with self.gpu_condition:
            while True:
                if self.stopping:raise RuntimeError('search stopping while waiting for GPUs')
                if all_gpus:
                    if not self.gpu_busy:self.gpu_busy.update([0,1]);return [0,1]
                else:
                    for slot in [preferred,1-preferred]:
                        if slot not in self.gpu_busy:self.gpu_busy.add(slot);return [slot]
                self.gpu_condition.wait(timeout=1)

    def release_gpus(self,slots):
        with self.gpu_condition:
            self.gpu_busy.difference_update(slots);self.gpu_condition.notify_all()

    def retire_old_supervisor(self):
        if not self.adoption or self.adoption.get('retired'):return
        pid=self.adoption['old_supervisor'];proc=Path(f'/proc/{pid}')
        if proc.exists() and proc.joinpath('stat').read_text().split(') ')[1].split()[0]!='Z':
            assert proc.joinpath('stat').read_text().split(') ')[1].split()[0]=='T'
            # Never resume the frozen old scheduler: its worker threads can
            # launch queued evaluations before the main thread handles SIGTERM.
            # Training has been adopted and completed/stopped by this owner;
            # only the frozen scheduler is retired, without running its queue.
            os.kill(pid,signal.SIGKILL)
            deadline=time.monotonic()+60
            while proc.exists() and time.monotonic()<deadline:
                try:
                    if proc.joinpath('stat').read_text().split(') ')[1].split()[0]=='Z':break
                except FileNotFoundError:break
                time.sleep(.2)
            if proc.exists():assert proc.joinpath('stat').read_text().split(') ')[1].split()[0]=='Z','old supervisor did not retire'
        self.adoption['retired']=True
        if Path(self.lockfile.name).name=='migration.lock':
            lock=(self.root/'search.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.lockfile.close();self.lockfile=lock
        atomic(self.root/'performance_handoff_complete.json',dict(self.adoption,finished=now()))
        self.persist()

    def process(self,command,name,slot,training=None,extra_env=None):
        if self.stopping:raise RuntimeError('search stopping')
        is_eval=name.startswith('eval_')
        all_gpus=is_eval and command[command.index('--gpus')+1]==','.join(self.groups)
        leased=self.acquire_gpus(slot,all_gpus);slot=leased[0]
        command=list(command)
        if is_eval:
            command[command.index('--gpus')+1]=','.join(self.groups) if all_gpus else self.groups[slot]
            command[command.index('--base-port')+1]=str(30000+slot*200)
            (self.root/'evaluation'/name.removeprefix('eval_')/'simulation_complete.json').unlink(missing_ok=True)
        env=dict(self.env,CUDA_VISIBLE_DEVICES=self.groups[slot]);env.update(extra_env or {})
        with (self.root/'logs'/f'{name}.log').open('a') as log:
            adopted=bool(training and self.adoption and str(training.resolve())==self.adoption['training_run'] and not self.adoption.get('retired'))
            if adopted:
                assert leased==[0], 'adopted training must retain its original physical GPU group'
                child=AdoptedTraining(self.adoption['training_pid'],training/f"checkpoint_{self.adoption['target_step']}")
            else:child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with self.mutex:self.children.append(child);self.state['active'][str(slot)]={'job':name,'pid':child.pid,'started':now()}
            latest=-1;progress=time.monotonic()
            try:
                while child.poll() is None:
                    if self.stopping:raise RuntimeError('search stopping')
                    item={'job':name,'pid':child.pid}
                    if training and (training/'metrics.jsonl').exists():
                        try:m=json.loads((training/'metrics.jsonl').read_text().splitlines()[-1])
                        except (json.JSONDecodeError,IndexError):m=None
                        if m:
                            if not all(math.isfinite(v) for v in m.values() if isinstance(v,(float,int))):raise RuntimeError('nonfinite training metric')
                            if m['step']>latest:latest=m['step'];progress=time.monotonic()
                            item.update(step=latest,loss=m['loss'],seconds_per_step=m['s_per_step'])
                    if training and time.monotonic()-progress>1200:raise RuntimeError('20 minutes without training progress')
                    if name.startswith('eval_'):
                        out=self.root/'evaluation'/name.removeprefix('eval_')
                        item.update(episodes=len(list(out.glob('*/run-*/summaries/episode_*.json'))),validated=len(list(out.glob('*/run-*/benchmark_result.json'))))
                        if (out/'simulation_complete.json').exists() and leased:
                            with self.mutex:
                                self.state['active'].pop(str(slot),None)
                                self.state['validating'][name]=item
                            self.release_gpus(leased);leased=[]
                    with self.mutex:
                        if leased:self.state['active'][str(slot)]=item
                        elif is_eval:self.state['validating'][name]=item
                        if time.monotonic()-self.last_telemetry>60:
                            if shutil.disk_usage(self.root).free<80*1024**3:raise RuntimeError('less than 80 GiB free; pause search before exhausting storage')
                            gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
                            with (self.root/'gpu.jsonl').open('a') as f:f.write(json.dumps({'time':now(),'gpus':gpu.stdout,'returncode':gpu.returncode})+'\n')
                            self.last_telemetry=time.monotonic()
                        self.persist()
                    time.sleep(15)
                if child.returncode:raise RuntimeError(f'{name} exited {child.returncode}; see logs/{name}.log')
            finally:
                stop(child)
                if adopted:self.retire_old_supervisor()
                if leased:
                    with self.mutex:self.state['active'].pop(str(slot),None)
                    self.release_gpus(leased)
                with self.mutex:self.state['validating'].pop(name,None);self.persist()

    def parallel(self,jobs,fn):
        pending=queue.Queue();results={}
        for i,job in enumerate(jobs):pending.put((i,job))
        def group(slot):
            try:
                while True:
                    try:i,job=pending.get_nowait()
                    except queue.Empty:return
                    results[i]=fn(job,slot)
            except BaseException:
                self.stopping=True
                raise
        # CPU-only official validation may overlap later GPU simulations. GPU
        # leases still cap actual training/inference to the two4-GPU groups.
        with ThreadPoolExecutor(min(4,max(1,len(jobs)))) as pool:list(pool.map(lambda i:group(i%2),range(min(4,max(1,len(jobs))))))
        return [results[i] for i in range(len(jobs))]

    def score(self,name,checkpoint,execution,diffusion,slot,stage='screen',episodes=None,seed_start=None,all_gpus=False):
        episodes=episodes or (100 if stage=='confirm' else 50)
        seed_start=seed_start if seed_start is not None else (22000 if stage=='confirm' else 20000)
        out=self.root/'evaluation'/name;score=out/'selection_score.json'
        identity=dict(checkpoint_sha256=digest(checkpoint/'training_state.pt'),checkpoint_config_sha256=digest(checkpoint/'config.json'),
            adapter_sha256=self.adapter_sha256,
            execution_frames=execution,diffusion_steps=diffusion,policy_seed=1000,seed_start=seed_start,episodes=episodes,max_steps=3000,
            conditions=[f'{s}-L{l}-{m}' for s in ['G1BallMove','G1BallRetrieve','G1IceBreak'] for l in range(3) for m in ['S','R']])
        if not score.exists():
            for previous in (self.root/'evaluation').glob('*/selection_score.json'):
                try:cached=read(previous)
                except json.JSONDecodeError:continue
                if all(cached.get(k)==v for k,v in identity.items()) and (previous.parent/'summary.json').exists():
                    out.mkdir(exist_ok=True)
                    for filename in ['summary.json','summary.md','evaluation_settings.json']:
                        source=previous.parent/filename
                        if source.exists():shutil.copy2(source,out/filename)
                    atomic(out/'CACHED_FROM.json',dict(source=str(previous.parent),reused_at=now(),reason='identical weight, config, inference and environment-seed identities'))
                    atomic(score,cached)
                    break
        if not score.exists():
            command=[sys.executable,'-u','scripts/eval_toolbench.py','--checkpoint',str(checkpoint),
                '--gpus',','.join(self.groups) if all_gpus else self.groups[slot],'--output',str(out),'--base-port',str(30000+slot*200),
                '--episodes',str(episodes),'--seed-start',str(seed_start),'--execution-frames',str(execution),
                '--diffusion-steps',str(diffusion)]
            for attempt in range(1,4):
                try:
                    self.process(command,'eval_'+name,slot)
                    break
                except RuntimeError as exc:
                    if self.stopping or attempt==3:raise
                    atomic(out/f'retry_{attempt}.json',dict(attempt=attempt,error=str(exc),time=now(),
                        reason='Retry after owned process cleanup; exact validated conditions are reused.'))
                    print(now(),f'retrying {name}, attempt{attempt+1}/3: {exc}',flush=True)
                    time.sleep(2)
        value=read(score)
        assert all(value.get(k)==v for k,v in identity.items()), 'evaluation identity mismatch'
        assert value['checkpoint_sha256']==digest(checkpoint/'training_state.pt')
        assert value['execution_frames']==execution and value['diffusion_steps']==diffusion and value['seed_start']==seed_start and value['episodes']==episodes
        assert value['primary_l12'] is not None and value['auxiliary_all18'] is not None
        row=dict(name=name,checkpoint=str(checkpoint),training_step=int(checkpoint.name.removeprefix('checkpoint_')),execution=execution,diffusion=diffusion,stage=stage,summary=str(out/'summary.json'),**value)
        with self.mutex:
            self.state['trials']=[v for v in self.state['trials'] if v['name']!=name]+[row]
            atomic(self.root/'leaderboard.json',sorted(self.state['trials'],key=self.rank,reverse=True))
            write_stage_reports(self.root,self.state['trials']);self.persist()
        return row

    @staticmethod
    def rank(row):return (row['primary_l12'],row['auxiliary_all18'],-row.get('diffusion',10),row.get('execution',50),-row.get('step',row.get('training_step',0)))

    def prepared(self,execution):
        if execution==50:return REPO/'data/toolbench/prepared'
        out=REPO/f'data/toolbench/history_e{execution}'
        if not (out/'history_audit.json').exists():
            self.process([sys.executable,'-u','scripts/recondition_toolbench.py','--source',str(REPO/'data/toolbench/prepared'),
                '--out',str(out),'--execution-frames',str(execution)],f'prepare_e{execution}',0)
        assert read(out/'history_audit.json')['passed'];return out

    def train(self,spec,step,execution,slot):
        from omegaconf import OmegaConf
        reuse_path=self.root/'training_reuse.json'
        reuse=read(reuse_path).get(spec['name']) if reuse_path.exists() else None
        if reuse:
            expected=OmegaConf.to_container(self.base,resolve=True)
            expected['main'].update(data_root=str(self.prepared_path),sampling=spec['sampling'],seed=spec.get('seed',1000))
            if 'max_steps' in spec:expected['main']['max_steps']=spec['max_steps']
            expected['training']['optimizer'].update(zip(['control_backbone_lr','control_adapter_lr','hand_lr'],spec['rates']))
            expected['training']['scheduler']['decay_steps']=spec.get('decay_steps',200000)
            for key in ('data_root','text_cache','save_root'):
                expected['main'][key]=str((REPO/os.path.expanduser(expected['main'][key])).resolve())
            checkpoint=Path(reuse['checkpoint'])
            actual=read(checkpoint/'config.json')
            assert reusable_training_prefix(actual,expected,step), 'reuse requires identical effective training recipe'
            assert digest(checkpoint/'config.json')==reuse['checkpoint_config_sha256']
            assert digest(checkpoint/'training_state.pt')==reuse['checkpoint_sha256']
            initial=read(checkpoint.parent/'initialization.json')
            assert initial['policy_checkpoint'] is None and initial['policy_419h_loaded'] is False
            import torch
            saved=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=True)
            assert saved['global_step']==step and all(torch.isfinite(t).all() for t in saved['model'].values())
            atomic(self.root/'logs'/f"reuse_{spec['name']}.json",dict(reuse,requested_spec=spec,reused_at=now(),
                config_preserved=True,counts_as_independent_training_seed=False))
            return checkpoint
        name=spec['name'];run=self.root/'training'/name;checkpoint=run/f'checkpoint_{step}'
        if not (checkpoint/'training_state.pt').exists():
            if run.exists():
                if not (run/'resume.pt').exists():raise RuntimeError(f'{run}: interrupted before first checkpoint; inspect before retry')
                command=['--resume',str(run)]
            else:
                config=OmegaConf.to_container(self.base,resolve=True)
                config['main'].update(data_root=str(self.prepared_path),sampling=spec['sampling'],seed=spec.get('seed',1000))
                if 'max_steps' in spec:config['main']['max_steps']=spec['max_steps']
                config['training']['optimizer'].update(zip(['control_backbone_lr','control_adapter_lr','hand_lr'],spec['rates']))
                config['training']['scheduler']['decay_steps']=spec.get('decay_steps',200000)
                assert config['model']==self.invariants['model'] and config['training']['loss']==self.invariants['loss']
                path=self.root/'configs'/f'{name}.yaml';OmegaConf.save(OmegaConf.create(config),path)
                command=['--config',str(path),'--run-dir',str(run)]
            self.process([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4','scripts/train.py',*command,'--stop-after',str(step)],f'train_{name}_{step}',slot,training=run)
        initial=read(run/'initialization.json');assert initial['policy_checkpoint'] is None and initial['policy_419h_loaded'] is False
        config=read(run/'config.json');assert config['model']==self.invariants['model'] and config['training']['loss']==self.invariants['loss']
        assert spec['rates']==[1e-5,3e-5,3e-5], 'learning-rate search is disabled'
        assert config['main']['sampling']==spec['sampling'] and config['main']['seed']==spec.get('seed',1000)
        assert [config['training']['optimizer'][k] for k in ['control_backbone_lr','control_adapter_lr','hand_lr']]==spec['rates']
        assert config['training']['scheduler']['decay_steps']==spec.get('decay_steps',200000)
        assert config['main']['data_root']==str(self.prepared_path)
        if 'max_steps' in spec:assert config['main']['max_steps']==spec['max_steps']
        import torch
        saved=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=True)
        assert saved['global_step']==step and all(torch.isfinite(t).all() for t in saved['model'].values())
        return checkpoint

    def train_score(self,job,slot):
        spec,step,execution,diffusion=job
        cp=self.train(spec,step,execution,slot)
        row=self.score(f"{spec['name']}_s{step}_screen",cp,execution,diffusion,slot)
        return dict(row,spec=spec,step=step)

    def train_scores(self,jobs):
        """Queue training and scoring separately so an odd third trial can train
        while the other GPU group evaluates the first two trained candidates.
        Return input order to keep exact-score ties independent of completion time.
        """
        pending=queue.Queue();done=threading.Event();results={}
        for i,job in enumerate(jobs):pending.put((i,'train',job))
        def worker(slot):
            try:
                while not done.is_set():
                    if self.stopping:raise RuntimeError('search stopping')
                    try:i,kind,payload=pending.get(timeout=1)
                    except queue.Empty:continue
                    if kind=='train':
                        spec,step,execution,diffusion=payload
                        cp=self.train(spec,step,execution,slot)
                        pending.put((i,'score',(payload,cp)))
                    elif kind=='prefetch':
                        spec,step,checkpoint,execution=payload
                        name=f'trained_budget{step}_e{execution}_d10'
                        atomic(self.root/'logs'/f'prefetch_{name}.json',dict(
                            reason='Idle GPU group while the other candidate trains; usable only if this budget wins',
                            checkpoint=str(checkpoint),started=now(),affects_budget_selection=False))
                        self.score(name,checkpoint,execution,10,slot)
                    else:
                        job,cp=payload;spec,step,execution,diffusion=job
                        row=self.score(f"{spec['name']}_s{step}_screen",cp,execution,diffusion,slot)
                        with self.mutex:
                            results[i]=dict(row,spec=spec,step=step)
                            if len(results)==len(jobs):done.set()
                        reuse_path=self.root/'training_reuse.json'
                        if (not done.is_set() and execution==50 and diffusion==10
                                and spec['name'].startswith('budget_fit_') and reuse_path.exists()
                                and spec['name'] in read(reuse_path)
                                and read(self.root/'plan.json').get('prefetch_reused_budget_inference',False)):
                            for e in [30,20,10]:pending.put((i,'prefetch',(spec,step,cp,e)))
            except BaseException:
                self.stopping=True;done.set();raise
        with ThreadPoolExecutor(2) as pool:list(pool.map(worker,[0,1]))
        return [results[i] for i in range(len(jobs))]

    def confirm(self,rows,prefix):
        def fn(row,slot):
            if row['stage']=='confirm':return row
            value=self.score(prefix+'_'+row['name'],Path(row['checkpoint']),row['execution'],row['diffusion'],slot,stage='confirm')
            return dict(value,spec=row.get('spec'),step=row.get('step',row['training_step']))
        return self.parallel(rows,fn)

    @staticmethod
    def near_tie(a,b):
        # Conservative screening rule, not a claim of statistical significance.
        se=math.sqrt(sum(max(r['primary_l12']*(1-r['primary_l12']),1/r['primary_episodes'])/r['primary_episodes'] for r in [a,b]))
        return abs(a['primary_l12']-b['primary_l12'])<=max(.005,1.96*se)+1e-12

    @staticmethod
    def budget_candidates(rows,explored=None):
        ordered=sorted(rows,key=Search.rank,reverse=True);steps=[]
        for row in ordered:
            if row['step'] not in steps:steps.append(row['step'])
            if len(steps)==2:break
        best=steps[0]
        neighbors=sorted((r['step'] for r in (explored or ordered) if abs(r['step']-best)==50000))
        if neighbors and not any(s in steps for s in neighbors):steps.append(neighbors[0])
        return steps

    def budget_pipeline(self,spec,steps,execution,diffusion):
        """One uninterrupted trajectory and a checkpoint-scoring queue.

        Group 0 trains; group 1 scores atomically completed checkpoints while
        training continues. Once training finishes, group 0 drains the score
        queue too. No speculative change to sampler or training schedule.
        """
        pending=queue.Queue();train_done=threading.Event();abort=threading.Event();results={}
        run=self.root/'training'/spec['name']
        def producer():
            try:
                self.train(spec,max(steps),execution,0)
            finally:train_done.set()
        def watch():
            remaining=list(steps)
            while remaining and not abort.is_set():
                for step in list(remaining):
                    cp=run/f'checkpoint_{step}'
                    # config is written after atomic training_state.pt rename.
                    if (cp/'config.json').exists() and (cp/'training_state.pt').exists():
                        pending.put(step);remaining.remove(step)
                if train_done.is_set() and remaining:
                    raise RuntimeError(f'missing budget checkpoints: {remaining}')
                if remaining:time.sleep(2)
        def consume(slot):
            while not abort.is_set():
                if self.stopping:raise RuntimeError('search stopping')
                if train_done.is_set() and len(results)==len(steps):return
                try:step=pending.get(timeout=1)
                except queue.Empty:continue
                row=self.score(f"{spec['name']}_s{step}_screen",run/f'checkpoint_{step}',execution,diffusion,slot)
                with self.mutex:
                    results[step]=dict(row,spec=spec,step=step)
                    self.budget_report(list(results.values()))
        def guarded(fn):
            try:return fn()
            except BaseException:
                abort.set();self.stopping=True;raise
        def train_then_consume():producer();consume(0)
        with ThreadPoolExecutor(3) as pool:
            futures=[pool.submit(guarded,fn) for fn in [train_then_consume,watch,lambda:consume(1)]]
            for future in futures:future.result()
        return [results[s] for s in steps]

    def fit_budget(self,base_spec,steps,execution,diffusion,prefix):
        specs=[dict(base_spec,name=f'{prefix}_seed1000_e{execution}_b{step}',seed=1000,
                    decay_steps=step,max_steps=step) for step in steps]
        rows=self.train_scores([(s,s['max_steps'],execution,diffusion) for s in specs])
        confirmed=self.confirm(rows,prefix+'_confirm')
        self.budget_report(rows+confirmed)
        return confirmed

    def stable_budget_choice(self,rows,prefix):
        ordered=sorted(rows,key=self.rank,reverse=True)
        if len(ordered)<2 or not self.near_tie(*ordered[:2]):return ordered[0]
        self.phase(prefix+'_second_training_seed')
        first=ordered[:2]
        specs=[dict(r['spec'],name=r['spec']['name'].replace('seed1000','seed1001'),seed=1001) for r in first]
        repeated=self.parallel(list(zip(first,specs)),lambda pair,slot:
            dict(self.score(prefix+'_seed1001_'+pair[1]['name'],
                 self.train(pair[1],pair[0]['step'],pair[0]['execution'],slot),
                 pair[0]['execution'],pair[0]['diffusion'],slot,stage='confirm'),spec=pair[1],step=pair[0]['step']))
        comparisons=[]
        for original,replica in zip(first,repeated):
            comparisons.append(dict(candidate=original['spec']['name'],budget=original['step'],
                mean_primary=(original['primary_l12']+replica['primary_l12'])/2,
                mean_auxiliary=(original['auxiliary_all18']+replica['auxiliary_all18'])/2,
                seed1000=original,seed1001=replica))
        comparisons.sort(key=lambda r:(r['mean_primary'],r['mean_auxiliary'],-r['budget']),reverse=True)
        atomic(self.root/(prefix+'_training_seed_comparison.json'),comparisons)
        return comparisons[0]['seed1000']

    def run(self):
        assert read(self.root/'smoke/resume_equivalence.json')['passed']
        assert read(self.root/'smoke/history_reconditioning.json')['passed']
        assert read(self.root/'smoke/eval_e10_d20/summary.json')['failed']==0
        assert read(self.root/'smoke/fullrecipe_validation.json')['passed']
        assert read(self.root/'smoke/cache_identity.json')['passed']
        assert read(self.root/'smoke/queue_validation.json')['passed']
        assert read(self.root/'smoke/eval_identity10/summary.json')['failed']==0
        assert read(self.root/'smoke/process_cleanup.json')['passed']
        assert read(self.root/'smoke/microbatch_profile.json')['passed']
        assert read(self.root/'smoke/microbatch_warmup.json')['passed']
        assert read(self.root/'smoke/metric_aggregation.json')['passed']
        assert read(REPO/'outputs/toolbench_validation/raw_audit.json')['passed']
        assert read(REPO/'outputs/toolbench_validation/prepared_audit.json')['passed']
        execution,diffusion=50,10
        self.prepared_path=self.prepared(execution)
        specs=[dict(name=f'{s}_simple1_seed1000_e{execution}',sampling=s,rates=[1e-5,3e-5,3e-5],seed=1000) for s in ['window_uniform','task_uniform','level_balanced']]
        provisional=getattr(self,'provisional_sampling',None)
        if provisional:
            assert provisional['sampling'] in [s['sampling'] for s in specs]
            self.phase('sampling_provisionally_fixed')
            selected_sampler=provisional['sampling']
            chosen_spec=next(s for s in specs if s['sampling']==selected_sampler)
            sampled=self.train_scores([(chosen_spec,50000,execution,diffusion)])
            sampling_confirm=[r for r in self.state['trials'] if r['stage']=='confirm' and r['name'].startswith('sampling_confirm_')]
        else:
            self.phase('sampling_50k')
            sampled=self.train_scores([(s,50000,execution,diffusion) for s in specs])
            self.phase('sampling_confirmation')
            sampling_confirm=self.confirm(sorted(sampled,key=self.rank,reverse=True)[:2],'sampling_confirm')
            ordered=sorted(sampling_confirm,key=self.rank,reverse=True)
            if self.near_tie(*ordered[:2]):
                self.phase('sampling_extended_confirmation')
                sampling_confirm=self.parallel(ordered,lambda r,slot:dict(self.score(
                    'sampling_extended_'+r['spec']['name'],Path(r['checkpoint']),execution,diffusion,slot,
                    stage='confirm_extended',episodes=200,seed_start=26000),spec=r['spec'],step=50000))
            selected_sampler=max(sampling_confirm,key=self.rank)['spec']['sampling']
        spec=next(s for s in specs if s['sampling']==selected_sampler)
        atomic(self.root/'fixed_training_choices.json',dict(sampling=selected_sampler,
            rates=spec['rates'],learning_rate_search=False,chosen_at=now(),
            sampling_confirmation=sampling_confirm,capacity_loss='unchanged',
            provisional=bool(provisional),sampling_selection=provisional or 'completed sampling comparison'))

        # Coarse budget exploration retains the exact original 200k schedule.
        self.phase('budget_exploration_100k_150k_200k')
        budget=[next(r for r in sampled if r['spec']['sampling']==selected_sampler)]
        budget+=self.budget_pipeline(spec,[100000,150000,200000],execution,diffusion)
        self.phase('budget_extension_confirmation')
        late=self.confirm([r for r in budget if r['step'] in [150000,200000]],'extension_confirm')
        latest=next(r for r in late if r['step']==200000)
        previous=next(r for r in late if r['step']==150000)
        gain=latest['primary_l12']-previous['primary_l12']
        recent=[r['primary_l12'] for r in budget if r['step']>=100000]
        noisy=max(recent)-min(recent)>=.005-1e-12
        # Extend on confirmed growth, or a fluctuating late curve without a
        # meaningful confirmed decline. Flat curves do not extend automatically.
        extend=gain>1e-12 or (noisy and abs(gain)<.005 and self.near_tie(latest,previous))
        atomic(self.root/'budget_extension_decision.json',dict(extend=extend,
            confirmed_gain=gain,late_screen_range=max(recent)-min(recent),
            reason='confirmed growth or fluctuating late curve without meaningful decline',
            confirmations=late))
        if extend:
            self.phase('budget_exploration_250k_300k')
            budget+=self.budget_pipeline(spec,[250000,300000],execution,diffusion)
        self.budget_report(budget+late)
        self.phase('budget_exploration_confirmation')
        top=self.confirm(sorted(budget,key=self.rank,reverse=True)[:3],'coarse_budget_confirm')
        candidates=self.budget_candidates(top,budget)
        atomic(self.root/'budget_retraining_candidates.json',dict(steps=candidates,
            common_decay_steps=200000,source_rows=top,
            rule='top two distinct confirmed-first budgets; add an adjacent explored budget if neither is adjacent to the winner'))

        # Every finalist starts fresh with both max_steps and cosine horizon
        # equal to its candidate budget. Sampling, peak LRs and losses stay fixed.
        self.phase('budget_matched_scratch_retraining')
        matched=self.fit_budget(spec,candidates,execution,diffusion,'budget_fit_'+selected_sampler)
        winner=self.stable_budget_choice(matched,'budget_fit')
        cp=Path(winner['checkpoint']);spec=winner['spec'];best_step=winner['step']
        atomic(self.root/'budget_initial_choice.json',winner)

        self.phase('trained_policy_inference_execution_screen')
        es=self.parallel([50,30,20,10],lambda e,slot:self.score(
            f'trained_budget{best_step}_e{e}_d10',cp,e,10,slot))
        selected_execution=max(es,key=self.rank)['execution']
        self.phase('trained_policy_inference_diffusion_screen')
        ds=self.parallel([20,30],lambda d,slot:self.score(
            f'trained_budget{best_step}_e{selected_execution}_d{d}',cp,selected_execution,d,slot))
        inference=self.confirm([dict(r,spec=spec,step=best_step) for r in sorted(es+ds,key=self.rank,reverse=True)[:2]],'trained_inference_confirm')
        proposed=max(inference,key=self.rank)
        atomic(self.root/'inference_initial_choice.json',dict(proposed,
            history_mismatch='Changed execution remains exploratory until matched-history scratch retraining.'))
        if proposed['execution']!=execution:
            new_execution=proposed['execution'];new_diffusion=proposed['diffusion']
            old_path=self.prepared_path;self.prepared_path=self.prepared(new_execution)
            self.phase('execution_matched_budget_retraining')
            neighbors=[s for s in [best_step-50000,best_step,best_step+50000] if 50000<=s<=300000]
            aligned=self.fit_budget(spec,neighbors,new_execution,new_diffusion,'aligned_'+selected_sampler)
            alternative=self.stable_budget_choice(aligned,'aligned_budget')
            # Compare equally evaluated, history-matched alternatives; the
            # unmatched inference scores never become a locked policy.
            reference=dict(self.score('history50_reference_confirm',cp,execution,new_diffusion,0,
                stage='confirm',all_gpus=True),spec=spec,step=best_step)
            if self.rank(alternative)>self.rank(reference):
                winner=alternative;execution=new_execution;diffusion=new_diffusion
                spec=winner['spec'];best_step=winner['step'];cp=Path(winner['checkpoint'])
            else:
                self.prepared_path=old_path;winner=reference;diffusion=new_diffusion
            atomic(self.root/'matched_history_decision.json',dict(adopted=execution==new_execution,
                alternative=alternative,reference=reference))
        else:
            winner=proposed;diffusion=proposed['diffusion']

        self.phase('final_diffusion_confirmation')
        ds=self.parallel([10,20,30],lambda d,slot:self.score(
            f'final_{spec["name"]}_d{d}',cp,execution,d,slot))
        dc=self.confirm([dict(r,spec=spec,step=best_step) for r in sorted(ds,key=self.rank,reverse=True)[:2]],'diffusion_final_confirm')
        winner=max(dc,key=self.rank);diffusion=winner['diffusion']
        locked=dict(spec=spec,budget=best_step,execution=execution,diffusion=diffusion,
            selection=winner,locked_at=now(),protocol_revision=read(self.root/'plan.json')['registered_revision'],learning_rate_search=False,
            provisional_sampling=provisional)
        from omegaconf import OmegaConf
        selected=OmegaConf.to_container(self.base,resolve=True)
        selected['main'].update(max_steps=best_step,data_root=str(self.prepared_path),sampling=spec['sampling'],seed=1000,
            score_checkpoints=[best_step],resume_snapshots=[best_step])
        selected['training']['optimizer'].update(zip(['control_backbone_lr','control_adapter_lr','hand_lr'],spec['rates']))
        selected['training']['scheduler']['decay_steps']=best_step
        assert spec['decay_steps']==best_step and spec['max_steps']==best_step
        OmegaConf.save(OmegaConf.create(selected),self.root/'selected_training.yaml')
        locked['training_config_sha256']=digest(self.root/'selected_training.yaml')
        atomic(self.root/'locked_configuration.json',locked)
        self.phase('independent_training_seeds')
        replicate=[dict(spec,name=spec['name'].replace('seed1000',f'seed{seed}'),seed=seed) for seed in [1001,1002]]
        # A second seed already trained during a close budget comparison is
        # reused only if it is exactly this final, locked configuration.
        reps=self.parallel(replicate,lambda s,slot:dict(spec=s,checkpoint=str(self.train(s,best_step,execution,slot))))
        final=[dict(spec=spec,checkpoint=str(cp)),*reps]
        self.phase('canonical_test_3_training_seeds')
        finals=self.parallel(final,lambda r,slot:self.score(f"test_{r['spec']['name']}",Path(r['checkpoint']),execution,diffusion,slot,stage='test',episodes=100,seed_start=10000))
        self.phase('fresh_holdout_test')
        holdout=self.score('fresh_holdout_seed1000',cp,execution,diffusion,0,stage='holdout',episodes=100,seed_start=30000,all_gpus=True)
        self.report(finals,holdout,locked)
        self.state.update(phase='complete',finished=now(),locked_configuration=locked,final_results=finals,holdout=holdout)
        self.persist()

    def budget_report(self,rows):
        output=[]
        all_rows={r['name']:r for r in [*self.state['trials'],*rows]}
        metric_cache={}
        for row in all_rows.values():
            run=Path(row['checkpoint']).parent
            if run not in metric_cache:metric_cache[run]=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
            step=row.get('step',row['training_step'])
            metrics=[v for v in metric_cache[run] if v['step']<=step]
            tail=metrics[-100:]
            config=read(run/'config.json')
            output.append(dict(name=row['name'],step=step,stage=row['stage'],primary_l12=row['primary_l12'],auxiliary_all18=row['auxiliary_all18'],
                sampling=config['main']['sampling'],training_seed=config['main']['seed'],
                cosine_horizon=config['training']['scheduler']['decay_steps'],execution=row['execution'],diffusion=row['diffusion'],
                examples_seen=step*128,window_equivalent_epochs=step*128/1247525,
                approximate_train_gpu_hours=sum(v['s_per_step']*10*4/3600 for v in metrics),
                last_1000_step_mean_loss=sum(v['loss'] for v in tail)/len(tail) if tail else None,summary=row['summary']))
        if not output:return
        target=self.root/'budget_comparison.csv';temporary=target.with_suffix('.csv.tmp')
        with temporary.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
        temporary.replace(target)

    def report(self,finals,holdout,locked):
        import statistics
        lines=['# ToolBench tuning results','',f"Selected budget: {locked['budget']} updates; execution={locked['execution']}, diffusion={locked['diffusion']}.",'',
            '| Training seed | L1/L2 mean | All18 mean |','| --- | ---: | ---: |']
        for row in finals:lines.append(f"| {row['name']} | {row['primary_l12']:.2%} | {row['auxiliary_all18']:.2%} |")
        columns=['S-L0','S-L1','S-L2','D-L0','D-L1','D-L2','L1/L2 均值','全18均值']
        historical=read(REPO/'log/toolbench/joint18_scratch40k_20261027_8gpu_supervision/evaluation_18x100/summary.json')
        base_values=[sum(r['success_rate'] for r in historical['conditions'] if f'-L{level}-{mode}' in r['env_id'])/3 for mode in ['S','R'] for level in range(3)]
        historical_row=['Historical scratch40k (1 seed)',*base_values,
            sum(r['success_rate'] for r in historical['conditions'] if '-L1-' in r['env_id'] or '-L2-' in r['env_id'])/12,historical['mean_success_rate']]
        table=[]
        for row in finals:
            conditions=read(row['summary'])['conditions'];values=[]
            for mode in ['S','R']:
                for level in range(3):
                    selected=[r['success_rate'] for r in conditions if f'-L{level}-{mode}' in r['env_id']]
                    assert len(selected)==3
                    values.append(sum(selected)/3)
            values += [row['primary_l12'],row['auxiliary_all18']]
            table.append([row['name'],*values])
        lines+=['','D uses the official R environment IDs; each S/D-level column averages the three scenarios.','','| 策略 | '+' | '.join(columns)+' |','| --- | '+' | '.join(['---:']*8)+' |']
        lines.append('| '+historical_row[0]+' | '+' | '.join(f'{v:.2%}' for v in historical_row[1:])+' |')
        lines += ['| '+r[0]+' | '+' | '.join(f'{v:.2%}' for v in r[1:])+' |' for r in table]
        means=[statistics.mean(r[i] for r in table) for i in range(1,9)]
        deviations=[statistics.stdev(r[i] for r in table) for i in range(1,9)]
        lines.append('| Kimodo full (3 seeds) | '+' | '.join(f'{m:.2%} ± {s:.2%}' for m,s in zip(means,deviations))+' |')
        with (self.root/'table3.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['策略',*columns]);writer.writerow(historical_row);writer.writerows(table);writer.writerow(['Kimodo full (mean)',*means]);writer.writerow(['Training-seed sample SD',*deviations])
        video_rows=[]
        for row in [*finals,holdout]:
            for condition in read(row['summary'])['conditions']:
                receipt=Path(condition['result'])
                for video in receipt.parent.rglob('*.mp4'):video_rows.append(dict(evaluation=row['name'],condition=condition['env_id'],video=str(video)))
        with (self.root/'video_index.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['evaluation','condition','video']);writer.writeheader();writer.writerows(video_rows)
        lines+=['',f"Mean ± sample SD across training seeds: L1/L2 {statistics.mean(r['primary_l12'] for r in finals):.2%} ± {statistics.stdev(r['primary_l12'] for r in finals):.2%}; all18 {statistics.mean(r['auxiliary_all18'] for r in finals):.2%} ± {statistics.stdev(r['auxiliary_all18'] for r in finals):.2%}.",'',f"Fresh environment-seed holdout (training seed1000): L1/L2={holdout['primary_l12']:.2%}, all18={holdout['auxiliary_all18']:.2%}.",'','Best means best among the registered configurations and budgets tested. Canonical test results are not used to revise the locked configuration. Capacity/loss unchanged; no 419h or task-policy checkpoint loaded.']
        lines+=['','The historical40k row reuses its previous canonical test for reference and is excluded from this search selection. Training seeds vary the data stream, diffusion noise and ControlNet initialization; hand_init_seed remains fixed by the original model configuration.',
            '', 'Adapter assumptions remain those documented in reports/toolbench_preflight_20261027.md: estimated root heading, command-filled observations between requests, and two continuous hand-closure values. These experiments measure this fixed adapter and full model.']
        (self.root/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--adopt-handoff',type=Path,help='retain an already-running owned training process; previous supervisor must be SIGSTOPped')
    a=p.parse_args();os.chdir(REPO)
    handoff=read(a.adopt_handoff) if a.adopt_handoff else None
    if handoff:
        command=Path(f"/proc/{handoff['training_pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode()
        assert 'torch.distributed.run' in command and handoff['training_run'] in command
        assert os.getpgid(handoff['training_pid'])==handoff['training_pid']
    old=Path(f"/proc/{handoff['old_supervisor']}") if handoff else None
    migration_from=(handoff['old_supervisor'] if old is not None and old.exists()
                    and old.joinpath('stat').read_text().split(') ')[1].split()[0]!='Z' else None)
    search=Search(a.root,migration_from=migration_from)
    search.adoption=handoff
    def interrupted(signum,frame):search.stopping=True;raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:search.run()
    except BaseException as exc:
        search.state.update(phase='failed',error=f'{type(exc).__name__}: {exc}');search.persist();raise
    finally:
        search.stopping=True
        for child in reversed(search.children):stop(child)
        if search.adoption and not search.adoption.get('retired'):search.retire_old_supervisor()

if __name__=='__main__':main()
