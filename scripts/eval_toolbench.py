"""Run the official 18-condition scorer using an independent server per evaluator.

Defaults are the reportable protocol: 100 test seeds (10000..10099), 3000
control steps. Smaller explicit --episodes/--max-steps are diagnostics only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import fcntl
import json
import os
import queue
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading
import urllib.request
from run_config import inference_fingerprint

REPO = Path(__file__).resolve().parents[1]


def lock_evaluation_output(output):
    """Prevent two evaluators with identical weights from sharing HTTP sessions."""
    lock=(Path(output)/'evaluation.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f'Evaluation already running for {output}; refusing duplicate policy servers')
    return lock


def aggregate_scores(rows):
    """Equal-size condition macros computed from integer counts, so exact ties
    cannot be broken by the order in which floating point rates are summed.
    """
    assert rows and len({r['episodes'] for r in rows})==1 and rows[0]['episodes']>0
    primary=[r for r in rows if '-L1-' in r['env_id'] or '-L2-' in r['env_id']]
    p_success=sum(r['successes'] for r in primary);p_episodes=sum(r['episodes'] for r in primary)
    successes=sum(r['successes'] for r in rows);episodes=sum(r['episodes'] for r in rows)
    return dict(primary_l12=p_success/p_episodes if len(primary)==12 else None,
        auxiliary_all18=successes/episodes if len(rows)==18 else None,
        primary_successes=p_success,primary_episodes=p_episodes,total_successes=successes,total_episodes=episodes)


def matching_receipt(directory, settings):
    """Reuse only complete official validation with the exact evaluation identity."""
    files=sorted(Path(directory).glob('run-*/benchmark_result.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    for path in files:
        try:
            r=json.loads(path.read_text());info=r['policy_info']
            if (r['status']=='validated' and r['episodes']==settings['episodes']
                and r['seeds']==list(range(settings['seed_start'],settings['seed_start']+settings['episodes']))
                and r['configuration']['max_episode_steps']==settings['max_steps']
                and info['checkpoint_sha256']==settings['checkpoint_sha256']
                and info['config_sha256']==settings['policy_config_sha256']
                and info['adapter_sha256']==settings['adapter_sha256']
                and info['inference']['execution_frames']==settings['execution_frames']
                and info['inference']['diffusion_steps']==settings['diffusion_steps']):return path
        except (KeyError,ValueError,OSError):continue
    return None


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def stop_many(processes, grace=5):
    """Signal every owned process group before waiting for any one of them.

    Sequential 20s waits can exceed the supervisor's deadline and strand the
    remaining independent policy servers when the evaluator is force-stopped.
    """
    active=[p for p in processes if p.poll() is None]
    for process in active:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    deadline=time.monotonic()+grace
    while any(p.poll() is None for p in active) and time.monotonic()<deadline:
        time.sleep(.05)
    for process in active:
        if process.poll() is None:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
    for process in active:
        process.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--text-cache', type=Path, default=REPO / 'data/cache/toolbench')
    p.add_argument('--benchmark', type=Path, default=Path('/root/workspace/HumanoidToolBench'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', default='1,2,3,4,5,6,7')
    p.add_argument('--workers-per-gpu', type=int, default=3, help='independent simulator/server pairs; each still serves one evaluator')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--max-steps', type=int, default=3000)
    p.add_argument('--diffusion-steps', type=int, default=10)
    p.add_argument('--execution-frames', type=int, default=50)
    p.add_argument('--seed-start', type=int, default=10000)
    p.add_argument('--policy-seed', type=int, default=1000)
    p.add_argument('--conditions', nargs='+', help='default all 18 conditions')
    p.add_argument('--base-port', type=int, default=27300)
    p.add_argument('--video-workers', type=int, default=2, choices=[1,2,3,4])
    a = p.parse_args()
    a.checkpoint, a.output, a.benchmark = a.checkpoint.resolve(), a.output.resolve(), a.benchmark.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    output_lock=lock_evaluation_output(a.output)
    gpus = [int(s) for s in a.gpus.split(',')]
    assert len(set(gpus)) == len(gpus)
    if a.workers_per_gpu < 1:
        raise ValueError('--workers-per-gpu must be positive')
    conditions = [f'{s}-L{l}-{m}' for s in ['G1BallMove','G1BallRetrieve','G1IceBreak']
                  for l in range(3) for m in ['S','R']]
    if a.conditions:
        if not set(a.conditions)<=set(conditions) or len(set(a.conditions))!=len(a.conditions):
            raise ValueError('invalid or duplicate condition')
        conditions = a.conditions
    sim_python = str(a.benchmark / '.venv/bin/python')
    weight_hash = hashlib.sha256((a.checkpoint / 'training_state.pt').read_bytes()).hexdigest()
    settings = dict(checkpoint_sha256=weight_hash,
        checkpoint_config_sha256=hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest(), execution_frames=a.execution_frames,
        diffusion_steps=a.diffusion_steps, policy_seed=a.policy_seed, seed_start=a.seed_start,
        episodes=a.episodes, max_steps=a.max_steps, conditions=conditions)
    settings['policy_config_sha256']=inference_fingerprint(a.checkpoint/'config.json',dict(dtype='bf16',execution_frames=a.execution_frames,
        diffusion_steps=a.diffusion_steps,rtc_overlap_frames=0,turn='operator',seed=a.policy_seed))
    core={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (REPO/'kimodo_vla').rglob('*.py')}
    settings['adapter_sha256']=hashlib.sha256(json.dumps(core,sort_keys=True).encode()).hexdigest()
    manifest = a.output/'evaluation_settings.json'
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError('output directory belongs to different evaluation settings')
    manifest.write_text(json.dumps(settings,indent=2)+'\n')
    source_hash = hashlib.sha256(b''.join(p.read_bytes() for p in sorted((REPO / 'kimodo_vla').rglob('*.py')))).hexdigest()
    processes, process_lock, stopping = [], threading.Lock(), threading.Event()
    validations=[]
    marker_root=a.output/'simulation_markers';marker_root.mkdir(exist_ok=True)
    simulation_complete=a.output/'simulation_complete.json'
    simulation_complete.unlink(missing_ok=True)
    def performance_status():
        with process_lock:
            current=dict(simulation_complete=simulation_complete.exists(),
                completed_simulations=len(list(marker_root.glob('*.json'))),
                validators_running=sum(p.poll() is None for p in validations),
                validators_finished=sum(p.poll()==0 for p in validations),time=time.time())
            tmp=a.output/'performance_status.json.tmp';tmp.write_text(json.dumps(current,indent=2)+'\n');tmp.replace(a.output/'performance_status.json')

    def own(process):
        with process_lock:
            processes.append(process)
        if stopping.is_set():stop_many([process])
        return process

    def interrupted(signum, frame):
        stopping.set()
        with process_lock:
            active = list(processes)
        stop_many(active)
        raise SystemExit(f'evaluation interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    pending=queue.Queue()
    for condition in conditions:
        existing=matching_receipt(a.output/condition,settings)
        if existing:print(f'{condition}: reuse validated {existing}',flush=True)
        else:pending.put(condition)

    def worker(gpu, slot):
        if stopping.is_set():return
        try:first_task=pending.get_nowait()
        except queue.Empty:return
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=str(gpu),
                   HUMANOIDTOOLBENCH_GPU=str(gpu), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_HUB_DISABLE_PROGRESS_BARS='1', PYTHONPATH=str(REPO)+os.pathsep+str(a.benchmark/'src'))
        env.update(HUMANOIDTOOLBENCH_CPU_THREADS='1',TOOLBENCH_VIDEO_WORKERS=str(a.video_workers),
                   TOOLBENCH_VALIDATION_ROOT=str(a.output.parent/'.validation_slots'))
        port = a.base_port + slot
        with (a.output / f'server_gpu{gpu}_slot{slot}.log').open('a') as log:
            server = own(subprocess.Popen([sys.executable, str(REPO/'scripts/serve_toolbench.py'),
                '--checkpoint', str(a.checkpoint), '--text-cache', str(a.text_cache), '--port', str(port),
                '--dtype', 'bf16', '--seed', str(a.policy_seed), '--diffusion-steps', str(a.diffusion_steps),
                '--execution-frames', str(a.execution_frames)],
                cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            try:
                deadline = time.monotonic()+180
                while True:
                    if stopping.is_set():return
                    if server.poll() is not None:
                        raise RuntimeError(f'GPU{gpu} policy exited; see {log.name}')
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/info', timeout=2) as r:
                            info = json.load(r)
                        if info.get('checkpoint_sha256') != weight_hash:
                            raise RuntimeError(f'Port {port} belongs to another checkpoint')
                        if info.get('config_sha256') != settings['policy_config_sha256']:
                            raise RuntimeError(f'Port {port} belongs to another checkpoint configuration')
                        if info.get('adapter_sha256') != settings['adapter_sha256']:
                            raise RuntimeError(f'Port {port} belongs to another policy source')
                        break
                    except (OSError, TimeoutError):
                        if time.monotonic() > deadline:
                            raise TimeoutError(f'GPU{gpu} server startup timed out')
                        time.sleep(1)
                task=first_task
                while True:
                    if stopping.is_set():return
                    marker=marker_root/f'{task}.json';marker.unlink(missing_ok=True)
                    with (a.output / f'{task}.log').open('a') as eval_log:
                        result = own(subprocess.Popen(['bash',str(a.benchmark/'scripts/run_mujoco.sh'),sim_python,'-u',str(REPO/'scripts/toolbench_eval_worker.py'),task,
                            '--host', '127.0.0.1', '--port', str(port), '--episodes', str(a.episodes),
                            '--max-steps', str(a.max_steps), '--seed-start', str(a.seed_start), '--output', str(a.output)],
                            cwd=a.benchmark, env=dict(env,TOOLBENCH_SIMULATION_MARKER=str(marker)), stdout=eval_log, stderr=subprocess.STDOUT, start_new_session=True))
                        with process_lock:validations.append(result)
                        while result.poll() is None:
                            if stopping.is_set():return
                            if marker.exists():
                                data=json.loads(marker.read_text())
                                assert data['simulation_complete'] and data['pid']==result.pid
                                break
                            time.sleep(1)
                    if result.poll() not in [None,0]:
                        raise RuntimeError(f'{task} scorer failed: {result.returncode}; see {task}.log')
                    performance_status()
                    print(f'GPU{gpu} {task} simulation finished; official validation continues on CPU', flush=True)
                    try:task=pending.get_nowait()
                    except queue.Empty:break
            except BaseException:
                if not stopping.is_set():
                    stopping.set()
                    with process_lock:active=list(processes)
                    stop_many(active)
                raise
            finally:
                stop(server)

    slots = gpus * a.workers_per_gpu
    # Work stealing avoids a fast worker waiting behind a different worker's
    # fixed list. Every condition still has its own exclusive evaluator session.
    with ThreadPoolExecutor(max_workers=min(len(slots), len(conditions))) as pool:
        list(pool.map(lambda pair: worker(*pair), [(g,i) for i,g in enumerate(slots)]))
    # Every server has stopped before this marker releases the GPU lease.
    tmp=simulation_complete.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(dict(time=time.time(),conditions=conditions))+'\n');tmp.replace(simulation_complete)
    while any(p.poll() is None for p in validations):
        if stopping.is_set():raise RuntimeError('evaluation stopping')
        performance_status();time.sleep(2)
    performance_status()
    assert all(p.returncode==0 for p in validations),'official CPU validation failed; see condition logs'
    # Use the official summary implementation and official fingerprints.
    merge = '''import json,sys
from pathlib import Path
from humanoidtoolbench.cli.public_eval import write_summary,run_fingerprint
root=Path(sys.argv[1]);rows=[]
settings=json.loads((root/'evaluation_settings.json').read_text())
for condition in json.loads(sys.argv[2]):
    files=sorted((root/condition).glob('run-*/benchmark_result.json'),key=lambda p:p.stat().st_mtime)
    assert files,condition
    candidates=[(p,json.loads(p.read_text())) for p in files]
    path,r=next((p,r) for p,r in reversed(candidates) if r.get('status')=='validated'
        and r.get('policy_info',{}).get('checkpoint_sha256')==settings['checkpoint_sha256']
        and r.get('policy_info',{}).get('config_sha256')==settings['policy_config_sha256']
        and r.get('policy_info',{}).get('adapter_sha256')==settings['adapter_sha256']
        and r.get('seeds')==list(range(settings['seed_start'],settings['seed_start']+settings['episodes'])))
    assert r['status']=='validated' and r['policy_info']['checkpoint_sha256']==sys.argv[3]
    assert r['policy_info']['config_sha256']==settings['policy_config_sha256']
    assert r['policy_info']['adapter_sha256']==settings['adapter_sha256']
    assert r['seeds']==list(range(settings['seed_start'],settings['seed_start']+settings['episodes']))
    assert r['configuration']['max_episode_steps']==settings['max_steps']
    assert r['policy_info']['inference']['execution_frames']==settings['execution_frames']
    assert r['policy_info']['inference']['diffusion_steps']==settings['diffusion_steps']
    rows.append(dict(env_id=r['env_id'],status='completed',successes=r['successes'],episodes=r['episodes'],
        success_rate=r['success_rate'],reportable=r['reportable'],wall_seconds=r.get('wall_seconds'),result=str(path),fingerprint=run_fingerprint(r)))
print(write_summary(root,rows))
'''
    env = os.environ.copy()
    env['PYTHONPATH'] = str(a.benchmark / 'src')
    subprocess.run([sim_python, '-c', merge, str(a.output), json.dumps(conditions), weight_hash], env=env, check=True)
    assert source_hash == hashlib.sha256(b''.join(p.read_bytes() for p in sorted((REPO/'kimodo_vla').rglob('*.py')))).hexdigest(), 'implementation changed during evaluation'
    summary = json.loads((a.output/'summary.json').read_text())
    assert summary['completed']+summary['skipped'] == len(conditions) and summary['failed'] == 0
    if a.episodes == 100 and a.max_steps == 3000 and a.seed_start==10000:
        assert summary['reportable'], summary['not_reportable_because']
    assert all(r['episodes']==a.episodes for r in summary['conditions'])
    score=dict(**aggregate_scores(summary['conditions']),**settings)
    (a.output/'selection_score.json').write_text(json.dumps(score,indent=2)+'\n')
    print(json.dumps({'mean_success_rate':summary['mean_success_rate'], 'reportable':summary['reportable']}))


if __name__ == '__main__':
    main()
