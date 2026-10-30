"""Supervise the scratch 40k joint run, diagnostic rollouts, then official 18x100 evaluation.

Only manages its own subprocesses. Status/metrics/GPU telemetry remain on disk;
any failed gate, nonfinite metric, stale training or failed scorer stops the pipeline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from eval_toolbench import REPO, stop


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def validate(root):
    raw, prepared, replay = (read(root / n) for n in ['raw_audit.json','prepared_audit.json','roundtrip18.json'])
    assert raw['passed'] and prepared['passed'] and raw['episodes'] == prepared['episodes'] == 3003
    assert len(raw['conditions']) == len(prepared['conditions']) == 18
    assert raw['motion_max_error'] < 1e-4 and raw['heading_max_error'] < 1e-5
    assert len(replay['runs']) == 54 and all(not r.get('error') and r['fell'] is False and r['steps']==r['demo_steps'] for r in replay['runs'])
    tests = (root/'pytest_full.log').read_text()
    assert '46 passed' in tests and 'failed' not in tests.lower()
    assets = read(root/'assets.json')
    assert assets['verified_files'] == 960
    smoke = REPO/'log/toolbench/smoke_full_100steps'
    assert read(smoke/'initialization.json')['policy_checkpoint'] is None
    assert (smoke/'checkpoint_100/training_state.pt').is_file()
    metrics = [json.loads(s) for s in (smoke/'metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['step'] == 100 and all(math.isfinite(m['loss']) for m in metrics)
    diagnostic = read(root/'eval18_smoke/summary.json')
    assert diagnostic['completed'] == 18 and diagnostic['failed'] == 0 and diagnostic['mean_success_rate'] is not None
    return {'raw_episodes':3003,'prepared_model_frames':prepared['model_frames'], 'windows':prepared['windows'],
            'replay_runs':54,'replay_falls':0,'tests_passed':46,'smoke_training_steps':100,
            'http_smoke_conditions':18,'hand_pose_max_residual_rad':raw['hand_max_error'],
            'manifest_sha256':raw['manifest_sha256']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--gpus', default='4,5,6,7')
    p.add_argument('--diagnostic-gpus', default='1,2,3')
    p.add_argument('--eval-gpus', default='1,2,3,4,5,6,7')
    p.add_argument('--resume', action='store_true', help='continue this exact run from its saved optimizer state')
    a = p.parse_args()
    os.chdir(REPO)
    run = a.run_dir.resolve()
    monitor = run.parent/(run.name+'_supervision')
    monitor.mkdir(parents=True, exist_ok=True)
    lock = (monitor/'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    gates = validate(REPO/'outputs/toolbench_validation')
    status = {'phase':'starting','supervisor_pid':os.getpid(),'started':now(),'heartbeat':now(),
              'run_dir':str(run),'scratch':True,'policy_checkpoint':None,'max_steps':40000,
              'train_gpus':a.gpus,'gates':gates,'latest_step':0,'final_evaluation_episodes_per_condition':100}
    metrics, children, handles = [], [], []

    def persist():
        status['heartbeat'] = now()
        temporary = monitor/'status.tmp'
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(monitor/'status.json')
        points = metrics[-1000:]
        plots = []
        for key in ['loss','motion_loss','hand_loss']:
            values = [(m['step'],m[key]) for m in points if key in m]
            if not values:
                continue
            top = max(v for _,v in values) or 1
            polyline = ' '.join(f'{35+540*s/40000:.1f},{175-140*v/top:.1f}' for s,v in values)
            plots.append(f'<svg viewBox="0 0 600 210"><text x="35" y="20">{key} (max {top:.4f})</text>'
                f'<path d="M35 30V175H580" fill="none" stroke="#aaa"/><polyline points="{polyline}" fill="none" stroke="#147bc4"/>'
                '<text x="35" y="198">0</text><text x="535" y="198">40000</text></svg>')
        (monitor/'progress.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30">'
            '<title>Kimodo VLA ToolBench scratch</title><style>body{font:16px system-ui;max-width:1100px;margin:2rem auto}'
            'svg{width:32%}pre{white-space:pre-wrap}</style><h1>Kimodo VLA · ToolBench · scratch 40k</h1>'
            f'<p>Phase: {html.escape(status["phase"])} · Step: {status["latest_step"]}/40000 · Updated: {status["heartbeat"]}</p>'
            +''.join(plots)+'<pre>'+html.escape(json.dumps(status,indent=2))+'</pre>')

    def spawn(command, logfile, env):
        handle = logfile.open('a'); handles.append(handle)
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        return process

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES=a.gpus, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', NCCL_DEBUG='WARN',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_PROGRESS_BARS='1')
    try:
        command = [sys.executable, '-u','-m','torch.distributed.run','--standalone',
                   f'--nproc_per_node={len(a.gpus.split(","))}', 'scripts/train.py']
        command += ['--resume',str(run)] if a.resume else ['--config','configs/toolbench.yaml','--run-dir',str(run)]
        training = spawn(command, monitor/'training.log', env)
        status.update(phase='training',training_pid=training.pid)
        persist()
        last_progress, last_telemetry, last_diagnostic = time.monotonic(), 0, 0
        diagnostic = None
        while training.poll() is None:
            metric_file = run/'metrics.jsonl'
            if metric_file.exists():
                metrics = [json.loads(s) for s in metric_file.read_text().splitlines() if s.strip()]
                for m in metrics:
                    if any(not math.isfinite(v) for k,v in m.items() if isinstance(v,(float,int))):
                        raise RuntimeError(f'Nonfinite metric: {m}')
                if metrics and metrics[-1]['step'] > status['latest_step']:
                    last_progress=time.monotonic()
                    status.update(latest_step=metrics[-1]['step'], latest_metrics=metrics[-1],
                        eta_training_hours=metrics[-1]['s_per_step']*(40000-metrics[-1]['step'])/3600)
            if time.monotonic()-last_progress > 1200:
                raise TimeoutError('No training progress for 20 minutes')
            if time.monotonic()-last_telemetry > 60:
                gpu = subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True)
                with (monitor/'gpu.jsonl').open('a') as f:
                    f.write(json.dumps({'time':now(),'returncode':gpu.returncode,'gpus':gpu.stdout})+'\n')
                status['gpu_telemetry']=gpu.stdout.strip().splitlines();last_telemetry=time.monotonic()
            if diagnostic is not None and diagnostic.poll() is not None:
                if diagnostic.returncode:
                    raise RuntimeError('Online diagnostic evaluation failed; see diagnostic log')
                status['last_diagnostic_finished']=now();diagnostic=None
            # Full-length, one-seed rollouts provide a behavior check during training;
            # these are explicitly diagnostic and never replace the final 100 seeds.
            milestone=(status['latest_step']//10000)*10000
            checkpoint=run/f'checkpoint_{milestone}'
            if milestone in [10000,20000,30000] and milestone>last_diagnostic and diagnostic is None and (checkpoint/'training_state.pt').exists():
                diagnostic=spawn([sys.executable,'-u','scripts/eval_toolbench.py','--checkpoint',str(checkpoint),
                    '--gpus',a.diagnostic_gpus,'--episodes','1','--output',str(monitor/f'diagnostic_{milestone}'),
                    '--base-port','27400'],monitor/f'diagnostic_{milestone}.log',env)
                last_diagnostic=milestone;status.update(diagnostic_pid=diagnostic.pid,diagnostic_step=milestone)
            persist();time.sleep(15)
        if training.returncode:
            raise RuntimeError(f'Training exited {training.returncode}; inspect training.log')
        # The trainer can finish between polls; include its final metrics in the receipt.
        metrics = [json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines() if s.strip()]
        assert metrics[-1]['step'] == 40000
        assert all(math.isfinite(v) for m in metrics for v in m.values() if isinstance(v, (float, int)))
        status.update(latest_metrics=metrics[-1], eta_training_hours=0)
        import torch
        final=run/'checkpoint_40000'
        checkpoint=torch.load(final/'training_state.pt',map_location='cpu',weights_only=True)
        assert checkpoint['global_step']==40000
        assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
        assert read(run/'initialization.json')['policy_checkpoint'] is None
        assert read(run/'config.json')['main']['max_steps']==40000
        del checkpoint
        status.update(phase='waiting_for_diagnostic',latest_step=40000,training_finished=now(),checkpoint=str(final),
                      checkpoint_sha256=hashlib.sha256((final/'training_state.pt').read_bytes()).hexdigest())
        persist()
        while diagnostic is not None and diagnostic.poll() is None:
            persist();time.sleep(15)
        if diagnostic is not None and diagnostic.returncode:
            raise RuntimeError('Last online diagnostic failed')
        evaluation=spawn([sys.executable,'-u','scripts/eval_toolbench.py','--checkpoint',str(final),
            '--gpus',a.eval_gpus,'--output',str(monitor/'evaluation_18x100')],monitor/'evaluation.log',env)
        status.update(phase='formal_evaluation',evaluation_pid=evaluation.pid);persist()
        while evaluation.poll() is None:
            status['evaluated_conditions']=len(list((monitor/'evaluation_18x100').glob('*/run-*/benchmark_result.json')))
            persist();time.sleep(15)
        if evaluation.returncode:
            raise RuntimeError('Formal evaluation failed; inspect evaluation.log')
        summary=read(monitor/'evaluation_18x100/summary.json')
        assert summary['reportable'] and summary['completed']+summary['skipped']==18
        lines=['# Kimodo VLA / HumanoidToolBench scratch 40k', '',
            'Task policy initialized without a policy checkpoint; frozen Kimodo-G1 motion prior, DINOv3 and cached LLM2Vec retained.', '',
            '18 conditions × 100 test episodes, seeds 10000–10099, 3000 control steps at 50 Hz. Official evaluator and success predicates.', '',
            f'Macro success rate: **{summary["mean_success_rate"]:.2%}**.', '',
            '| Condition | Successes | Success rate | 95% Wilson interval |', '| --- | ---: | ---: | ---: |']
        for row in summary['conditions']:
            n,k=row['episodes'],row['successes'];rate=k/n;z=1.959963984540054
            mid=(rate+z*z/(2*n))/(1+z*z/n)
            half=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
            lines.append(f'| {row["env_id"].split("/")[-1]} | {k}/{n} | {rate:.1%} | {mid-half:.1%}–{mid+half:.1%} |')
        lines += ['', 'Interpretation must follow the measured results. Loss reduction and completed integration alone do not demonstrate effective tool use.', '',
            'Representation limits: one closure scalar per hand cannot preserve both recorded thumb poses (maximum raw joint residual 0.5 rad). '
            'Root heading is inferred from prior commands because the public policy state omits measured pelvis pose. '
            'Demonstration motion replay checks control consistency; the public scenes do not exactly reproduce recorded tool layouts.', '',
            f'Checkpoint SHA256: `{status["checkpoint_sha256"]}`.', '',
            'Full receipts, source/config/asset hashes, decoded-video checks and official reportability verdict are retained under `evaluation_18x100/`.']
        (monitor/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        status.update(phase='complete',finished=now(),evaluated_conditions=18,
            mean_success_rate=summary['mean_success_rate'],results=str(monitor/'evaluation_18x100/summary.md'))
        persist()
    except BaseException as exc:
        status.update(phase='failed',error=f'{type(exc).__name__}: {exc}')
        persist()
        raise
    finally:
        for child in children:
            stop(child)
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
