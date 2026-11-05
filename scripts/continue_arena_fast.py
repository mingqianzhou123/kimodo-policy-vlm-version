"""Supervise a validated execution-only migration of the seven-task scratch run."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from groot_experiment import atomic_json, check_preflight
from omegaconf import OmegaConf
from run_config import REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--status', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = json.loads((run/'config.json').read_text())
    check_preflight(OmegaConf.create(cfg))
    receipt_path = run/'execution_migration.json'
    migration = json.loads(receipt_path.read_text())
    assert migration['passed'] and migration['optimizer_and_scheduler_preserved']
    gpus = [int(x) for x in args.gpus.split(',')]
    assert cfg['main']['batch_size'] == 128 and cfg['main']['micro_batch_size']*len(gpus) == 128
    assert cfg['main']['max_steps'] == 500000 and not cfg.get('initialization')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpus, 'OMP_NUM_THREADS': '2',
           'PYTHONPATH': str(REPO), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
           'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_PROGRESS_BARS': '1',
           'TOKENIZERS_PARALLELISM': 'false', 'NO_ALBUMENTATIONS_UPDATE': '1',
           'TORCHINDUCTOR_CACHE_DIR': '/root/workspace/model-cache/torchinductor-sonicx7'}
    status = dict(phase='training', pid=os.getpid(), pipeline_script=Path(__file__).name,
                  run_dir=str(run), world_size=len(gpus), gpus=gpus, scratch_no_419h=True,
                  max_steps=500000, parent_run=migration['parent_run'],
                  resumed_from_step=migration['step'], execution_migration=str(receipt_path),
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    def record():
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(args.status, status)
    record()
    try:
        command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
                   '--nnodes=1', f'--nproc_per_node={len(gpus)}', str(REPO/'scripts/train.py'),
                   '--resume', str(run)]
        with (run/'supervised_train.log').open('a') as output, subprocess.Popen(command,
                cwd=REPO, env=env, stdout=output, stderr=subprocess.STDOUT) as child:
            status.update(child_pid=child.pid, training_command=command)
            record()
            # The run already exists; the monitor can safely wait for new metrics.
            subprocess.run([sys.executable, str(REPO/'scripts/monitor_groot_training.py'),
                '--run-dir', str(run), '--status', str(args.status), '--detach'], cwd=REPO, env=env, check=True)
            code = child.wait()
            status['exit_code'] = code
            if code:
                raise RuntimeError(f'Training exited {code}; inspect supervised_train.log')
        final = run/'checkpoint_500000'
        if not (final/'training_state.pt').exists():
            status['phase'] = 'stopped_with_checkpoint'
            return
        status['phase'] = 'evaluation'
        record()
        baseline = Path('/root/workspace/kimodo-policy/log/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse/2026-10-18_21-07-12/checkpoint_500000')
        from eval_arena_doubledesk import TASKS
        results = {}
        for label, checkpoint in [('groot',final), ('original',baseline)]:
            results[label] = {}
            for task in TASKS:
                status.update(evaluation_model=label, evaluation_task=task)
                record()
                out = run/f'{label}_{task}_formal'
                subprocess.run([sys.executable,str(REPO/'scripts/eval_arena_doubledesk.py'),
                    '--checkpoint',str(checkpoint),'--task',task,'--gpu',str(gpus[-1]),
                    '--out',str(out),'--reference-manifest',
                    str(REPO/f'outputs/sonicx7_preflight/sim_{task}/evaluation_manifest.json')], cwd=REPO,check=True)
                measured = json.loads((out/'summary.json').read_text())
                model, = measured['per_model'].values()
                rates = list(model['per_seed_success_rate'].values())
                n,k = measured['total_episodes'], measured['total_successes']
                rate,z = k/n,1.96
                center = (rate+z*z/(2*n))/(1+z*z/n)
                radius = z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
                results[label][task] = dict(successes=k,episodes=n,success_rate=rate,
                    per_seed=model['per_seed_success_rate'],seed_sample_sd=statistics.stdev(rates),
                    wilson_95=[center-radius,center+radius])
                atomic_json(run/'experiment_summary.json',dict(results=results,evaluation_complete=False))
        atomic_json(run/'experiment_summary.json',dict(results=results,evaluation_complete=True,
            delta_vs_matched_original={t:results['groot'][t]['success_rate']-results['original'][t]['success_rate'] for t in TASKS},
            delta_vs_paper_scratch={t:results['groot'][t]['success_rate']-TASKS[t]['paper_sr'] for t in TASKS},
            parent_run=migration['parent_run'],execution_migration=str(receipt_path),training_seeds=[cfg['main']['seed']]))
        status['phase'] = 'complete'
    except BaseException as exc:
        status.update(phase='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        record()


if __name__ == '__main__':
    main()
