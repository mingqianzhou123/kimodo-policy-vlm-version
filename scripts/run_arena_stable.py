"""Validate continuous/resumed DDP trajectories, then train a fresh 500k policy.

Run under setsid; this supervisor records failure and never silently restarts.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import statistics
import sys
import time
import traceback

import torch
from omegaconf import OmegaConf

from groot_experiment import atomic_json, check_preflight, fingerprint
from run_config import REPO, load_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'configs/arena_sonicx7_groot_stable.yaml')
    parser.add_argument('--name', required=True)
    parser.add_argument('--gpus', default='4,5,6,7')
    args = parser.parse_args()
    config = load_yaml(args.config)
    check_preflight(config)
    audit = json.loads((Path(config.main.preflight_report).parent / 'data_audit.json').read_text())
    assert audit['passed'] and audit['fingerprint'] == fingerprint(config)
    assert not config.get('initialization') and config.main.max_steps == 500000
    assert config.main.expected_tasks and sum(config.main.expected_tasks.values()) == 700
    gpus = [int(x) for x in args.gpus.split(',')]
    assert len(gpus) == 4 and len(set(gpus)) == 4
    run = (REPO / config.main.save_root / args.name).resolve()
    validation = REPO / 'outputs' / args.name
    validation.mkdir(parents=True, exist_ok=False)
    status_path = REPO / 'outputs' / f'groot_{args.name}_status.json'
    status = dict(phase='validation', pid=os.getpid(), run_dir=str(run), gpus=gpus,
                  pipeline_script=Path(__file__).name,
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  scratch_no_419h=True, max_steps=500000, fingerprint=fingerprint(config),
                  started_at=datetime.now(timezone.utc).isoformat(), validation_dir=str(validation))
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpus, 'OMP_NUM_THREADS': '2',
           'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_PROGRESS_BARS': '1',
           'NO_ALBUMENTATIONS_UPDATE': '1', 'PYTHONPATH': str(REPO),
           'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    def receipt():
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(status_path, status)
    def train(arguments, log):
        command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
                   '--nnodes=1', '--nproc_per_node=4', str(REPO / 'scripts/train.py'), *arguments]
        with log.open('a') as output, subprocess.Popen(command, cwd=REPO, env=env,
                                                       stdout=output, stderr=subprocess.STDOUT) as child:
            status.update(child_pid=child.pid, training_command=command)
            receipt()
            if status['phase'] == 'training':
                while not (run / 'config.json').exists() and child.poll() is None:
                    time.sleep(1)
                if child.poll() is None:
                    subprocess.run([sys.executable, str(REPO / 'scripts/monitor_groot_training.py'),
                                    '--run-dir', str(run), '--status', str(status_path), '--detach'],
                                   env=env, cwd=REPO, check=True)
            code = child.wait()
            if code:
                raise RuntimeError(f'Trainer exited {code}; see {log}')
    receipt()
    try:
        common = ['--config', str(args.config.resolve()), '--audit-distributed']
        overrides = ['main.require_preflight=false', 'main.log_every=1',
                     # Validation starts/stops three DDP jobs in one process.
                     # Keep this lifecycle check deterministic; production
                     # retains the configured worker pool for throughput.
                     'main.num_workers=0',
                     # The production AdaLN config compiles the denoiser for
                     # throughput.  Resume equivalence compares every tensor,
                     # so validation must use eager kernels to avoid compiler
                     # cache/RNG and reduction-order differences.
                     'model.compile=false',
                     'training.stages.adapter_start=3', 'training.scheduler.warmup_steps=10',
                     'training.scheduler.adapter_warmup_steps=5']
        # Keep the lifecycle check short enough to run before production while
        # still crossing the staged adapter boundary and exercising resume.
        validation_steps = int(config.main.get('validation_steps', 12))
        split_steps = validation_steps // 2
        train([*common, '--run-dir', str(validation/'continuous'), '--stop-after', str(validation_steps), *overrides],
              validation/'continuous.log')
        train([*common, '--run-dir', str(validation/'split'), '--stop-after', str(split_steps), *overrides],
              validation/'split.log')
        train(['--resume', str(validation/'split'), '--stop-after', str(validation_steps), '--audit-distributed'],
              validation/'resume.log')
        a = torch.load(validation/'continuous/resume.pt', map_location='cpu', weights_only=True, mmap=True)
        b = torch.load(validation/'split/resume.pt', map_location='cpu', weights_only=True, mmap=True)
        assert a['global_step'] == b['global_step'] == validation_steps
        assert len(a['rng_states']) == len(b['rng_states']) == 4
        errors = {}
        for name, value in a['model'].items():
            other = b['model'][name]
            assert torch.isfinite(value).all() and torch.isfinite(other).all()
            errors[name] = float((value - other).abs().max())
            assert torch.allclose(value, other, atol=1e-6, rtol=1e-5), f'Resume drift: {name}'
        for key, state in a['optimizer']['state'].items():
            for field, value in state.items():
                other = b['optimizer']['state'][key][field]
                if torch.is_tensor(value):
                    assert torch.allclose(value, other, atol=1e-7, rtol=1e-5), f'Adam drift: {key}/{field}'
        assert a['scheduler'] == b['scheduler']
        for left, right in zip(a['rng_states'], b['rng_states']):
            assert torch.equal(left['cpu'], right['cpu']) and torch.equal(left['cuda'], right['cuda'])
        rows = [json.loads(x) for x in (validation/'continuous/metrics.jsonl').read_text().splitlines()]
        result = dict(passed=True, steps=validation_steps, resumed_at=split_steps, gpus=gpus,
                      max_parameter_error=max(errors.values()),
                      median_seconds_per_step=sorted(r['s_per_step'] for r in rows[10:])[len(rows[10:])//2],
                      final_loss=rows[-1]['loss'], simulated_adapter_start=3,
                      simulated_global_warmup=10, simulated_adapter_warmup=5,
                      optimizer_scheduler_rng_checked=True, fingerprint=fingerprint(config))
        atomic_json(validation/'resume_equivalence.json', result)
        del a, b
        status.update(phase='training', resume_validation=result)
        receipt()
        train(['--config', str(args.config.resolve()), '--run-dir', str(run)], validation/'production.log')
        if not (run/'checkpoint_500000/training_state.pt').is_file():
            status['phase'] = 'stopped_with_checkpoint'
            return
        status['phase'] = 'evaluation'
        receipt()
        # The existing evaluator supports the three paper Table 2 HOI tasks.
        # Train jointly on seven tasks; report these matched endpoints separately.
        baseline = Path('/root/workspace/kimodo-policy/log/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse/2026-10-18_21-07-12/checkpoint_500000')
        results = {}
        for label, checkpoint in [('groot', run/'checkpoint_500000'), ('original', baseline)]:
            results[label] = {}
            for task in ['football', 'doubledesk', 'pp_box']:
                status.update(evaluation_model=label, evaluation_task=task)
                receipt()
                subprocess.run([sys.executable, str(REPO/'scripts/eval_arena_doubledesk.py'),
                    '--checkpoint', str(checkpoint), '--task', task, '--gpu', str(gpus[-1]),
                    '--out', str(run/f'{label}_{task}_formal'), '--reference-manifest',
                    str(REPO/f'outputs/sonicx7_preflight/sim_{task}/evaluation_manifest.json')],
                    cwd=REPO, check=True)
                measured = json.loads((run/f'{label}_{task}_formal/summary.json').read_text())
                model_result, = measured['per_model'].values()
                seed_rates = list(model_result['per_seed_success_rate'].values())
                n, k = measured['total_episodes'], measured['total_successes']
                rate, z = k/n, 1.96
                center = (rate + z*z/(2*n))/(1+z*z/n)
                radius = z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
                results[label][task] = dict(successes=k, episodes=n, success_rate=rate,
                    per_seed=model_result['per_seed_success_rate'],
                    seed_mean=statistics.mean(seed_rates), seed_sample_sd=statistics.stdev(seed_rates),
                    wilson_95=[center-radius, center+radius])
                atomic_json(run/'experiment_summary.json', dict(results=results, evaluation_complete=False))
        from eval_arena_doubledesk import TASKS
        atomic_json(run/'experiment_summary.json', dict(results=results, evaluation_complete=True,
            delta_vs_matched_original={t: results['groot'][t]['success_rate']-results['original'][t]['success_rate'] for t in TASKS},
            delta_vs_paper_scratch={t: results['groot'][t]['success_rate']-TASKS[t]['paper_sr'] for t in TASKS},
            training_seeds=[int(config.main.seed)], evaluation_seeds=[0,1,2],
            caveat='Evaluation-seed SD is not training-seed SD; optimization differs from the original baseline.'))
        status['phase'] = 'complete'
    except BaseException as exc:
        status.update(phase='failed', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        receipt()


if __name__ == '__main__':
    main()
