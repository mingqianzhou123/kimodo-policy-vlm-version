"""Verified seven-task scratch training followed by three matched HOI evaluations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from eval_arena_doubledesk import TASKS
from groot_experiment import atomic_json, check_preflight, fingerprint
from run_config import REPO, load_yaml
from run_groot_experiment import keepalive_receipt, training_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'configs/arena_sonicx7_groot.yaml')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--gpus', default='4,5,6,7')
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--validation-root', type=Path, default=REPO / 'outputs/sonicx7_preflight')
    parser.add_argument('--keepalive', type=Path, default=Path('/root/workspace/xiancun/xiancun'))
    parser.add_argument('--baseline-checkpoint', type=Path,
                        help='Optional original seven-task scratch checkpoint; evaluation ONLY, never initialization')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    config = load_yaml(args.config)
    gpus = [int(value) for value in args.gpus.split(',')]
    if len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError('Distinct nonnegative GPU IDs required')
    if config.main.batch_size != config.main.micro_batch_size * len(gpus):
        raise ValueError('This recipe requires the original global 128 = four GPUs x 32')
    tasks = list(config.evaluation.tasks)
    if tasks != ['football', 'doubledesk', 'pp_box']:
        raise ValueError('Expected the three Table 2 HOI tasks')
    check_preflight(config)
    checks = {}
    for name in ['data_audit', 'distributed']:
        record = json.loads((args.validation_root / f'{name}.json').read_text())
        if record.get('passed') is not True or record.get('fingerprint') != fingerprint(config):
            raise RuntimeError(f'{name} validation must pass for this exact recipe')
        if name == 'distributed' and record.get('gpus') != gpus:
            raise RuntimeError('Distributed validation used different GPUs')
        checks[name] = record
    for task in tasks:
        directory = args.validation_root / f'sim_{task}'
        record = json.loads((directory / 'validation.json').read_text())
        manifest = json.loads((directory / 'evaluation_manifest.json').read_text())
        if not record.get('valid_inference') or record.get('successful_inferences', 0) < 1:
            raise RuntimeError(f'{task} real simulator inference has not passed')
        if manifest.get('task_slug') != task or not manifest.get('smoke_only'):
            raise RuntimeError(f'Unexpected simulator receipt for {task}')
        checks[task] = record
    if args.baseline_checkpoint:
        baseline = json.loads((args.baseline_checkpoint / 'config.json').read_text())
        if not (args.baseline_checkpoint / 'training_state.pt').is_file():
            raise FileNotFoundError(args.baseline_checkpoint)
        if baseline.get('initialization') or baseline.get('model', {}).get('visual_conditioning') == 'groot_n1d7':
            raise ValueError('Baseline must be an original scratch ControlNet checkpoint')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    run = (args.run_dir or REPO / config.main.save_root / f'scratch_ddp4_{stamp}').resolve()
    if run.exists():
        raise FileExistsError(f'Refusing to overwrite {run}')
    status_path = REPO / 'outputs' / f'groot_{run.name}_status.json'
    if args.detach:
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--config', str(args.config.resolve()),
                   '--run-dir', str(run), '--gpus', args.gpus, '--gpu', str(args.gpu),
                   '--validation-root', str(args.validation_root.resolve()), '--keepalive', str(args.keepalive)]
        if args.baseline_checkpoint:
            command += ['--baseline-checkpoint', str(args.baseline_checkpoint.resolve())]
        log = REPO / 'outputs' / f'groot_{run.name}_pipeline.log'
        with log.open('x') as handle:
            process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
        print(json.dumps({'pid': process.pid, 'run_dir': str(run), 'status': str(status_path),
                          'pipeline_log': str(log)}, indent=2))
        return
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus)), 'OMP_NUM_THREADS': '2',
           'TOKENIZERS_PARALLELISM': 'false', 'PYTHONPATH': str(REPO), 'NO_ALBUMENTATIONS_UPDATE': '1',
           'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_PROGRESS_BARS': '1',
           'TORCHINDUCTOR_CACHE_DIR': '/root/workspace/model-cache/torchinductor-sonicx7',
           'KIMODO_TEXT_EMBEDDING_CACHE': str(config.main.text_cache)}
    training = training_command(args.config.resolve(), run, gpus)
    status = {'pid': os.getpid(), 'run_dir': str(run), 'gpus': gpus, 'gpu': args.gpu,
              'source_repo': str(REPO),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
              'baseline_checkpoint_evaluation_only': str(args.baseline_checkpoint) if args.baseline_checkpoint else None,
              'world_size': len(gpus), 'phase': 'training', 'pipeline_script': Path(__file__).name,
              'max_steps': config.main.max_steps,
              'global_batch': config.main.batch_size, 'micro_batch': config.main.micro_batch_size,
              'scratch_no_419h': True, 'tasks': tasks, 'fingerprint': fingerprint(config),
              'started_at': datetime.now(timezone.utc).isoformat(), 'training_command': training,
              'checks': checks, 'keepalive_before': keepalive_receipt(args.keepalive)}
    atomic_json(status_path, status)
    try:
        monitor = [sys.executable, '-u', str(REPO / 'scripts/monitor_groot_training.py'),
                   '--run-dir', str(run), '--status', str(status_path), '--detach']
        with subprocess.Popen(training, cwd=REPO, env=env) as process:
            try:
                # The trainer must create the run first; the monitor creates a
                # subdirectory and would otherwise defeat overwrite protection.
                while not (run / 'config.json').exists() and process.poll() is None:
                    time.sleep(1)
                if process.poll() is not None:
                    raise subprocess.CalledProcessError(process.returncode, training)
                subprocess.run(monitor, cwd=REPO, env=env, check=True)
                result = process.wait()
                if result:
                    raise subprocess.CalledProcessError(result, training)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                raise
        checkpoint = run / f'checkpoint_{config.main.max_steps}'
        if not (checkpoint / 'training_state.pt').is_file():
            status['phase'] = 'stopped_with_checkpoint'
            return
        status['keepalive_after_training'] = keepalive_receipt(args.keepalive)
        status['phase'] = 'evaluation'
        results = {}
        for label, model in [('groot', checkpoint), ('original', args.baseline_checkpoint)]:
            if model is None:
                continue
            results[label] = {}
            for task in tasks:
                status.update(evaluation_model=label, evaluation_task=task)
                atomic_json(status_path, status)
                out = run / f'{label}_{task}_formal'
                command = [sys.executable, '-u', str(REPO / 'scripts/eval_arena_doubledesk.py'),
                           '--checkpoint', str(model), '--task', task, '--out', str(out), '--gpu', str(args.gpu),
                           '--reference-manifest', str(args.validation_root / f'sim_{task}' / 'evaluation_manifest.json')]
                subprocess.run(command, cwd=REPO, env=env, check=True)
                measured = json.loads((out / 'summary.json').read_text())
                sr = measured['overall_success_rate']
                results[label][task] = {'success_rate': sr, 'successes': measured['total_successes'],
                                       'episodes': measured['total_episodes'], 'paper_reference': TASKS[task]['paper_sr'],
                                       'delta_vs_paper': sr - TASKS[task]['paper_sr']}
                summary = {'paper_table': 2, 'paper_row': 'Ours (scratch)', 'results': results,
                           'all_models_use_same_evaluator': True, 'evaluation_complete': False}
                atomic_json(run / 'experiment_summary.json', summary)
        summary['evaluation_complete'] = True
        for label in results:
            summary[f'{label}_hoi_mean'] = sum(r['success_rate'] for r in results[label].values()) / len(tasks)
        if 'original' in results:
            summary['delta_vs_matched_original'] = {t: results['groot'][t]['success_rate'] -
                                                   results['original'][t]['success_rate'] for t in tasks}
        atomic_json(run / 'experiment_summary.json', summary)
        status['phase'] = 'complete'
    except Exception as exc:
        status.update(phase='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        try:
            status['keepalive_after'] = keepalive_receipt(args.keepalive)
        except Exception as exc:
            status.update(phase='keepalive_recovery_failed', keepalive_error=str(exc))
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(status_path, status)


if __name__ == '__main__':
    main()
