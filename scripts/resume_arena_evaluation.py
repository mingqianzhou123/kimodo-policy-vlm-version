"""Restore missing Arena evaluation dependencies and score a completed run.

The model and training configuration are never changed. Recovery uses the
existing additive archive restorer, which preserves existing workspace files.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

from eval_arena_doubledesk import TASKS
from groot_experiment import atomic_json
from run_config import REPO
from run_groot_experiment import keepalive_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    final = run / 'checkpoint_500000'
    if not (final / 'training_state.pt').is_file():
        raise FileNotFoundError(final)
    status_path = REPO / 'outputs' / f'groot_{run.name}_status.json'
    if args.detach:
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--run', str(run), '--gpu', str(args.gpu)]
        if args.restore:
            command += ['--restore']
        log = run / 'evaluation_recovery.log'
        with log.open('a') as output:
            child = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': child.pid, 'status': str(status_path), 'log': str(log)}))
        return
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    atomic_json(run / f'evaluation_status_before_recovery_{stamp}.json', previous)
    status = {**previous, 'pid': os.getpid(), 'pipeline_script': Path(__file__).name,
              'phase': 'restoring_evaluation', 'training_complete': True, 'run_dir': str(run),
              'evaluation_gpu': args.gpu, 'error': None, 'traceback': None}
    def record():
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(status_path, status)
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_PROGRESS_BARS': '1',
           'NO_ALBUMENTATIONS_UPDATE': '1', 'TOKENIZERS_PARALLELISM': 'false',
           'OMP_NUM_THREADS': '2', 'PYTHONPATH': str(REPO),
           'KIMODO_TEXT_EMBEDDING_CACHE': '/root/workspace/kimodo-policy/data/cache/HumanoidArena'}
    keepalive = Path('/root/workspace/xiancun/xiancun')
    record()
    try:
        status['keepalive_before_evaluation'] = keepalive_receipt(keepalive)
        subprocess.run([sys.executable, str(REPO / 'scripts/monitor_groot_training.py'),
                        '--run-dir', str(run), '--status', str(status_path), '--detach'],
                       cwd=REPO, env=env, check=True)
        sim = Path('/root/workspace/miniforge3/envs/unitree_sim_env/bin/python')
        arena = Path('/root/workspace/HumanoidArena')
        components = []
        if not sim.is_file():
            components.append('sim')
        if not all((arena / 'isaaclab_twist2_g1/assets' / name).is_dir() for name in ['objects', 'robots']):
            components.append('assets')
        status['restoring_components'] = components
        record()
        if components:
            if not args.restore:
                raise RuntimeError(f'Missing evaluation dependencies: {components}; use --restore')
            subprocess.run(['/usr/bin/python3', '/root/workspace/restore_backup_additive.py', *components],
                           check=True, env=os.environ.copy())
        if not sim.is_file():
            raise FileNotFoundError(sim)
        sim_env = {**env, 'OMNI_KIT_ACCEPT_EULA': 'YES',
                   'LD_LIBRARY_PATH': str(sim.resolve().parents[1] / 'lib') + os.pathsep + env.get('LD_LIBRARY_PATH', '')}
        subprocess.run([str(sim), '-c',
                        "import sys, torch, isaacsim; print(sys.version, torch.__version__, 'Isaac Sim import OK')"],
                       check=True, env=sim_env)
        # Smoke the restored simulator before consuming the formal seed set.
        status['phase'] = 'evaluation_smoke'
        record()
        smoke = run / f'restored_sim_smoke_{stamp}'
        reference = REPO / 'outputs/sonicx7_preflight/sim_football/evaluation_manifest.json'
        subprocess.run([sys.executable, str(REPO / 'scripts/eval_arena_doubledesk.py'),
                        '--checkpoint', str(final), '--task', 'football', '--out', str(smoke),
                        '--gpu', str(args.gpu), '--port', '19873', '--smoke',
                        '--reference-manifest', str(reference)], cwd=REPO, env=env, check=True)
        baseline = Path('/root/workspace/kimodo-policy/log/HumanoidArena_Multi_Task/'
                        'humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse/'
                        '2026-10-18_21-07-12/checkpoint_500000')
        results = {}
        for label, checkpoint in [('groot', final), ('original', baseline)]:
            results[label] = {}
            for task in TASKS:
                status.update(phase='evaluation', evaluation_model=label, evaluation_task=task)
                record()
                out = run / f'{label}_{task}_formal'
                valid = out / 'validation.json'
                if not valid.exists() or not json.loads(valid.read_text()).get('valid_inference'):
                    # Preserve any incomplete previous attempt and its diagnostics.
                    if out.exists():
                        out.rename(run / f'{out.name}_incomplete_{stamp}')
                    subprocess.run([sys.executable, str(REPO / 'scripts/eval_arena_doubledesk.py'),
                        '--checkpoint', str(checkpoint), '--task', task, '--out', str(out),
                        '--gpu', str(args.gpu), '--port', '19873', '--reference-manifest',
                        str(REPO / f'outputs/sonicx7_preflight/sim_{task}/evaluation_manifest.json')],
                        cwd=REPO, env=env, check=True)
                measured = json.loads((out / 'summary.json').read_text())
                model, = measured['per_model'].values()
                n, k = measured['total_episodes'], measured['total_successes']
                if n != 60:
                    raise ValueError(f'{label}/{task}: expected 60 formal episodes, got {n}')
                rate, z = k/n, 1.96
                center = (rate + z*z/(2*n)) / (1 + z*z/n)
                radius = z*math.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / (1 + z*z/n)
                results[label][task] = {'episodes': n, 'successes': k, 'success_rate': rate,
                    'per_seed': model['per_seed_success_rate'], 'wilson_95': [center-radius, center+radius],
                    'seed_sample_sd': statistics.stdev(model['per_seed_success_rate'].values())}
                atomic_json(run / 'experiment_summary.json', {'results': results, 'evaluation_complete': False})
        atomic_json(run / 'experiment_summary.json', {'results': results, 'evaluation_complete': True,
            'delta_vs_matched_original': {t: results['groot'][t]['success_rate'] - results['original'][t]['success_rate'] for t in TASKS},
            'delta_vs_paper_scratch': {t: results['groot'][t]['success_rate'] - TASKS[t]['paper_sr'] for t in TASKS},
            'training_config': str(run / 'config.json'), 'training_seeds': [42]})
        status['phase'] = 'complete'
    except BaseException as exc:
        status.update(phase='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        try:
            status['keepalive_after_evaluation'] = keepalive_receipt(keepalive)
        except Exception as exc:
            status['keepalive_error'] = repr(exc)
        record()


if __name__ == '__main__':
    main()
