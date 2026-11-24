"""Capture the first-person RGB-D image set for the pointing test (experiment 3).

One simulator start per scene. Model inputs go to <out>/inputs, evaluator-only truth to <out>/truth,
and <out>/manifest.json lists every frame's coverage class, measured visibility and file hashes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from paths import AGENTIC as ROOT, REPO, sim_cache

ARENA = 'sources/HumanoidArena-44a72229f11bab47a145e12db39ed891b677e783/isaaclab_twist2_g1'
START_TIMEOUT_S, ATTEMPTS = 300, 6


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def capture_scene(name, scene, spec_path, out, gpu, probe, cache):
    arena, runtime = cache/ARENA, out/f'runtime-{name}'
    uuid, used, load = map(str.strip, subprocess.check_output(['nvidia-smi', f'--id={gpu}',
        '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).split(','))
    if int(used) > 100 or int(load) != 0:
        raise RuntimeError(f'GPU {gpu} is occupied')
    env = {k: v for k, v in os.environ.items() if k.lower() not in ('http_proxy', 'https_proxy', 'all_proxy')}
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', ENABLE_DEPTH='1', CUDA_VISIBLE_DEVICES=uuid,
               PYTHONPATH=os.pathsep.join((str(ROOT), str(REPO), str(arena))), KVLA_RENDER_GPU=str(gpu),
               ACCEPT_EULA='Y', OMNI_KIT_ACCEPT_EULA='YES', CYCLONEDDS_HOME=str(cache/'cyclonedds-0.10.2'),
               LD_LIBRARY_PATH=str(cache/'system-libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH', ''),
               POINTING_SPEC=str(spec_path), POINTING_SCENE=name, POINTING_OUT=str(out), POINTING_PROBE='1' if probe else '0')
    for key, folder in [('XDG_CACHE_HOME', 'cache'), ('XDG_DATA_HOME', 'data'), ('XDG_CONFIG_HOME', 'config'), ('CUDA_CACHE_PATH', 'cuda')]:
        env[key] = str(runtime/folder)
    episodes = out/f'episodes-{name}.json'
    episodes.write_text(json.dumps({'episodes': [{'seed': 1000, 'episode_seed': 1000, 'repeat_idx': 0, 'episode_index': 0,
        'max_steps': 1, 'result_json': str(out/f'episode-{name}.json'), 'success_video_dir': str(runtime/'videos'),
        'failure_video_dir': str(runtime/'videos'), 'recording_save_dir': str(runtime/'recordings'), 'video_fps': 30,
        'post_termination_record_steps': 0, 'model_label': 'pointing-capture', 'eval_model_path': str(spec_path)}]}))
    command = [str(cache/'envs/isaaclab-2.2/bin/python'), '-u', str(ROOT/'scripts/pointing_sim_bootstrap.py'),
        str(arena/'script/eval_scripts/sonic/sim_eval_vla.py'), '--headless', '--device', 'cuda:0', '--kit_args',
        f'--allow-root --portable-root {runtime} --/renderer/activeGpu={gpu} --/renderer/multiGpu/enabled=false',
        '--task', scene['task'], '--env_config_yaml', str(arena/scene['env_config']),
        '--seed', '1000', '--episode_seed', '1000', '--max_steps', '1', '--episode_batch_json', str(episodes),
        # Capture stops before any control; the server address and tracker are required arguments only.
        '--lerobot_server_url', 'http://127.0.0.1:9', '--step_log_every_n', '50',
        '--sonic_encoder_path', str(cache/'sim-assets/sonic/model_encoder.onnx'),
        '--sonic_decoder_path', str(cache/'sim-assets/sonic/model_decoder.onnx')]
    done, started = out/f'captured-{name}', out/f'started-{name}'
    try:
        # Simulator start-up sometimes deadlocks while loading materials (seen on DoubleDesk), before
        # any capture code runs. Nothing has been written at that point, so a restart is clean.
        for attempt in range(1, ATTEMPTS+1):
            with (out/f'simulation-{name}.log').open('w') as log:
                sim = subprocess.Popen(command, env=env, cwd=arena, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                begun = time.monotonic()
                while sim.poll() is None:
                    waited = time.monotonic()-begun
                    if (waited > START_TIMEOUT_S and not started.exists()) or waited > 1800:
                        os.killpg(sim.pid, signal.SIGKILL)
                    time.sleep(2)
            shutil.rmtree(runtime, ignore_errors=True)
            if done.is_file():
                break
            if started.exists():
                raise RuntimeError(f'Capture of {name} failed after it began; inspect simulation-{name}.log')
            print(f'{name}: simulator did not start within {START_TIMEOUT_S} s (attempt {attempt})', flush=True)
        else:
            raise RuntimeError(f'Simulator for {name} did not start in {ATTEMPTS} attempts')
    finally:
        shutil.rmtree(runtime, ignore_errors=True)
        for path in (done, started, episodes):
            path.unlink(missing_ok=True)
    return {'arena_env_config_sha256': sha(arena/scene['env_config']), 'simulator_starts': attempt}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, default=ROOT/'configs/pointing-frames.json')
    p.add_argument('--out', type=Path, required=True, help='New dataset directory; use persistent storage')
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--scenes', nargs='+')
    p.add_argument('--probe', action='store_true', help='Also dump furniture extents, asset sizes and an overview image')
    p.add_argument('--allow-dirty', action='store_true', help='Debug only: capture with uncommitted code changes')
    a = p.parse_args()
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'], text=True).strip())
    if dirty and not a.allow_dirty:
        raise SystemExit('Uncommitted code changes: commit them, or pass --allow-dirty for a debug capture')
    spec = json.loads(a.spec.read_text())
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    scenes = {}
    for name in a.scenes or spec['scenes']:
        print('START', name, flush=True)
        scenes[name] = capture_scene(name, spec['scenes'][name], a.spec.resolve(), out, a.gpu, a.probe, sim_cache())
        print('COMPLETE', name, flush=True)
    from kimodo_vla.vlm.pointing_eval import build_manifest
    manifest = build_manifest(out, {'git_commit': commit, 'git_dirty': dirty, 'spec_sha256': sha(a.spec),
        'captured_unix': time.time(), 'scenes': scenes})
    print(json.dumps(manifest['coverage'], indent=1), flush=True)


if __name__ == '__main__':
    main()
