"""Experiment 2, parts B and C: replay official raw recordings through the tracker in their recorded layout.

The official raw recordings (ModelScope ``Twang2026/HumanoidArena_raw``, one ``.npz`` per demonstration)
store the episode's object seed and every object's initial pose (``episode_init_env_obj_*``), so a
demonstration's layout is known. A replay here:

1. ``prepare`` turns a raw recording into an episode file: the 40-d reference-pose actions, built by
   HumanoidArena's own converter (``build_sonic_actions_from_recording``, heading-aligned as in the
   released LeRobot data), the recorded object seed and initial object poses, and where the objects
   ended up in the recording.
2. ``run`` plays each episode file in Arena's own VLA evaluation (``script/eval_scripts/sonic/sim_eval_vla.py``,
   the task's test configuration, Arena's own success verdict): the robot settles at the task start
   (aligned actuators, as in every run of this line since 2026-11-17), the objects are put where the
   recording had them by Arena's seeded object reset with the recorded seed (``exp2_replay_bootstrap.py``),
   and the actions are sent open loop on the 50 Hz grid after a 1 s blend from the settled pose
   (``EpisodeReference``), hands included. Each episode is played ``--repeats`` times with identical inputs.
3. ``analyze`` collects Arena's verdict per run, the placement check (objects at the first control step
   against the recorded initial poses) and, per episode, whether all repeats agree and succeed (V).

    python exp2_replay.py prepare --task pp_box <raw .npz>...
    python exp2_replay.py run --task pp_box --label partb --repeats 1 --gpus 0 0 1 1 [--episodes ...]
    python exp2_replay.py analyze --task pp_box --label partb
"""
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AGENTIC = HERE.parents[1]
REPO = AGENTIC.parent
sys.path.insert(0, str(AGENTIC/'scripts'))
ARENA_SOURCE = 'sources/HumanoidArena-44a72229f11bab47a145e12db39ed891b677e783/isaaclab_twist2_g1'
TASKS = {
    'pp_box': {'task': 'Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody', 'folder': 'HOI_pp_box', 'track': 'box',
               'config': {'sonic': 'tasks/common_test_config/base_test/pp_box_sonic_test.yaml',
                          'twist2': 'tasks/common_test_config/base_test/pp_box_twist2_test.yaml'}},
    'sit_sofa': {'task': 'Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody', 'folder': 'HSI_sit_sofa', 'track': 'carton_obstacle_01',
                 'config': {'sonic': 'tasks/common_test_config/base_test/sit_sofa_sonic_test.yaml',
                            'twist2': 'tasks/common_test_config/base_test/sit_sofa_twist2_test.yaml'}},
}
BLEND_S, HOLD_S, RATE = 1., 1., 50.
SIM_SEED = 1000   # Arena's own seed: identical in every repeat, so that repeats differ only by the simulator


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def episode_id(raw_path):
    """The recording's timestamp, unique within a task and tracker."""
    return Path(raw_path).stem.rsplit('_', 1)[-1]


# ---- prepare ------------------------------------------------------------------------------------
def prepare(raw_path, tracker='sonic'):
    """An episode file's content from one raw recording (needs HumanoidArena's data tools on sys.path)."""
    from smpl_lerobot_common import build_sonic_actions_from_recording, build_twist2_actions_from_recording
    data = np.load(raw_path, allow_pickle=True)
    frames = int(np.asarray(data['num_frames']).item()) if 'num_frames' in data.files else len(data['robot_root_position'])
    if tracker == 'sonic':
        actions = build_sonic_actions_from_recording(data, num_frames=frames, align_heading_targets=True)
    else:
        actions = build_twist2_actions_from_recording(data, num_frames=frames, control_dt=1/RATE)
    objects, final = {}, {}
    for key in data.files:
        for field in ('position', 'orientation'):
            if key.startswith('episode_init_env_obj_') and key.endswith('_'+field):
                objects.setdefault(key[len('episode_init_env_obj_'):-len(field)-1], {})[field] = np.asarray(data[key]).tolist()
            if key.startswith('env_obj_') and key.endswith('_'+field):
                final.setdefault(key[len('env_obj_'):-len(field)-1], {})[field] = np.asarray(data[key])[-1].tolist()
    return {'source': Path(raw_path).name, 'raw_sha256': sha256(raw_path), 'tracker': tracker,
            'task': str(np.asarray(data['task']).item()), 'episode': episode_id(raw_path),
            'seed': int(np.asarray(data['episode_object_seed']).item()),
            'seed_source': str(np.asarray(data['episode_object_seed_source']).item()),
            'num_frames': frames, 'objects': objects, 'recorded_final_objects': final,
            'robot_start': {'position': np.asarray(data['robot_root_position'])[0].tolist(),
                            'orientation': np.asarray(data['robot_root_orientation'])[0].tolist()},
            'actions': np.asarray(actions, dtype=float).round(6).tolist()}


# ---- the reference served to the simulator ------------------------------------------------------
def episode_timeline(episode, sample):
    """Blend, the whole recording, hold: kimodo_vla's demo timeline with the recording's hand commands kept."""
    from kimodo_vla.oracle.demo_walks import Timeline
    a = np.asarray(episode['actions'], dtype=float)
    timeline = Timeline({'actions': a, 'window': [0, len(a)-1]}, sample, blend_s=BLEND_S, settle_s=HOLD_S)
    n = timeline.segment_start
    timeline.actions[:n, 38:40] = a[0, 38:40]
    timeline.actions[n:n+len(a), 38:40] = a[:, 38:40]
    timeline.actions[n+len(a):, 38:40] = a[-1, 38:40]
    return timeline


class EpisodeReference:
    """Serves one episode's timeline on the measured 50 Hz clock; no model, no feedback."""
    def __init__(self, run, episode_path):
        self.run, self.path = Path(run), Path(episode_path)
        self.episode = json.loads(self.path.read_text())
        self.reset()

    def reset(self, seed=None):
        self.timeline, self.start, self.plan = None, None, 0
        return {'status': 'ok'}

    def __call__(self, request):
        from kimodo_vla.core.types import yaw_matrix, yaw_of
        from kimodo_vla.oracle.geometry import rotation_wxyz
        samples = request.get('observation', {}).get('oracle_history')
        if not samples:
            return self.open_loop()
        sample = samples[-1]
        if self.timeline is None:
            first = samples[0]
            self.timeline, self.start = episode_timeline(self.episode, first), first['time']
            heading = yaw_matrix(float(yaw_of(rotation_wxyz(first['root_quat']))))
            self.target = {'source': 'official-raw', 'episode': self.episode['episode'], 'origin': first['root_pos'],
                           'heading': heading.tolist(), 'start_time': first['time'],
                           'segment_start_time': first['time']+self.timeline.segment_start/RATE,
                           'timeline_steps': self.timeline.length, 'arrival_time': first['time']+self.timeline.length/RATE,
                           'targets_world': [[first['wrist_pos'][0]], [first['wrist_pos'][1]]],
                           'wrist_rotation_world': first['wrist_rot']}
            (self.run/'target.json').write_text(json.dumps(self.target, indent=2))
        index = int(round((sample['time']-self.start)*RATE))
        actions = self.timeline.chunk(index)
        with (self.run/'plans.jsonl').open('a') as f:
            f.write(json.dumps({'plan': self.plan, 'time': sample['time'], 'source': 'official-raw',
                                'timeline_index': index, 'commands': len(actions)})+'\n')
        with (self.run/'commands.jsonl').open('a') as f:
            f.write(''.join(json.dumps({'plan': self.plan, 'timeline_index': index+i, 'hands': r[38:40].tolist()})+'\n'
                            for i, r in enumerate(actions)))
        self.plan += 1
        return {'action_chunk': actions.tolist(), 'oracle_target': self.target}

    def open_loop(self, size=25):
        """Without a measured history (a tracker other than SONIC, no oracle bridge): the recording from its
        first frame, then its last frame held, ``size`` commands per request, each request taking the next ones."""
        a = np.asarray(self.episode['actions'], dtype=float)
        if self.timeline is None:
            hold = np.tile(a[-1:], (int(round(HOLD_S*RATE)), 1))
            hold[:, :2] = 0.
            self.timeline, self.cursor = np.concatenate([a, hold]), 0
            (self.run/'target.json').write_text(json.dumps({'source': 'official-raw-open-loop', 'episode': self.episode['episode'],
                                                             'timeline_steps': len(self.timeline)}, indent=2))
        rows = self.timeline[min(self.cursor, len(self.timeline)-1):self.cursor+size]
        if len(rows) < size:
            pad = np.tile(self.timeline[-1:], (size-len(rows), 1))
            pad[:, :2] = 0.
            rows = np.concatenate([rows, pad])
        with (self.run/'plans.jsonl').open('a') as f:
            f.write(json.dumps({'plan': self.plan, 'source': 'official-raw-open-loop', 'timeline_index': self.cursor,
                                'commands': size})+'\n')
        self.cursor += size
        self.plan += 1
        return {'action_chunk': rows.tolist()}


# ---- simulator side (imported by the bootstraps) -------------------------------------------------
def place_recorded(env, episode_path, run):
    """Put every recorded object at its recorded initial pose; what was done goes to layout-applied.json.

    Arena's seeded object reset with the recording's seed runs first, as Arena's replay does. In the
    HumanoidArena version used here (44a7222) it restores the box exactly but not the shelf: the
    shelf's sampling no longer matches the recordings (recorded shelf y lies outside the configured
    range). So each recorded object is then written to its recorded initial pose with Arena's own
    pose writers: rigid objects through their scene entity, plain prims (the shelf) through the
    prim's world transform. Success is judged by Arena on the live shelf geometry, so it follows.
    """
    import torch
    from common_env_objects import (_read_prim_world_pose, _write_prim_world_pose, apply_deterministic_object_resets_with_seed,
                                    get_recordable_env_object_specs, resolve_env_object_scene_key)
    episode = json.loads(Path(episode_path).read_text())
    applied = apply_deterministic_object_resets_with_seed(env.cfg, env, episode_seed=int(episode['seed']),
                                                          seed_source='recorded')
    specs = {s['record_name']: s for s in get_recordable_env_object_specs(env.cfg)}
    written, readback = {}, {}
    for name, recorded in episode['objects'].items():
        spec = specs.get(name, {})
        position = np.asarray(recorded['position'], dtype=float)
        orientation = np.asarray(recorded['orientation'], dtype=float)
        key = None if spec.get('prim_paths') else resolve_env_object_scene_key(env, env.cfg, name)
        if key is not None:
            entity = env.scene[key]
            state = entity.data.root_state_w.clone()
            state[0, 0:3] = torch.as_tensor(position, dtype=state.dtype, device=state.device)
            state[0, 3:7] = torch.as_tensor(orientation, dtype=state.dtype, device=state.device)
            state[0, 7:] = 0.
            entity.write_root_state_to_sim(state)
            written[name] = f'scene:{key}'
        else:
            for template in spec.get('prim_paths', []):
                path = template.format(env_idx=0)
                if _write_prim_world_pose(path, position, orientation):
                    written[name] = f'prim:{path}'
                    pose = _read_prim_world_pose(path)
                    if pose is not None:
                        readback[name] = {'position': np.asarray(pose[0]).tolist(), 'orientation': np.asarray(pose[1]).tolist()}
                    break
    env.scene.write_data_to_sim()
    missing = sorted(set(episode['objects'])-set(written))
    (Path(run)/'layout-applied.json').write_text(json.dumps(
        {'seed': episode['seed'], 'seeded_reset': applied, 'written': written, 'missing': missing,
         'prim_readback': readback, 'recorded': episode['objects']}, indent=1))
    if missing:
        raise RuntimeError(f'Recorded objects not placed: {missing}')


# ---- launching ----------------------------------------------------------------------------------
def free_mib(gpu):
    out = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    uuid, free = (x.strip() for x in out.split(','))
    return uuid, int(free)


def launch_one(run_id, episode_path, task, gpu, dds_domain, label):
    from paths import CHECKPOINTS, RUNS, sim_cache
    cache = sim_cache()
    arena = cache/ARENA_SOURCE
    run = RUNS/run_id
    episode = json.loads(Path(episode_path).read_text())
    uuid, free = free_mib(gpu)
    if free < 16384:
        raise RuntimeError(f'GPU {gpu} has only {free} MiB free')
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise SystemExit('Uncommitted code changes')
    run.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    steps = int(round((BLEND_S+HOLD_S)*RATE))+episode['num_frames']+10
    env = {k: v for k, v in os.environ.items() if k.lower() not in ('http_proxy', 'https_proxy', 'all_proxy')}
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', ORACLE_RUN=str(run), SONIC_DYNAMICS_PROFILE='actuator',
               KVLA_EXP2_EPISODE=str(Path(episode_path).resolve()), KVLA_TRACK_OBJECT=TASKS[task]['track'],
               KVLA_DDS_DOMAIN=str(dds_domain))
    path = os.pathsep.join((str(HERE), str(AGENTIC), str(REPO)))
    policyenv = {**env, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': path, 'HF_HUB_OFFLINE': '1'}
    policycmd = [sys.executable, '-u', str(HERE/'exp2_replay_server.py'), '--run', str(run),
                 '--episode', str(Path(episode_path).resolve()), '--port', str(port)]
    runtime = run/'runtime'
    simenv = {**env, 'CUDA_VISIBLE_DEVICES': uuid, 'PYTHONPATH': os.pathsep.join((path, str(arena))),
              'KVLA_RENDER_GPU': str(gpu), 'ACCEPT_EULA': 'Y', 'OMNI_KIT_ACCEPT_EULA': 'YES',
              'CYCLONEDDS_HOME': str(cache/'cyclonedds-0.10.2'),
              'LD_LIBRARY_PATH': str(cache/'system-libs/usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH', '')}
    for key, folder in [('XDG_CACHE_HOME', 'cache'), ('XDG_DATA_HOME', 'data'), ('XDG_CONFIG_HOME', 'config'), ('CUDA_CACHE_PATH', 'cuda')]:
        simenv[key] = str(runtime/folder)
    config = TASKS[task]['config'][episode['tracker']]
    batch = {'seed': SIM_SEED, 'episode_seed': SIM_SEED, 'repeat_idx': 0, 'episode_index': 0, 'max_steps': steps,
             'result_json': str(run/'episode.json'), 'success_video_dir': str(run/'videos'),
             'failure_video_dir': str(run/'videos'), 'recording_save_dir': str(run/'recordings'), 'video_fps': 30,
             'post_termination_record_steps': 0, 'model_label': f'official-raw-{episode["episode"]}',
             'eval_model_path': str(Path(episode_path).resolve())}
    (run/'episodes.json').write_text(json.dumps({'episodes': [batch]}, indent=2))
    tracker = episode['tracker']
    bootstrap = 'exp2_replay_bootstrap.py' if tracker == 'sonic' else 'exp2_twist2_bootstrap.py'
    simcmd = [str(cache/'envs/isaaclab-2.2/bin/python'), '-u', str(HERE/bootstrap),
              str(arena/f'script/eval_scripts/{tracker}/sim_eval_vla.py'), '--headless', '--device', 'cuda:0', '--kit_args',
              f'--allow-root --portable-root {runtime} --/renderer/activeGpu={gpu} --/renderer/multiGpu/enabled=false',
              '--task', TASKS[task]['task'], '--env_config_yaml', str(arena/config),
              '--seed', str(SIM_SEED), '--episode_seed', str(SIM_SEED), '--max_steps', str(steps),
              '--episode_batch_json', str(run/'episodes.json'),
              '--lerobot_server_url', f'http://127.0.0.1:{port}', '--lerobot_server_timeout', '120', '--step_log_every_n', '50']
    if tracker == 'sonic':
        simcmd += ['--sonic_encoder_path', str(cache/'sim-assets/sonic/model_encoder.onnx'),
                   '--sonic_decoder_path', str(cache/'sim-assets/sonic/model_decoder.onnx')]
    record = {'status': 'starting', 'experiment': 'P2E2-OFFICIAL-REPLAY', 'label': label, 'task': task,
              'episode_file': str(Path(episode_path).resolve()), 'episode_sha256': sha256(episode_path),
              'raw_source': episode['source'], 'raw_sha256': episode['raw_sha256'], 'tracker': episode['tracker'],
              'sim_seed': SIM_SEED, 'steps': steps, 'gpu': gpu, 'gpu_uuid': uuid, 'dds_domain': dds_domain,
              'dynamics_profile': 'actuator' if tracker == 'sonic' else None, 'settled_start': tracker == 'sonic',
              'sim_command': simcmd, 'policy_command': policycmd,
              'git_commit': commit, 'started_unix': time.time(),
              'tracker_sha256': {p.name: sha256(p) for p in ((cache/'sim-assets/sonic').glob('model_*.onnx') if tracker == 'sonic'
                                 else (arena.parent/'TWIST2/assets/ckpts').glob('*.onnx'))},
              'arena_config_sha256': sha256(arena/config)}
    manifest = run/'run.json'
    manifest.write_text(json.dumps(record, indent=2))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with (run/'server.log').open('x') as plog, (run/'simulation.log').open('x') as slog:
        server = subprocess.Popen(policycmd, env=policyenv, cwd=AGENTIC, stdout=plog, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+120
            while True:
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Replay server did not start; inspect server.log')
                try:
                    with opener.open(f'http://127.0.0.1:{port}/health', timeout=2):
                        break
                except OSError:
                    time.sleep(1)
            record['status'] = 'running'
            manifest.write_text(json.dumps(record, indent=2))
            sim = subprocess.run(simcmd, env=simenv, cwd=arena, stdout=slog, stderr=subprocess.STDOUT, timeout=max(1800, 3*steps))
            record['sim_exit_code'] = sim.returncode
            sim.check_returncode()
            result = json.loads((run/'episode.json').read_text())
            if result.get('error'):
                raise RuntimeError(result['error'])
            record['status'] = 'completed'
        except BaseException as exc:
            record.update(status='failed', error=repr(exc))
            raise
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            record['ended_unix'] = time.time()
            manifest.write_text(json.dumps(record, indent=2))
            shutil.rmtree(runtime, ignore_errors=True)


def run_name(label, task, episode, repeat, tracker='sonic'):
    return f'p2e2-rep-{label}-{task}-{episode}-r{repeat}' if tracker == 'sonic' else f'p2e2-rep-{label}-{tracker}-{task}-{episode}-r{repeat}'


def episode_files(task, tracker='sonic'):
    from paths import RUNS
    return sorted((RUNS/'p2e2-episodes'/tracker/task).glob('*.json'))


def launch(args):
    from paths import RUNS
    files = episode_files(args.task, args.tracker)
    if args.episodes:
        files = [f for f in files if f.stem in args.episodes]
    elif args.first:
        files = files[:args.first]
    jobs = [(f, r) for r in range(args.repeats) for f in files]
    lock = threading.Lock()

    def worker(slot, gpu):
        while True:
            with lock:
                if not jobs:
                    return
                path, repeat = jobs.pop(0)
            run_id = run_name(args.label, args.task, path.stem, repeat, args.tracker)
            manifest = RUNS/run_id/'run.json'
            if manifest.exists() and json.loads(manifest.read_text()).get('status') == 'completed':
                print('SKIP', run_id, flush=True)
                continue
            if (RUNS/run_id).exists():
                print('EXISTS-UNFINISHED', run_id, flush=True)
                continue
            print('START', run_id, 'gpu', gpu, flush=True)
            try:
                launch_one(run_id, path, args.task, gpu, args.dds_domain+slot, args.label)
                print('COMPLETE', run_id, flush=True)
            except BaseException as exc:
                print('FAILED', run_id, repr(exc)[:200], flush=True)
            time.sleep(5)

    threads = [threading.Thread(target=worker, args=(slot, gpu)) for slot, gpu in enumerate(args.gpus)]
    for thread in threads:
        thread.start()
        time.sleep(20)
    for thread in threads:
        thread.join()


# ---- analysis -----------------------------------------------------------------------------------
def placement_error(first_sample, episode, track):
    """Distance (m) and yaw difference (deg) of the tracked object at the first control step from its recorded start."""
    recorded = episode['objects'].get(track)
    if recorded is None or 'box_pos' not in first_sample:
        return None
    from scipy.spatial.transform import Rotation
    w, x, y, z = recorded['orientation']
    a = Rotation.from_quat([x, y, z, w])
    w, x, y, z = first_sample['box_quat']
    b = Rotation.from_quat([x, y, z, w])
    return {'position_m': float(np.linalg.norm(np.asarray(first_sample['box_pos'])-np.asarray(recorded['position']))),
            'angle_deg': float(np.degrees((a.inv()*b).magnitude()))}


def analyze_run(run):
    record = json.loads((run/'run.json').read_text())
    out = {'run': run.name, 'status': record['status'], 'episode': Path(record['episode_file']).stem}
    if record['status'] != 'completed':
        return out
    episode = json.loads(Path(record['episode_file']).read_text())
    result = json.loads((run/'episode.json').read_text())
    out.update(success=bool(result.get('success')), failure_reason=result.get('failure_reason'),
               episode_steps=result.get('episode_steps'), max_reward=result.get('max_reward'))
    first = json.loads((run/'execution.jsonl').open().readline())['before']
    out['placement'] = placement_error(first, episode, TASKS[record['task']]['track'])
    return out


def verdicts(results, repeats):
    """Per episode: successes over the repeats, whether they agree, and whether it enters V."""
    by = {}
    for r in results:
        if 'success' in r:
            by.setdefault(r['episode'], []).append(r['success'])
    table = {}
    for episode, s in sorted(by.items()):
        table[episode] = {'runs': len(s), 'successes': sum(s), 'agree': len(set(s)) == 1,
                          'in_V': len(s) == repeats and all(s)}
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['prepare', 'run', 'analyze'])
    p.add_argument('raw', nargs='*', type=Path)
    p.add_argument('--task', choices=list(TASKS), required=True)
    p.add_argument('--label', default='partb')
    p.add_argument('--tracker', choices=['sonic', 'twist2'], default='sonic')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--gpus', type=int, nargs='+', default=[])
    p.add_argument('--episodes', nargs='+')
    p.add_argument('--first', type=int)
    p.add_argument('--dds-domain', type=int, default=70)
    args = p.parse_args()
    if args.action == 'prepare':
        from paths import RUNS, sim_cache
        sys.path.insert(0, str(sim_cache()/ARENA_SOURCE/'tools/data_tools'))
        out = RUNS/'p2e2-episodes'/args.tracker/args.task
        out.mkdir(parents=True, exist_ok=True)
        for raw in args.raw:
            episode = prepare(raw, args.tracker)
            if TASKS[args.task]['task'].split('-G129')[0] not in episode['task'].replace('Wholedoby', 'Wholebody'):
                raise ValueError(f'{raw} is a {episode["task"]} recording')
            (out/f"{episode['episode']}.json").write_text(json.dumps(episode))
            print(episode['episode'], episode['num_frames'], {k: np.round(v['position'], 3).tolist() for k, v in episode['objects'].items()})
        return
    if args.action == 'run':
        if not args.gpus:
            raise SystemExit('--gpus is required')
        return launch(args)
    from paths import RUNS
    pattern = run_name(args.label, args.task, '*', '*', args.tracker)
    results = [analyze_run(r) for r in sorted(RUNS.glob(pattern)) if (r/'run.json').exists()
               and json.loads((r/'run.json').read_text()).get('tracker') == args.tracker]
    table = verdicts(results, args.repeats)
    done = [r for r in results if 'success' in r]
    summary = {'runs': len(results), 'completed': len(done), 'successes': sum(r['success'] for r in done),
               'episodes': len(table), 'agree': sum(v['agree'] for v in table.values()),
               'V': sorted(e for e, v in table.items() if v['in_V'])}
    out = RUNS/f"{run_name(args.label, args.task, 'all', 'all', args.tracker)}.json"
    out.write_text(json.dumps({'summary': summary, 'episodes': table, 'runs': results}, indent=1))
    for r in results:
        print(json.dumps(r))
    print(json.dumps({k: v for k, v in summary.items() if k != 'V'}), 'V size', len(summary['V']))
    print(out)


if __name__ == '__main__':
    main()
