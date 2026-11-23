"""Run one reach episode in the existing HumanoidArena + SONIC environment.

The defaults are the backend every experiment since 2026-11-17 has used: bounded handoff,
root support, continuous hand conditions, IK-hold projection, 4 rad/s, 400 steps.
``--settled-goal`` additionally starts from stable double support with aligned actuators,
which is the configuration of the 2026-11-23 result. Earlier variants need explicit flags.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

from paths import AGENTIC as ROOT, CHECKPOINTS, REPO, RUNS, sim_cache

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-id',required=True)
    p.add_argument('--mode',choices=['hold','reach','offset','box'],default='hold')
    p.add_argument('--sim-gpu',type=int,default=1)
    p.add_argument('--policy-gpu',type=int,default=2)
    p.add_argument('--share-policy-sim-gpu',action='store_true',
                   help='Explicitly colocate this run\'s policy and simulation on one otherwise empty GPU')
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--diffusion-steps',type=int,default=50)
    p.add_argument('--constraint-cfg',type=float,default=2.)
    p.add_argument('--text-mode',choices=['none','task'],default='none')
    p.add_argument('--root-anchor',choices=['reference','generated'],default='reference')
    p.add_argument('--reference-source',choices=['kimodo','fixed','replay'],default='kimodo')
    p.add_argument('--settled-goal',type=Path,help='Start from stable double support, keeping archived hand goals in world coordinates; '
        'implies --visual-markers --actuator-audit --dynamics-profile actuator')
    p.add_argument('--replay-fixture',type=Path)
    p.add_argument('--motion-goal',type=Path,help='Goal file with a root ground path and/or one hand target (kimodo_vla/oracle/locomotion.py). '
        'Starts settled on both feet with aligned actuators; implies no standing support, no IK repair')
    p.add_argument('--start-xy',type=float,nargs=2,metavar=('X','Y'),help='With --motion-goal: world position to start from instead of the task reset position')
    p.add_argument('--history-source',choices=['reference','measured','measured-last'],default='reference',
        help='With --motion-goal: what Kimodo sees as the past, the reference it sent or the measured body')
    p.add_argument('--noise-seed',type=int,help='With --motion-goal: seed of Kimodo\'s sampling noise, to repeat one goal with different motions')
    p.add_argument('--actuator-audit',action='store_true')
    p.add_argument('--dynamics-profile',choices=['sensor-load','actuator','actuator-load'])
    p.add_argument('--root-support',choices=['none','raised'],help='Root diagnostic: keep ground or add a reference-height support platform')
    p.add_argument('--variant',choices=['original','handoff','bounded'],default='bounded')
    p.add_argument('--handoff-seconds',type=float,default=.2)
    p.add_argument('--joint-speed',type=float,default=4.)
    p.add_argument('--support-constraints',choices=['none','root','root-feet','root-feet-keyframes'],default='root')
    p.add_argument('--hand-constraints',choices=['keyframes','continuous'],default='continuous')
    p.add_argument('--trajectory-projection',choices=['none','ik','ik-hold'],default='ik-hold')
    p.add_argument('--vlm-model',type=Path)
    p.add_argument('--vlm-localizer',choices=['baseline','refined'],default='baseline')
    p.add_argument('--vlm-python',default=sys.executable)
    p.add_argument('--capture-vision',action='store_true')
    p.add_argument('--visual-markers',action='store_true')
    p.add_argument('--camera',choices=['front_camera','world_camera'],default='front_camera')
    p.add_argument('--marker-reach-rad',type=float,choices=[.10,.15,.20],default=.15)
    p.add_argument('--camera-lateral-m',type=float,choices=[-.06,0.,.06],default=0.)
    p.add_argument('--text-cache',type=Path,help='Default: <KVLA_SIM_CACHE>/text-eval')
    p.add_argument('--allow-dirty',action='store_true',help='Debug only: run with uncommitted code changes')
    p.add_argument('--gpu-min-free-mib',type=int,
        help='Explicitly allow existing GPU users if at least this much memory is free (minimum 16384 MiB)')
    a=p.parse_args(argv)
    if a.settled_goal:
        a.visual_markers=a.actuator_audit=True
        a.dynamics_profile=a.dynamics_profile or 'actuator'
    if a.motion_goal:
        a.support_constraints,a.hand_constraints,a.trajectory_projection='none','keyframes','none'
        a.dynamics_profile=a.dynamics_profile or 'actuator'
    return a


def main():
    a=parse_args()
    CACHE=sim_cache()
    a.text_cache=a.text_cache or CACHE/'text-eval'
    run=RUNS/a.run_id
    replay=a.reference_source=='replay'
    fixed=a.reference_source in ('fixed','replay')
    if a.settled_goal:
        if (fixed or a.vlm_model or a.root_support or a.capture_vision or not a.visual_markers
                or a.camera!='front_camera' or a.mode!='hold' or a.trajectory_projection!='ik-hold' or not a.actuator_audit
                or a.dynamics_profile!='actuator'):
            raise ValueError('Settled comparison requires live IK-hold, aligned actuators and original floor/load')
        if not a.settled_goal.is_file():raise FileNotFoundError(a.settled_goal)
    if a.motion_goal:
        if (fixed or a.settled_goal or a.vlm_model or a.root_support or a.capture_vision or a.visual_markers
                or a.actuator_audit or a.mode!='hold' or a.variant!='bounded' or a.dynamics_profile!='actuator'):
            raise ValueError('Moving goals run live on the bounded handoff with aligned actuators and nothing else')
        if not a.motion_goal.is_file():raise FileNotFoundError(a.motion_goal)
    elif a.start_xy or a.noise_seed is not None:raise ValueError('--start-xy and --noise-seed need --motion-goal')
    if a.actuator_audit and not (replay or a.settled_goal):
        raise ValueError('Actuator audit requires an immutable replay fixture')
    if a.dynamics_profile and not a.motion_goal and not ((replay or a.settled_goal) and a.actuator_audit):
        raise ValueError('Dynamics interventions require audited immutable replay')
    if a.root_support and not (a.actuator_audit and replay and a.dynamics_profile in ('actuator','actuator-load')):
        raise ValueError('Root diagnostics require fixed reference and aligned actuators')
    if replay != (a.replay_fixture is not None):
        raise ValueError('Replay source requires --replay-fixture exclusively')
    if replay:
        fixture=json.loads((a.replay_fixture/'replay.json').read_text())
        if a.steps != fixture['consumed_frames']:
            raise ValueError('Replay must consume exactly the archived frame count')
        if not a.capture_vision or not a.visual_markers or a.marker_reach_rad != fixture['marker_reach_rad'] or a.camera_lateral_m != fixture['camera_lateral_m']:
            raise ValueError('Replay scene must match the archived marker/camera configuration')
    if a.vlm_model and (fixed or a.mode!='hold' or a.trajectory_projection!='ik-hold' or not a.visual_markers):
        raise ValueError('First VLM experiment requires visual markers and the standing IK-hold backend')
    if fixed and a.mode not in ('hold','reach'):
        raise ValueError('Fixed reference supports hold/reach only')
    if run.resolve().parent != RUNS.resolve() or (not fixed and a.sim_gpu==a.policy_gpu and not a.share_policy_sim_gpu):
        raise ValueError('Invalid run ID or GPU assignment')
    commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    dirty=bool(subprocess.check_output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'],text=True).strip())
    if dirty and not a.allow_dirty:
        raise SystemExit('Uncommitted code changes: commit them, or pass --allow-dirty for a debug run')
    uuids={}
    if a.gpu_min_free_mib is not None and a.gpu_min_free_mib<16384:
        raise ValueError('Shared-resource runs require at least 16 GiB free on every selected GPU')
    gpu_preflight={}
    for gpu in ((a.sim_gpu,) if fixed else (a.sim_gpu,a.policy_gpu)):
        result=subprocess.check_output(['nvidia-smi',f'--id={gpu}',
            '--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        uuid,mem,util=map(str.strip,result.split(','))
        if a.gpu_min_free_mib is None:
            if int(mem)>100 or int(util)!=0:raise RuntimeError(f'GPU {gpu} is occupied: {result}')
        else:
            free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}',
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<a.gpu_min_free_mib:raise RuntimeError(f'GPU {gpu} has only {free} MiB free')
        gpu_preflight[gpu]={'used_mib':int(mem),'utilization_percent':int(util)}
        uuids[gpu]=uuid
    RUNS.mkdir(parents=True,exist_ok=True)
    run.mkdir(exist_ok=False)
    arena=CACHE/'sources/HumanoidArena-44a72229f11bab47a145e12db39ed891b677e783/isaaclab_twist2_g1'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
    env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',ORACLE_RUN=str(run))
    if a.dynamics_profile:env['SONIC_DYNAMICS_PROFILE']=a.dynamics_profile
    if a.root_support:env['SONIC_ROOT_SUPPORT']=a.root_support
    if a.settled_goal:
        for key in ('VLM_CAPTURE','SONIC_ROOT_SUPPORT'):
            env.pop(key,None)
        env.update(VLM_MARKER_REACH_RAD=str(a.marker_reach_rad),VLM_CAMERA_LATERAL_M=str(a.camera_lateral_m))
    if a.start_xy:env['KVLA_START_XY']=f'{a.start_xy[0]},{a.start_xy[1]}'
    if a.capture_vision or a.vlm_model:
        env.update(VLM_CAPTURE='1',VLM_CAMERA=a.camera,ENABLE_DEPTH='1')
        if a.visual_markers:
            env.update(VLM_MARKERS='1',VLM_MARKER_REACH_RAD=str(a.marker_reach_rad),
                       VLM_CAMERA_LATERAL_M=str(a.camera_lateral_m))
    policyenv={**env,'CUDA_VISIBLE_DEVICES':'' if fixed else uuids[a.policy_gpu],
        'PYTHONPATH':os.pathsep.join((str(ROOT),str(REPO))),'HF_HUB_OFFLINE':'1'}
    checkpoint=CHECKPOINTS/'Kimodo-G1-RP-v1'
    policycmd=[sys.executable,'-u',str(ROOT/'scripts/serve_oracle.py'),'--checkpoint',str(checkpoint),
        '--text-cache',str(a.text_cache),'--port',str(port),'--run',str(run),'--mode',a.mode,
        '--diffusion-steps',str(a.diffusion_steps),'--constraint-cfg',str(a.constraint_cfg),
        '--text-mode',a.text_mode,'--root-anchor',a.root_anchor,'--variant',a.variant,
        '--handoff-seconds',str(a.handoff_seconds),'--joint-speed',str(a.joint_speed),
        '--support-constraints',a.support_constraints,'--hand-constraints',a.hand_constraints,'--trajectory-projection',a.trajectory_projection]
    if a.settled_goal:policycmd+=['--archived-goal',str(a.settled_goal.resolve())]
    if a.motion_goal:
        policycmd+=['--motion-goal',str(a.motion_goal.resolve()),'--history-source',a.history_source]
        if a.noise_seed is not None:policycmd+=['--noise-seed',str(a.noise_seed)]
    if a.vlm_model:policycmd+=['--vlm-model',str(a.vlm_model.resolve()),'--vlm-python',a.vlm_python,'--vlm-localizer',a.vlm_localizer]
    if fixed:
        policycmd=[sys.executable,'-u',str(ROOT/'scripts/serve_fixed_reference.py'),
                   '--run',str(run),'--mode',a.mode,'--port',str(port)]
    if replay:
        policycmd=[sys.executable,'-u',str(ROOT/'scripts/serve_recorded_reference.py'),
                   '--run',str(run),'--fixture',str(a.replay_fixture.resolve()),'--port',str(port)]
    runtime=run/'runtime'
    simenv={**env,'CUDA_VISIBLE_DEVICES':uuids[a.sim_gpu], 'PYTHONPATH':os.pathsep.join((str(ROOT),str(REPO),str(arena))),
        'KVLA_RENDER_GPU':str(a.sim_gpu),'ACCEPT_EULA':'Y','OMNI_KIT_ACCEPT_EULA':'YES',
        'CYCLONEDDS_HOME':str(CACHE/'cyclonedds-0.10.2')}
    simenv['LD_LIBRARY_PATH']=str(CACHE/'system-libs/usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
    for key,folder in [('XDG_CACHE_HOME','cache'),('XDG_DATA_HOME','data'),('XDG_CONFIG_HOME','config'),('CUDA_CACHE_PATH','cuda')]:
        simenv[key]=str(runtime/folder)
    episode={'seed':1000,'episode_seed':1000,'repeat_idx':0,'episode_index':0,'max_steps':a.steps,
        'result_json':str(run/'episode.json'),'success_video_dir':str(run/'videos'),
        'failure_video_dir':str(run/'videos'),'recording_save_dir':str(run/'recordings'),'video_fps':30,
        'post_termination_record_steps':0,'model_label':('replay-' if replay else 'vlm-markers-' if a.vlm_model else 'fixed-' if fixed else 'oracle-A-')+a.mode,
        'eval_model_path':str(a.replay_fixture.resolve()) if replay else str(REPO/'kimodo_vla/oracle/fixed_reference.py') if fixed else str(checkpoint)}
    (run/'episodes.json').write_text(json.dumps({'episodes':[episode]},indent=2))
    simcmd=[str(CACHE/'envs/isaaclab-2.2/bin/python'),'-u',str(ROOT/'scripts'/('free_sim_bootstrap.py' if a.motion_goal else 'settled_sim_bootstrap.py' if a.settled_goal else 'root_sim_bootstrap.py' if a.root_support else 'dynamics_sim_bootstrap.py' if a.dynamics_profile else 'actuator_sim_bootstrap.py' if a.actuator_audit else 'oracle_sim_bootstrap.py')),
        str(arena/'script/eval_scripts/sonic/sim_eval_vla.py'),'--headless','--device','cuda:0','--kit_args',
        f'--allow-root --portable-root {runtime} --/renderer/activeGpu={a.sim_gpu} --/renderer/multiGpu/enabled=false',
        '--task','Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody','--env_config_yaml',
        str(arena/'tasks/common_test_config/base_test/pp_box_sonic_test.yaml'),
        '--seed','1000','--episode_seed','1000','--max_steps',str(a.steps),'--episode_batch_json',str(run/'episodes.json'),
        '--lerobot_server_url',f'http://127.0.0.1:{port}','--lerobot_server_timeout','660' if a.vlm_model else '120','--step_log_every_n','50',
        '--sonic_encoder_path',str(CACHE/'sim-assets/sonic/model_encoder.onnx'),
        '--sonic_decoder_path',str(CACHE/'sim-assets/sonic/model_decoder.onnx')]
    record={'status':'starting','experiment':'A','oracle':True,'mode':a.mode,'sim_command':simcmd,
        'vlm_localizer':a.vlm_localizer,
        'target_source':'qwen_rgb_depth' if a.vlm_model else 'oracle',
        'marker_reach_rad':a.marker_reach_rad,'camera_lateral_m':a.camera_lateral_m,
        'vlm_model':str(a.vlm_model) if a.vlm_model else None,'vision_camera':a.camera,
        'shared_policy_sim_gpu':bool(not fixed and a.sim_gpu==a.policy_gpu),
        'gpu_min_free_mib':a.gpu_min_free_mib,'gpu_preflight':gpu_preflight,
        'reference_source':a.reference_source,'generator_loaded':not fixed,'actuator_audit':a.actuator_audit,
        'dynamics_profile':a.dynamics_profile,'dynamics_profile_sha256':sha(ROOT/'configs/sonic-dynamics.json') if a.dynamics_profile else None,
        'root_support':a.root_support,'root_support_config_sha256':sha(ROOT/'configs/root-dynamics.json') if a.root_support else None,
        'variant':a.variant,'handoff_seconds':a.handoff_seconds,'joint_speed_rad_s':a.joint_speed,
        'support_constraints':a.support_constraints,'hand_constraints':a.hand_constraints,'trajectory_projection':a.trajectory_projection,
        'policy_command':policycmd,'gpu_uuids':uuids,'started_unix':time.time(),
        'grasp_attempted':False,'success_rate_evaluation':False,
        'git_commit':commit,'git_dirty':dirty,
        # Keys keep the pre-merge layout so runs stay comparable with those before 2026-11-23.
        'source_sha256':{str(p.relative_to(base)):sha(p) for base,directory in ((REPO,'kimodo_vla'),(ROOT,'arena'),(ROOT,'scripts'))
                         for p in (base/directory).rglob('*.py')},
        'checkpoint_sha256':{} if fixed else {str(p.relative_to(checkpoint)):sha(p) for p in checkpoint.rglob('*') if p.is_file()},
        'tracker_sha256':{p.name:sha(p) for p in (CACHE/'sim-assets/sonic').glob('model_*.onnx')}}
    if a.settled_goal:record.update(experiment='SETTLED-VISUAL-GOAL',oracle=False,target_source='archived_visual_world',
        archived_goal_sha256=sha(a.settled_goal),marker_reach_rad=a.marker_reach_rad,camera_lateral_m=a.camera_lateral_m)
    if a.vlm_model:record.update(experiment='VLM-A',oracle=False)
    if a.motion_goal:record.update(experiment='MOTION-GOAL',oracle=False,target_source='motion_goal',
        motion_goal=json.loads(a.motion_goal.read_text()),motion_goal_sha256=sha(a.motion_goal),
        start_xy=a.start_xy,history_source=a.history_source,noise_seed=a.noise_seed)
    record['arena_sha256']={f:sha(arena/f) for f in (
        'action_provider/action_provider_sonic.py','action_provider/vla_robot_current_local_runtime_v3.py',
        'tasks/common_test_config/base_test/pp_box_sonic_test.yaml')}
    if replay:
        for key in ('tracker_sha256','arena_sha256'):
            if record[key] != fixture[key]:
                raise ValueError(f'Replay environment changed: {key}')
        record.update(experiment='SONIC-REPLAY',oracle=False,target_source='archived_evaluation_only',
                      replay_fixture=str(a.replay_fixture.resolve()),replay_manifest_sha256=sha(a.replay_fixture/'replay.json'))
    manifest=run/'run.json'
    manifest.write_text(json.dumps(record,indent=2))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with (run/'server.log').open('x') as plog,(run/'simulation.log').open('x') as slog:
        server=subprocess.Popen(policycmd,env=policyenv,cwd=ROOT,stdout=plog,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+180
            while True:
                if server.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError('Native policy server did not start; inspect server.log')
                try:
                    with opener.open(f'http://127.0.0.1:{port}/health',timeout=2) as response:
                        health=json.load(response)
                        assert health.get('source')==a.reference_source if fixed else health['native']
                    break
                except OSError:
                    time.sleep(1)
            record['status']='running'; manifest.write_text(json.dumps(record,indent=2))
            sim=subprocess.run(simcmd,env=simenv,cwd=arena,stdout=slog,stderr=subprocess.STDOUT,timeout=1800)
            record['sim_exit_code']=sim.returncode
            sim.check_returncode()
            if not (run/'execution.jsonl').is_file() or not (run/'plans.jsonl').is_file():
                raise RuntimeError('No actual execution or native-model evidence')
            if fixed and not (run/'commands.jsonl').is_file():
                raise RuntimeError('Missing prescribed command evidence')
            episode_result=json.loads((run/'episode.json').read_text())
            if episode_result.get('error'):
                raise RuntimeError(episode_result['error'])
            record['status']='completed'
        except BaseException as exc:
            record.update(status='failed',error=repr(exc))
            raise
        finally:
            server.terminate()
            try: server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait()
            record['ended_unix']=time.time();manifest.write_text(json.dumps(record,indent=2))
            shutil.rmtree(runtime,ignore_errors=True)  # simulator caches, ~0.8 GB per run and rebuilt on demand

if __name__=='__main__': main()
