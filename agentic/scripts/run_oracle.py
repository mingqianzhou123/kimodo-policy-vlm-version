"""Run one reach episode in the existing HumanoidArena + SONIC environment.

The defaults are the backend every experiment since 2026-11-17 has used: bounded handoff,
root support, continuous hand conditions, IK-hold projection, 4 rad/s, 400 steps.
``--settled-goal`` additionally starts from stable double support with aligned actuators,
which is the configuration of the 2026-11-23 result. Earlier variants need explicit flags.
``--command-loop`` runs a whole task (P&PBox or SitSofa) one body command at a time from a scripted
or VLM commander (kimodo_vla/agent; experiment 3 of the second pilot batch).
``--grasp-loop`` runs blocking walk-reach-close-lift rounds on the box from the same settled
start, with a physical success criterion; it defaults to the task's 1450-step time limit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from paths import AGENTIC as ROOT, CHECKPOINTS, REPO, RUNS, sim_cache

TASKS={'pp_box':('Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody','tasks/common_test_config/base_test/pp_box_sonic_test.yaml'),
       'doubledesk':('Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody','tasks/common_test_config/base_test/doubledesk_sonic_test.yaml'),
       'sit_sofa':('Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody','tasks/common_test_config/base_test/sit_sofa_sonic_test.yaml')}


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
    p.add_argument('--steps',type=int,help='Default: 400, 1450 with --grasp-loop, 3000 with --command-loop')
    p.add_argument('--command-loop',type=Path,help='Command loop configuration (JSON); runs --task pp_box or sit_sofa '
        'one body command at a time (kimodo_vla/agent)')
    p.add_argument('--commander',help='With --command-loop: the commander named in the configuration')
    p.add_argument('--condition',default='',help='With --command-loop: condition name, recorded in run.json')
    p.add_argument('--seed',type=int,default=1000,help='Episode seed: object placement and generator noise')
    p.add_argument('--grasp-loop',action='store_true',help='Blocking grasp rounds on the box: latch its pose, wait, '
        'walk, reach, close both hands on measured arrival, lift; retry until success, fall or the step limit')
    p.add_argument('--latch-delay-steps',type=int,default=75,help='Control steps each round waits after latching the box pose')
    p.add_argument('--start-at-stance',action='store_true',help='Diagnostic for --grasp-loop: start the robot at the grasp '
        'stance instead of the task position, so an attempt needs no walking')
    p.add_argument('--stand-still',action='store_true',help='Grasp loop: the robot starts at the stance for the placed box '
        'and never walks; every round reaches from where it stands. Needs --box-at or --box-in-region')
    p.add_argument('--box-at',type=float,nargs=4,metavar=('DEPTH_M','LATERAL_M','YAW_DEG','EXTRA_M'),
        help='Workspace map cell: box centre this far behind the table edge and to the robot\'s left, turned, '
             'with the robot this much further back than the stance rule')
    p.add_argument('--box-in-region',action='store_true',help='Box at a seeded place inside the region fixed in '
        'configs/grasp-loop.json, robot at the configured stance distance')
    p.add_argument('--box-wide',action='store_true',help='Grasp loop: box at a seeded place over the wide range of '
        'configs/grasp-loop.json (wide); the robot starts at the task position and walks every round')
    p.add_argument('--native-walk',action='store_true',help='Grasp loop: walk the official Kimodo way '
        '(kimodo_vla/oracle/grasp_walk.py, settings in configs/grasp-loop.json wide) instead of whole clips')
    p.add_argument('--box-speed',type=float,default=0.,help='With --box-in-region: drive the box back and forth '
        'across the region at this speed (m/s) from the first latch until a hand touches it')
    p.add_argument('--task',choices=['pp_box','doubledesk','sit_sofa'],default='pp_box',help='Grasp loop task: the two-handed box '
        '(pp_box) or the one-handed hammer on DoubleDesk (experiment 1b; needs --stand-still and a --hammer-* placement); '
        'command loop: pp_box or sit_sofa')
    p.add_argument('--hammer-at',type=float,nargs=4,metavar=('DEPTH_M','LATERAL_M','YAW_DEG','EXTRA_M'),
        help='DoubleDesk reach-map cell: grip point this far behind the table edge and to the robot\'s left, handle '
             'turned from the robot\'s left (head towards the robot positive), robot this much further back than the stance rule')
    p.add_argument('--hammer-in-region',action='store_true',help='DoubleDesk: hammer at a seeded place inside the region '
        'fixed in configs/hammer-grasp.json')
    p.add_argument('--inject',type=float,nargs=3,metavar=('ALONG_M','ACROSS_M','TURN_DEG'),help='DoubleDesk: one step after '
        'the first latch move the hammer this much along / across its handle and turn it about the grip point')
    p.add_argument('--hammer-speed',type=float,default=0.,help='DoubleDesk: drive the hammer along the table edge at this '
        'speed (m/s) from the first latch until something touches it')
    p.add_argument('--hammer-direction',type=int,choices=[-1,1],help='With --hammer-speed: +1 to the robot\'s left, -1 to '
        'its right; default from the region seed')
    p.add_argument('--start-timeout-s',type=float,help='Kill the simulator if it has not begun settling after this many '
        'seconds (a start-up hang); off by default')
    p.add_argument('--sim-cache-seed',type=Path,help='Copy this folder (a finished run\'s runtime/cache) into the new '
        'run\'s simulator cache before start, so that DoubleDesk textures are not rebuilt; off by default')
    p.add_argument('--diffusion-steps',type=int,default=50)
    p.add_argument('--constraint-cfg',type=float,default=2.)
    p.add_argument('--text-mode',choices=['none','task'],default='none')
    p.add_argument('--root-anchor',choices=['reference','generated'],default='reference')
    p.add_argument('--reference-source',choices=['kimodo','fixed','replay','demo'],default='kimodo',
        help='demo: replay a walking segment that did not come from Kimodo (kimodo_vla/oracle/demo_walks.py); needs --demo-segment and --start-xy')
    p.add_argument('--demo-segment',type=Path,help='With --reference-source demo: the segment file to replay')
    p.add_argument('--settled-goal',type=Path,help='Start from stable double support, keeping archived hand goals in world coordinates; '
        'implies --visual-markers --actuator-audit --dynamics-profile actuator')
    p.add_argument('--replay-fixture',type=Path)
    p.add_argument('--motion-goal',type=Path,help='Goal file with a root ground path and/or one hand target (kimodo_vla/oracle/locomotion.py). '
        'Starts settled on both feet with aligned actuators; implies no standing support, no IK repair')
    p.add_argument('--start-xy',type=float,nargs=2,metavar=('X','Y'),help='With --motion-goal: world position to start from instead of the task reset position')
    p.add_argument('--walk-steps',type=int,default=75,help='With --motion-goal: control steps executed from each 1.63 s walking plan before planning again')
    p.add_argument('--arm-ik',action='store_true',help='With --motion-goal: solve the reaching arm for the wrist over the last 25 cm of its path; legs and trunk stay as generated')
    p.add_argument('--text',help='With --motion-goal: phrase key in --text-cache; Kimodo then uses separated text/constraint guidance')
    p.add_argument('--text-cfg',type=float,default=2.,help='With --text: the text guidance weight')
    p.add_argument('--root-constraints',choices=['dense','sparse','none'],default='dense',help='With --motion-goal: walking root path on every frame, every 0.5 s, or not at all')
    p.add_argument('--history-frames',type=int,default=100,help='With --motion-goal: consumed-reference frames conditioning each plan')
    p.add_argument('--generator',choices=['policy','segment','native'],default='policy',help="With --motion-goal: 'segment' samples whole walking segments the official Kimodo way (kimodo_vla/model/segment_sampler.py) instead of the policy canvas; 'native' makes every plan an official multi-prompt continuation of the sent reference (--transition-frames)")
    p.add_argument('--transition-frames',type=int,default=5,help="With --generator native: frames of the sent reference each plan continues from")
    p.add_argument('--approach-gain',type=float,default=1.,help='With --motion-goal: walk and stop paths aim at here + gain x (goal - here); the tracker covers only part of each step')
    p.add_argument('--text-until-m',type=float,default=0.,help='With --text auto: drop the walking phrase once the goal is this close')
    p.add_argument('--open-loop',action='store_true',help="With --generator native: plan from the reference's own position and heading, never the measured ones")
    p.add_argument('--segment-seconds',type=float,default=4.)
    p.add_argument('--segment-steps',type=int,default=100)
    p.add_argument('--walk-clips',type=Path,help='With --motion-goal: walk on looped recorded gait cycles from this library (kimodo_vla/oracle/clip_legs.py) instead of Kimodo')
    p.add_argument('--lead-segment',type=Path,help='With --motion-goal: recorded walking segment played before the walk (kimodo_vla/oracle/demo_walks.py)')
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
    a.steps=a.steps or (1450 if a.grasp_loop else 3000 if a.command_loop else 400)
    if a.command_loop:
        if not a.commander or a.task not in ('pp_box','sit_sofa'):
            raise ValueError('--command-loop needs --commander and --task pp_box or sit_sofa')
        if (a.grasp_loop or a.motion_goal or a.settled_goal or a.reference_source!='kimodo' or a.root_support
                or a.capture_vision or a.visual_markers or a.actuator_audit or a.mode!='hold' or a.box_at or a.box_in_region
                or a.box_wide or a.stand_still or a.start_at_stance):
            raise ValueError('The command loop runs on its own: no grasp loop, goal file, replay, markers or placement flags')
        if not a.command_loop.is_file():raise FileNotFoundError(a.command_loop)
        a.dynamics_profile=a.dynamics_profile or 'actuator'   # the aligned actuators of every walking and grasping run
    elif a.task=='sit_sofa' or a.commander or a.condition:
        raise ValueError('--task sit_sofa, --commander and --condition belong to --command-loop')
    hammer=a.task=='doubledesk'
    placements=[bool(a.box_at),a.box_in_region] if not hammer else [bool(a.hammer_at),a.hammer_in_region]
    if (a.hammer_at or a.hammer_in_region or a.inject or a.hammer_speed or a.hammer_direction) and not hammer:
        raise ValueError('--hammer-*, --inject and --hammer-speed belong to --task doubledesk')
    if hammer and (not a.grasp_loop or not a.stand_still or a.box_at or a.box_in_region or a.box_speed):
        raise ValueError('--task doubledesk runs the grasp loop standing still with a --hammer-* placement')
    if hammer and (a.hammer_speed<0 or (a.hammer_speed and a.inject) or (a.hammer_direction and not a.hammer_speed)
                   or (a.hammer_speed and not a.hammer_direction and not a.hammer_in_region)):
        raise ValueError('--hammer-speed is non-negative, excludes --inject, and needs a direction outside the region')
    if any(placements) != a.stand_still or all(placements):
        raise ValueError('--stand-still goes with exactly one placement: --box-at or --box-in-region '
                         '(--hammer-at or --hammer-in-region with --task doubledesk)')
    if a.stand_still and (not a.grasp_loop or a.start_at_stance):
        raise ValueError('--stand-still belongs to the grasp loop and replaces --start-at-stance')
    if a.box_speed and not (a.box_in_region or a.box_wide) or a.box_speed<0:
        raise ValueError('--box-speed needs --box-in-region or --box-wide and a non-negative speed')
    if (a.box_wide or a.native_walk) and (not a.grasp_loop or a.stand_still or a.start_at_stance):
        raise ValueError('--box-wide and --native-walk belong to the grasp loop from the task start')
    if a.grasp_loop:
        a.dynamics_profile=a.dynamics_profile or 'actuator'
    if a.settled_goal:
        a.visual_markers=a.actuator_audit=True
        a.dynamics_profile=a.dynamics_profile or 'actuator'
    if a.motion_goal:
        a.support_constraints,a.hand_constraints,a.trajectory_projection='none','keyframes','none'
        a.dynamics_profile=a.dynamics_profile or 'actuator'
    if a.reference_source=='demo':
        a.dynamics_profile=a.dynamics_profile or 'actuator'  # the same aligned actuators as experiment two
    return a


def main():
    a=parse_args()
    hammer=a.task=='doubledesk'
    grasp_config=ROOT/'configs'/('hammer-grasp.json' if hammer else 'grasp-loop.json')
    CACHE=sim_cache()
    a.text_cache=a.text_cache or CACHE/'text-eval'
    run=RUNS/a.run_id
    replay=a.reference_source=='replay'
    demo=a.reference_source=='demo'
    fixed=a.reference_source in ('fixed','replay','demo')
    if demo != (a.demo_segment is not None):
        raise ValueError('The demo source needs --demo-segment, and --demo-segment needs the demo source')
    if demo and (not a.start_xy or a.motion_goal or a.grasp_loop or a.settled_goal or a.vlm_model or a.root_support or a.capture_vision
                 or a.visual_markers or a.actuator_audit or a.dynamics_profile!='actuator' or a.mode!='hold'):
        raise ValueError('Demo replays start on the open floor (--start-xy) with aligned actuators and nothing else')
    if demo and not a.demo_segment.is_file():raise FileNotFoundError(a.demo_segment)
    if a.settled_goal:
        if (fixed or a.vlm_model or a.root_support or a.capture_vision or not a.visual_markers
                or a.camera!='front_camera' or a.mode!='hold' or a.trajectory_projection!='ik-hold' or not a.actuator_audit
                or a.dynamics_profile!='actuator'):
            raise ValueError('Settled comparison requires live IK-hold, aligned actuators and original floor/load')
        if not a.settled_goal.is_file():raise FileNotFoundError(a.settled_goal)
    if a.motion_goal:
        if (fixed or a.settled_goal or a.grasp_loop or a.vlm_model or a.root_support or a.capture_vision or a.visual_markers
                or a.actuator_audit or a.mode!='hold' or a.variant!='bounded' or a.dynamics_profile!='actuator'):
            raise ValueError('Moving goals run live on the bounded handoff with aligned actuators and nothing else')
        if not a.motion_goal.is_file():raise FileNotFoundError(a.motion_goal)
    elif (a.start_xy and not demo) or a.noise_seed is not None or a.arm_ik:raise ValueError('--start-xy, --noise-seed and --arm-ik need --motion-goal')
    if not a.motion_goal and (a.text or a.lead_segment or a.walk_clips or a.generator!='policy' or a.root_constraints!='dense' or a.history_frames!=100):
        raise ValueError('--text, --root-constraints, --history-frames and --lead-segment need --motion-goal')
    if a.grasp_loop and (fixed or a.settled_goal or a.vlm_model or a.root_support or a.capture_vision or a.visual_markers
            or a.actuator_audit or a.mode!='hold' or a.trajectory_projection!='ik-hold' or a.dynamics_profile!='actuator'
            or a.latch_delay_steps<0):
        raise ValueError('The grasp loop runs on the live IK-hold backend with aligned actuators and nothing else')
    if a.actuator_audit and not (replay or a.settled_goal):
        raise ValueError('Actuator audit requires an immutable replay fixture')
    if a.dynamics_profile and not a.motion_goal and not a.grasp_loop and not a.command_loop and not demo and not ((replay or a.settled_goal) and a.actuator_audit):
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
    if fixed and a.mode not in ('hold','reach'):  # a demo replay is logged as 'hold'
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
    if a.grasp_loop:env['GRASP_LOOP']='1'
    if a.command_loop:env.update(COMMAND_LOOP='1',COMMAND_TASK=a.task)
    if hammer:env['GRASP_TASK']='hammer'
    if a.start_at_stance:
        if not a.grasp_loop:raise ValueError('--start-at-stance belongs to the grasp loop')
        env['GRASP_START_AT_STANCE']='1'
    layout=None
    if a.box_at:
        d,l,y,e=a.box_at
        layout={'cell':{'depth':d,'lateral':l,'yaw':y,'extra':e}}
    elif a.box_in_region:
        layout={'region_seed':a.seed,'speed_mps':a.box_speed}
    elif a.box_wide:
        layout={'wide_seed':a.seed,'speed_mps':a.box_speed}
    if a.hammer_at:
        d,l,y,e=a.hammer_at
        layout={'cell':{'depth':d,'lateral':l,'yaw':y,'extra':e}}
    elif a.hammer_in_region:
        layout={'region_seed':a.seed}
    if hammer and a.inject:
        layout['inject']=dict(zip(('along_m','across_m','turn_deg'),a.inject))
    if hammer and a.hammer_speed:
        layout['motion']={'speed_mps':a.hammer_speed,'direction':a.hammer_direction}
    if layout:env['GRASP_LAYOUT']=json.dumps(layout)
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
        policycmd+=['--motion-goal',str(a.motion_goal.resolve()),'--walk-steps',str(a.walk_steps)]
        if a.arm_ik:policycmd.append('--arm-ik')
        if a.noise_seed is not None:policycmd+=['--noise-seed',str(a.noise_seed)]
        policycmd+=['--root-constraints',a.root_constraints,'--history-frames',str(a.history_frames)]
        if a.text:policycmd+=['--text',a.text,'--text-cfg',str(a.text_cfg)]
        if a.lead_segment:policycmd+=['--lead-segment',str(a.lead_segment.resolve())]
        if a.walk_clips:policycmd+=['--walk-clips',str(a.walk_clips.resolve())]
        if a.generator!='policy':policycmd+=['--generator',a.generator,'--segment-seconds',str(a.segment_seconds),'--segment-steps',str(a.segment_steps)]
        if a.generator=='native':policycmd+=['--transition-frames',str(a.transition_frames)]
        if a.approach_gain!=1.:policycmd+=['--approach-gain',str(a.approach_gain)]
        if a.text_until_m:policycmd+=['--text-until-m',str(a.text_until_m)]
        if a.open_loop:policycmd+=['--open-loop']
    if a.grasp_loop:
        policycmd+=['--grasp-loop',str(grasp_config),'--latch-delay-steps',str(a.latch_delay_steps)]
        if hammer:policycmd+=['--grasp-task','hammer']
        if a.stand_still:policycmd.append('--stand-still')
        if a.native_walk:policycmd.append('--native-walk')
    if a.command_loop:
        policycmd+=['--command-loop',str(a.command_loop.resolve()),'--commander',a.commander,'--command-task',a.task,
                    '--condition',a.condition]
    if a.vlm_model:policycmd+=['--vlm-model',str(a.vlm_model.resolve()),'--vlm-python',a.vlm_python,'--vlm-localizer',a.vlm_localizer]
    if fixed:
        policycmd=[sys.executable,'-u',str(ROOT/'scripts/serve_fixed_reference.py'),
                   '--run',str(run),'--mode',a.mode,'--port',str(port)]
    if demo:
        policycmd+=['--demo-segment',str(a.demo_segment.resolve())]
    if replay:
        policycmd=[sys.executable,'-u',str(ROOT/'scripts/serve_recorded_reference.py'),
                   '--run',str(run),'--fixture',str(a.replay_fixture.resolve()),'--port',str(port)]
    runtime=run/'runtime'
    simenv={**env,'CUDA_VISIBLE_DEVICES':uuids[a.sim_gpu], 'PYTHONPATH':os.pathsep.join((str(ROOT),str(REPO),str(arena))),
        'KVLA_RENDER_GPU':str(a.sim_gpu),'ACCEPT_EULA':'Y','OMNI_KIT_ACCEPT_EULA':'YES',
        'CYCLONEDDS_HOME':str(CACHE/'cyclonedds-0.10.2')}
    simenv['LD_LIBRARY_PATH']=str(CACHE/'system-libs/usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
    if a.sim_cache_seed:
        shutil.copytree(a.sim_cache_seed,runtime/'cache',ignore=shutil.ignore_patterns('*.lock'))
    for key,folder in [('XDG_CACHE_HOME','cache'),('XDG_DATA_HOME','data'),('XDG_CONFIG_HOME','config'),('CUDA_CACHE_PATH','cuda')]:
        simenv[key]=str(runtime/folder)
    episode={'seed':a.seed,'episode_seed':a.seed,'repeat_idx':0,'episode_index':0,'max_steps':a.steps,
        'result_json':str(run/'episode.json'),'success_video_dir':str(run/'videos'),
        'failure_video_dir':str(run/'videos'),'recording_save_dir':str(run/'recordings'),'video_fps':30,
        'post_termination_record_steps':0,'model_label':('command-loop-' if a.command_loop else 'demo-' if demo else 'grasp-loop-' if a.grasp_loop else 'replay-' if replay else 'vlm-markers-' if a.vlm_model else 'fixed-' if fixed else 'oracle-A-')+a.mode,
        'eval_model_path':str(a.replay_fixture.resolve()) if replay else str(a.demo_segment.resolve()) if demo else str(REPO/'kimodo_vla/oracle/fixed_reference.py') if fixed else str(checkpoint)}
    (run/'episodes.json').write_text(json.dumps({'episodes':[episode]},indent=2))
    simcmd=[str(CACHE/'envs/isaaclab-2.2/bin/python'),'-u',str(ROOT/'scripts'/('command_sim_bootstrap.py' if a.command_loop else 'free_sim_bootstrap.py' if a.motion_goal or demo else ('hammer_sim_bootstrap.py' if hammer else 'grasp_sim_bootstrap.py') if a.grasp_loop else 'settled_sim_bootstrap.py' if a.settled_goal else 'root_sim_bootstrap.py' if a.root_support else 'dynamics_sim_bootstrap.py' if a.dynamics_profile else 'actuator_sim_bootstrap.py' if a.actuator_audit else 'oracle_sim_bootstrap.py')),
        str(arena/'script/eval_scripts/sonic/sim_eval_vla.py'),'--headless','--device','cuda:0','--kit_args',
        f'--allow-root --portable-root {runtime} --/renderer/activeGpu={a.sim_gpu} --/renderer/multiGpu/enabled=false',
        '--task',TASKS[a.task][0],'--env_config_yaml',str(arena/TASKS[a.task][1]),
        '--seed',str(a.seed),'--episode_seed',str(a.seed),'--max_steps',str(a.steps),'--episode_batch_json',str(run/'episodes.json'),
        '--lerobot_server_url',f'http://127.0.0.1:{port}','--lerobot_server_timeout','660' if a.vlm_model or a.command_loop else '120','--step_log_every_n','50',
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
        'grasp_attempted':a.grasp_loop,'success_rate_evaluation':a.grasp_loop,'seed':a.seed,
        'latch_delay_steps':a.latch_delay_steps if a.grasp_loop else None,'start_at_stance':a.start_at_stance,
        'stand_still':a.stand_still,'layout_request':layout,'native_walk':a.native_walk,
        'git_commit':commit,'git_dirty':dirty,
        # Keys keep the pre-merge layout so runs stay comparable with those before 2026-11-23.
        'source_sha256':{str(p.relative_to(base)):sha(p) for base,directory in ((REPO,'kimodo_vla'),(ROOT,'arena'),(ROOT,'scripts'),(ROOT,'pilot_experiments'))
                         for p in (base/directory).rglob('*.py')},
        'checkpoint_sha256':{} if fixed else {str(p.relative_to(checkpoint)):sha(p) for p in checkpoint.rglob('*') if p.is_file()},
        'tracker_sha256':{p.name:sha(p) for p in (CACHE/'sim-assets/sonic').glob('model_*.onnx')}}
    if a.settled_goal:record.update(experiment='SETTLED-VISUAL-GOAL',oracle=False,target_source='archived_visual_world',
        archived_goal_sha256=sha(a.settled_goal),marker_reach_rad=a.marker_reach_rad,camera_lateral_m=a.camera_lateral_m)
    if a.vlm_model:record.update(experiment='VLM-A',oracle=False)
    if a.motion_goal:record.update(experiment='MOTION-GOAL',oracle=False,target_source='motion_goal',
        motion_goal=json.loads(a.motion_goal.read_text()),motion_goal_sha256=sha(a.motion_goal),
        start_xy=a.start_xy,noise_seed=a.noise_seed,walk_steps=a.walk_steps,
        arm_ik=a.arm_ik,text=a.text,text_cfg=a.text_cfg if a.text else None,root_constraints=a.root_constraints,
        history_frames=a.history_frames,lead_segment=str(a.lead_segment) if a.lead_segment else None,
        lead_segment_sha256=sha(a.lead_segment) if a.lead_segment else None,
        walk_clips=str(a.walk_clips) if a.walk_clips else None,walk_clips_sha256=sha(a.walk_clips) if a.walk_clips else None,
        generator=a.generator,segment_seconds=a.segment_seconds if a.generator!='policy' else None,
        segment_steps=a.segment_steps if a.generator!='policy' else None,
        transition_frames=a.transition_frames if a.generator=='native' else None,
        approach_gain=a.approach_gain,text_until_m=a.text_until_m or None,open_loop=a.open_loop)
    if demo:record.update(experiment='DEMO-REPLAY',oracle=False,target_source='demo_segment',
        demo_segment=str(a.demo_segment.resolve()),demo_segment_sha256=sha(a.demo_segment),start_xy=a.start_xy)
    if a.grasp_loop:record.update(experiment='GRASP-LOOP',target_source='latched_truth_pose_per_round',
        grasp_config_sha256=sha(grasp_config))
    if hammer:record.update(experiment='HAMMER-GRASP-LOOP',task=a.task,env_config=TASKS[a.task][1])
    if a.command_loop:record.update(experiment='COMMAND-LOOP',oracle=False,task=a.task,env_config=TASKS[a.task][1],
        target_source='commander',command_loop=str(a.command_loop.resolve()),command_loop_sha256=sha(a.command_loop),
        commander=a.commander,condition=a.condition,layout_id=a.seed,max_steps=a.steps,
        vlm_model=str(a.vlm_model) if a.vlm_model else None)
    record['arena_sha256']={f:sha(arena/f) for f in (
        'action_provider/action_provider_sonic.py','action_provider/vla_robot_current_local_runtime_v3.py',
        'tasks/common_test_config/base_test/pp_box_sonic_test.yaml')+((TASKS[a.task][1],) if a.task!='pp_box' else ())}
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
            if a.start_timeout_s:
                # DoubleDesk start-up sometimes deadlocks while loading materials, before anything runs.
                proc=subprocess.Popen(simcmd,env=simenv,cwd=arena,stdout=slog,stderr=subprocess.STDOUT,start_new_session=True)
                begun=time.monotonic()
                while proc.poll() is None:
                    waited=time.monotonic()-begun
                    hung=waited>a.start_timeout_s and not (run/'settling.jsonl').exists()
                    if hung or waited>max(1800,3*a.steps):
                        os.killpg(proc.pid,signal.SIGKILL); proc.wait()
                        raise RuntimeError('Simulator start-up hung' if hung else 'Simulator timed out')
                    time.sleep(2)
                sim=subprocess.CompletedProcess(simcmd,proc.returncode)
            else:
                sim=subprocess.run(simcmd,env=simenv,cwd=arena,stdout=slog,stderr=subprocess.STDOUT,timeout=max(1800,3*a.steps))
            record['sim_exit_code']=sim.returncode
            sim.check_returncode()
            evidence=run/('loop/commands.jsonl' if a.command_loop else 'plans.jsonl')
            if not (run/'execution.jsonl').is_file() or not evidence.is_file():
                raise RuntimeError('No actual execution or native-model evidence')
            if fixed and not (run/'commands.jsonl').is_file():
                raise RuntimeError('Missing prescribed command evidence')
            episode_result=json.loads((run/'episode.json').read_text())
            if episode_result.get('error'):
                raise RuntimeError(episode_result['error'])
            if a.grasp_loop:
                # The physical criterion decides success; Arena's own verdict and reward are kept beside it.
                result=json.loads((run/'grasp-result.json').read_text())
                video=Path(episode_result.get('video_path') or '')
                if result['success'] and video.is_file():
                    # Arena names the file after its own verdict, which knows nothing of this success.
                    episode_result['video_path']=str(video.rename(video.with_name(video.stem.rsplit('__',1)[0]+'__grasp_success.mp4')))
                result.update(termination='success' if result['success'] else episode_result['failure_reason'],
                    seed=a.seed,latch_delay_steps=a.latch_delay_steps,start_at_stance=a.start_at_stance,
                    stand_still=a.stand_still,native_walk=a.native_walk,**({'task':a.task} if hammer else {}),
                    episode_steps=episode_result['episode_steps'],
                    max_steps=a.steps,video_path=episode_result.get('video_path'),
                    arena={k:episode_result.get(k) for k in ('success','failure_reason','final_reward','max_reward')})
                (run/'grasp-result.json').write_text(json.dumps(result,indent=1))
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
