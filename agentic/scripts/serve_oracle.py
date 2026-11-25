"""Native Kimodo experiment A, bound only to localhost."""
import argparse
import sys
from pathlib import Path
import kimodo_vla
from kimodo_vla.oracle.policy import OraclePolicy
from kimodo_vla.benchmarks.server import make_server

# Another checkout of the package may be installed in the same environment; never run on it silently.
if Path(__file__).resolve().parents[2] not in Path(kimodo_vla.__file__).resolve().parents:
    raise SystemExit(f'kimodo_vla was imported from {kimodo_vla.__file__}, not from this checkout')

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', required=True)
p.add_argument('--text-cache', required=True)
p.add_argument('--run', required=True)
p.add_argument('--mode', choices=['hold','reach','offset','box'], default='hold')
p.add_argument('--port', type=int, required=True)
p.add_argument('--diffusion-steps', type=int, default=50)
p.add_argument('--constraint-cfg', type=float, default=2.)
p.add_argument('--text-mode', choices=['none','task'], default='none')
p.add_argument('--root-anchor', choices=['reference','generated'], default='reference')
p.add_argument('--variant',choices=['original','handoff','bounded'],default='bounded')
p.add_argument('--handoff-seconds',type=float,default=.2)
p.add_argument('--joint-speed',type=float,default=4.)
p.add_argument('--support-constraints',choices=['none','root','root-feet','root-feet-keyframes'],default='root')
p.add_argument('--hand-constraints',choices=['keyframes','continuous'],default='continuous')
p.add_argument('--trajectory-projection',choices=['none','ik','ik-hold'],default='ik-hold')
p.add_argument('--archived-goal')
p.add_argument('--motion-goal')
p.add_argument('--noise-seed',type=int)
p.add_argument('--walk-steps',type=int,default=75)
p.add_argument('--arm-ik',action='store_true')
p.add_argument('--text')
p.add_argument('--text-cfg',type=float,default=2.)
p.add_argument('--root-constraints',choices=['dense','sparse','none'],default='dense')
p.add_argument('--history-frames',type=int,default=100)
p.add_argument('--lead-segment')
p.add_argument('--walk-clips')
p.add_argument('--generator',choices=['policy','segment','native'],default='policy')
p.add_argument('--transition-frames',type=int,default=5)
p.add_argument('--approach-gain',type=float,default=1.)
p.add_argument('--text-until-m',type=float,default=0.)
p.add_argument('--open-loop',action='store_true')
p.add_argument('--segment-seconds',type=float,default=4.)
p.add_argument('--segment-steps',type=int,default=100)
p.add_argument('--grasp-loop',help='Grasp loop configuration (JSON); selects the blocking box grasp policy')
p.add_argument('--grasp-task',choices=['box','hammer'],default='box',help='Grasp loop: the box, or the one-handed DoubleDesk hammer')
p.add_argument('--latch-delay-steps',type=int,default=75)
p.add_argument('--stand-still',action='store_true',help='Grasp loop: never walk, reach from where the robot stands')
p.add_argument('--native-walk',action='store_true',help='Grasp loop: walk the official Kimodo way (settings: the configuration\'s wide section)')
p.add_argument('--target-update',choices=['blocking','fixed','anchored'],default='blocking',
               help='Grasp loop: when the reach target is refreshed (kimodo_vla/oracle/async_grasp.py)')
p.add_argument('--latency-samples',help='Grasp loop: JSON list of measured call latencies in seconds to draw delays from')
p.add_argument('--perception',choices=['truth','tracker'],default='truth',
               help="Grasp loop, anchored: where the box is now, from the simulator or from two tracked points")
p.add_argument('--tracker-source',help='With --perception tracker: folder holding the co-tracker package')
p.add_argument('--tracker-checkpoint',help='With --perception tracker: CoTracker3 online weights')
p.add_argument('--vlm-model')
p.add_argument('--vlm-localizer',choices=['baseline','refined'],default='baseline')
p.add_argument('--vlm-python',default=sys.executable)
a = p.parse_args()
policy_class=OraclePolicy
extra={}
if a.archived_goal:
    from kimodo_vla.oracle.archived_goal import ArchivedGoalPolicy
    policy_class = ArchivedGoalPolicy
    extra = dict(archived_goal=a.archived_goal)
elif a.motion_goal:
    from kimodo_vla.oracle.locomotion import LocomotionPolicy
    policy_class = LocomotionPolicy
    extra = dict(motion_goal=a.motion_goal, noise_seed=a.noise_seed, walk_steps=a.walk_steps, arm_ik=a.arm_ik,
                 text=a.text, text_cfg=a.text_cfg, root_constraints=a.root_constraints,
                 history_frames=a.history_frames, lead_segment=a.lead_segment, walk_clips=a.walk_clips,
                 generator=a.generator, segment_seconds=a.segment_seconds, segment_steps=a.segment_steps,
                 transition_frames=a.transition_frames, approach_gain=a.approach_gain, text_until_m=a.text_until_m,
                 open_loop=a.open_loop)
elif a.grasp_loop:
    import json
    from kimodo_vla.oracle.grasp_policy import BoxGraspPolicy
    policy_class = BoxGraspPolicy
    if a.grasp_task == 'hammer':
        from kimodo_vla.oracle.hammer_policy import HammerGraspPolicy
        policy_class = HammerGraspPolicy
    grasp_config = json.loads(Path(a.grasp_loop).read_text())
    extra = dict(config=grasp_config['policy'], latch_delay_steps=a.latch_delay_steps,
                 stand_still=a.stand_still, native_walk=grasp_config['wide'] if a.native_walk else None)
    if a.target_update != 'blocking' or a.latency_samples:
        from kimodo_vla.oracle.async_grasp import AsyncBoxGraspPolicy, load_latencies
        policy_class = AsyncBoxGraspPolicy
        extra.update(target_update=a.target_update,
                     latency_samples=load_latencies(a.latency_samples) if a.latency_samples else None)
    if a.perception == 'tracker':
        if a.target_update != 'anchored':
            raise ValueError('The tracker only replaces the anchored target')
        sys.path.insert(0, a.tracker_source)
        from kimodo_vla.vlm.box_tracker import BoxPointTracker
        extra['perception'] = BoxPointTracker(a.run, extra['config']['box_half_m'], a.tracker_checkpoint)
elif a.vlm_model:
    from kimodo_vla.vlm.policy import VisualTargetPolicy
    if a.mode!='hold' or a.trajectory_projection!='ik-hold':raise ValueError('Visual mode requires the standing IK-hold backend')
    policy_class=VisualTargetPolicy
    extra=dict(vlm_model=a.vlm_model,vlm_python=a.vlm_python,vlm_localizer=a.vlm_localizer)
policy = policy_class(a.checkpoint, a.text_cache, a.run, a.mode, diffusion_steps=a.diffusion_steps,
                      constraint_cfg=a.constraint_cfg, text_mode=a.text_mode, root_anchor=a.root_anchor,
                      variant=a.variant,handoff_seconds=a.handoff_seconds,joint_speed=a.joint_speed,
                      support_constraints=a.support_constraints,hand_constraints=a.hand_constraints,trajectory_projection=a.trajectory_projection,**extra)
routes = {('GET','/health'):lambda _: (200,{'status':'ok','native':True}),
          ('POST','/reset'):lambda r: (200,policy.reset(r.get('seed'))),
          ('POST','/infer'):lambda r: (200,policy(r))}
make_server(routes, '127.0.0.1', a.port).serve_forever()
