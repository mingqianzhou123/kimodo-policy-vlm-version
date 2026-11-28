"""Native Kimodo experiment A, bound only to localhost."""
import argparse
import json
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
p.add_argument('--command-loop',help='Command loop configuration (JSON): a commander issues body commands one at a time (kimodo_vla/agent)')
p.add_argument('--commander',help='With --command-loop: which commander of the configuration')
p.add_argument('--command-task',choices=['pp_box','sit_sofa','vision_navi'],help='With --command-loop: the task')
p.add_argument('--condition',default='',help='With --command-loop: condition name, recorded only')
p.add_argument('--tracker',default='sonic',help='With --command-loop: the tracker under the planner (selects its avoid rules)')
p.add_argument('--carry-goal',help='Grasp loop: after a lifted grasp, walk back and run this motion goal carrying the box (kimodo_vla/oracle/carry_policy.py)')
p.add_argument('--carry-noise-seed',type=int,help='With --carry-goal: Kimodo sampling noise of the carrying walks')
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
elif a.vlm_model:
    from kimodo_vla.vlm.policy import VisualTargetPolicy
    if a.mode!='hold' or a.trajectory_projection!='ik-hold':raise ValueError('Visual mode requires the standing IK-hold backend')
    policy_class=VisualTargetPolicy
    extra=dict(vlm_model=a.vlm_model,vlm_python=a.vlm_python,vlm_localizer=a.vlm_localizer)
policy = None
if a.command_loop:
    import json
    from kimodo_vla.agent.commander import ScriptedCommander, VLMCommander, load_prompt
    from kimodo_vla.agent.loop import CommandLoopPolicy
    config_path = Path(a.command_loop).resolve()
    config = json.loads(config_path.read_text())
    spec, task = config['commanders'][a.commander], config['tasks'][a.command_task]
    limits = config.get('limits')
    if spec['kind'] == 'scripted':
        commander = ScriptedCommander.from_file(config_path.with_name(spec['file']), spec['name'],
                                                {**spec.get('options', {}), 'task': a.command_task}, limits)
    else:
        from kimodo_vla.vlm.backends import API_BACKENDS, QwenBackend
        backend = QwenBackend(Path(a.vlm_model)) if spec['backend'] == 'qwen' else API_BACKENDS[spec['backend']](spec['api'])
        commander = VLMCommander(backend, load_prompt(config_path.with_name(spec['page'])), limits, spec.get('max_tokens', 256))
    # A commander may override planner settings (e.g. the wrist feedback of the first batch's grasp).
    config = {**config, 'planner': {**config.get('planner', {}), **spec.get('planner', {}), 'tracker': a.tracker}}
    policy = CommandLoopPolicy(a.checkpoint, a.text_cache, a.run, config=config, task_text=task['task_text'],
        commander=commander, meta={'task': a.command_task, 'commander_name': a.commander, 'commander_spec': spec,
                                   'condition': a.condition, 'tracker': a.tracker, 'config': str(config_path)})
if policy is None:
    policy = policy_class(a.checkpoint, a.text_cache, a.run, a.mode, diffusion_steps=a.diffusion_steps,
                          constraint_cfg=a.constraint_cfg, text_mode=a.text_mode, root_anchor=a.root_anchor,
                          variant=a.variant,handoff_seconds=a.handoff_seconds,joint_speed=a.joint_speed,
                          support_constraints=a.support_constraints,hand_constraints=a.hand_constraints,trajectory_projection=a.trajectory_projection,**extra)
if a.carry_goal:
    if not a.grasp_loop or a.grasp_task!='box':raise ValueError('--carry-goal carries the box after the box grasp loop')
    from kimodo_vla.oracle.carry_policy import CarryPolicy
    from kimodo_vla.oracle.locomotion import LocomotionPolicy
    grasp_model = policy.model

    def make_locomotion(goal, folder, anchor_heading):
        # The settings of the 2026-11-25 native walks (experiment 2's command test), on the grasp policy's model.
        folder.mkdir(parents=True, exist_ok=True)
        return LocomotionPolicy(a.checkpoint, a.text_cache, str(folder), 'hold', diffusion_steps=a.diffusion_steps,
            constraint_cfg=a.constraint_cfg, text_mode='none', root_anchor=a.root_anchor, variant='bounded',
            handoff_seconds=a.handoff_seconds, joint_speed=a.joint_speed, support_constraints='none',
            hand_constraints='keyframes', trajectory_projection='none', motion_goal=goal, noise_seed=a.carry_noise_seed,
            walk_steps=50, generator='native', segment_seconds=5., segment_steps=100, transition_frames=5,
            approach_gain=2., model=grasp_model, anchor_heading=anchor_heading)
    policy = CarryPolicy(policy, make_locomotion, json.loads(Path(a.carry_goal).read_text()), a.run)
routes = {('GET','/health'):lambda _: (200,{'status':'ok','native':True}),
          ('POST','/reset'):lambda r: (200,policy.reset(r.get('seed'))),
          ('POST','/infer'):lambda r: (200,policy(r))}
make_server(routes, '127.0.0.1', a.port).serve_forever()
