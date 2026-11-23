"""Read-only PhysX instrumentation around the unchanged SONIC provider/physics steps."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import numpy as np
from kimodo_vla.core.types import G1_JOINT_NAMES


def plain(value):
    if hasattr(value,'detach'):return value.detach().cpu().numpy().tolist()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,(dict,)):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    if hasattr(value,'to_dict'):return plain(value.to_dict())
    if hasattr(value,'GetReal') and hasattr(value,'GetImaginary'):
        return [float(value.GetReal()),*[float(v) for v in value.GetImaginary()]]
    if hasattr(value,'__iter__'):return [plain(v) for v in value]
    return str(value)


def array(value):
    return np.asarray(plain(value),dtype=float).copy()


def install_actuator_audit():
    factory=importlib.import_module('action_provider.create_action_provider')
    original_create=factory.create_action_provider

    def create(env,args):
        provider=original_create(env,args)
        robot=env.scene['robot'];view=robot.root_physx_view
        run=Path(os.environ['ORACLE_RUN']);names=list(robot.joint_names)
        ids=[names.index(n) for n in G1_JOINT_NAMES]
        sonic_ids=array(provider._sonic_idx).astype(int).tolist()
        sonic_to_canonical=[sonic_ids.index(i) for i in ids]
        active=False;control=0;substep=0;errors={}
        original_step=env.sim.step;original_action=provider.get_action
        metadata=dict(joint_names=names,body_names=list(robot.body_names),canonical_ids=ids,
            sonic_ids=sonic_ids,physics_dt=env.physics_dt,control_dt=env.step_dt,
            runtime_env={k:os.environ.get(k) for k in ('ROBOT_USD_OVERRIDE','MIMIC_LITE_ROBOT_CFG','SONIC_OUTPUT_DELAY_STEPS')},
            robot_cfg=plain(robot.cfg),sim_cfg=plain(env.cfg.sim),
            provider=dict(use_effort_control=provider._use_effort_control,output_delay_steps=provider._sonic_output_delay_steps,
                kp=plain(provider._pd_kp_t),kd=plain(provider._pd_kd_t),effort_limit=plain(provider._pd_effort_limit_t)),
            dof_paths=plain(view.dof_paths),link_paths=plain(view.link_paths),
            torque_note='IsaacLab computed/applied torque is an estimate using actuator cache; PhysX projected joint forces are the motion-axis component of incoming joint force, not an isolated motor torque sensor.')
        physics_file=(run/'actuator-physics.jsonl').open('x',buffering=1)
        controls_file=(run/'actuator-controls.jsonl').open('x',buffering=1)

        def optional(method):
            if method in errors:return None
            try:return array(getattr(view,method)())[0]
            except Exception as exc:
                errors[method]=repr(exc)
                (run/'actuator-query-errors.json').write_text(json.dumps(errors,indent=2))
                return None

        def capture_config(label):
            result={}
            for key in ('dof_stiffnesses','dof_dampings','dof_max_forces','dof_max_velocities',
                        'dof_armatures','dof_friction_coefficients','dof_limits','masses','coms','inertias','material_properties'):
                result[key]=plain(optional('get_'+key))
            result['actuator_cache']={name:dict(type=type(a).__name__,joint_indices=plain(a.joint_indices),
                stiffness=plain(a.stiffness),damping=plain(a.damping),effort_limit=plain(a.effort_limit),
                armature=plain(a.armature)) for name,a in robot.actuators.items()}
            metadata[label]=result
            (run/'actuator-config.json').write_text(json.dumps(plain(metadata),indent=2))

        capture_config('before_first_control')
        # USD metadata is read only: preserve actual inertial/drive/joint frames for audit.
        import omni.usd
        from pxr import Usd,UsdPhysics
        stage=omni.usd.get_context().get_stage();joints=[]
        root=stage.GetPrimAtPath('/World/envs/env_0/Robot')
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.Joint):
                j=UsdPhysics.Joint(prim)
                record=dict(name=prim.GetName(),path=str(prim.GetPath()),type=prim.GetTypeName(),
                    body0=[str(x) for x in j.GetBody0Rel().GetTargets()],body1=[str(x) for x in j.GetBody1Rel().GetTargets()],
                    attributes={a.GetName():plain(a.Get()) for a in prim.GetAttributes()
                                if a.GetName().startswith(('physics:','physxJoint:','drive:'))})
                joints.append(record)
        usd_path=Path(robot.cfg.spawn.usd_path)
        metadata['robot_asset']=dict(path=str(usd_path),sha256=hashlib.sha256(usd_path.read_bytes()).hexdigest() if usd_path.is_file() else None)
        (run/'actuator-usd-joints.json').write_text(json.dumps(joints,indent=2))

        def step(*args,**kwargs):
            nonlocal substep,view
            if not active:return original_step(*args,**kwargs)
            # Arena resets/reinitializes PhysX after provider creation. Never retain
            # the pre-reset tensor view, even though the robot object is unchanged.
            view=robot.root_physx_view
            if control==0 and substep==0:capture_config('active_first_physics_step')
            before_time=float(env.sim.current_time)
            q0=array(view.get_dof_positions())[0];v0=array(view.get_dof_velocities())[0]
            target=array(view.get_dof_position_targets())[0];velocity=array(view.get_dof_velocity_targets())[0]
            feedforward=array(view.get_dof_actuation_forces())[0]
            output=original_step(*args,**kwargs)
            row=dict(control=control,substep=substep,time_before=before_time,time_after=float(env.sim.current_time),
                q_before=q0[ids],qd_before=v0[ids],q_after=array(view.get_dof_positions())[0,ids],
                qd_after=array(view.get_dof_velocities())[0,ids],physx_position_target=target[ids],
                physx_velocity_target=velocity[ids],physx_feedforward=feedforward[ids],
                decoder_target=array(provider._latest_decoder_target)[sonic_to_canonical],
                estimated_isaac_torque=array(robot.data.applied_torque)[0,ids])
            for method,key in [('get_dof_projected_joint_forces','projected_joint_force'),
                               ('get_gravity_compensation_forces','gravity_compensation'),
                               ('get_coriolis_and_centrifugal_compensation_forces','coriolis_compensation')]:
                value=optional(method)
                if value is not None:
                    value=np.asarray(value).reshape(-1)
                    # Floating-base generalized forces prepend six root components.
                    if len(value) not in (len(names),len(names)+6):raise ValueError((method,value.shape))
                    row[key]=value[-len(names):][ids]
            physics_file.write(json.dumps(plain(row))+'\n');substep+=1
            return output

        def action(current_env):
            nonlocal active,control,substep,view
            active=True;substep=0
            try:result=original_action(current_env)
            finally:active=False
            view=robot.root_physx_view
            record=dict(control=control,time=float(env.sim.current_time),substeps=substep,
                        link_transforms=plain(view.get_link_transforms()[0]),
                        external_force=plain(robot._external_force_b),external_torque=plain(robot._external_torque_b))
            sensor=env.scene.sensors.get('contact_forces')
            if sensor is not None:
                record.update(contact_body_names=list(sensor.body_names),contact_net_forces=plain(sensor.data.net_forces_w[0]))
            controls_file.write(json.dumps(record)+'\n');control+=1
            return result

        env.sim.step=step;provider.get_action=action
        return provider
    factory.create_action_provider=create
