"""One-factor, process-local dynamics interventions at the first replay physics step."""
import importlib
import json
import os
from pathlib import Path
import numpy as np
from arena.actuator_audit import array,plain
from kimodo_vla.core.types import G1_JOINT_NAMES

ROOT=Path(__file__).resolve().parents[1]
FIELDS=('dof_stiffnesses','dof_dampings','dof_max_forces','dof_max_velocities',
        'dof_armatures','dof_friction_coefficients','dof_limits','masses','coms','inertias')


def intervene(state, joint_names, body_names, profile, spec):
    """Return changed values; untouched values must remain byte-identical."""
    if profile=='actuator-load':
        aligned=intervene(state,joint_names,body_names,'actuator',spec)
        return intervene(aligned,joint_names,body_names,'sensor-load',spec)
    result={k:np.array(v,copy=True) for k,v in state.items()}
    if profile=='sensor-load':
        for name in spec['sensor-load']['bodies']:
            i=body_names.index(name);old=float(result['masses'][i])
            if not np.isclose(old,1.,atol=1e-7):raise ValueError('Unexpected sensor mass: '+name)
            mass=spec['sensor-load']['mass_kg']
            if not 0<mass<old:raise ValueError('Invalid positive surrogate mass')
            result['masses'][i]=mass;result['inertias'][i]*=mass/old
    elif profile in ('actuator','heft'):   # heft: spec is configs/heft-dynamics.json (HEFT's own MuJoCo drives)
        for key,values in spec['actuator'].items():
            if set(values)!=set(G1_JOINT_NAMES):raise ValueError('Incomplete named actuator profile')
            for name,value in values.items():result[key][joint_names.index(name)]=value
    else:raise ValueError('Unknown dynamics profile')
    return result


def install_dynamics_override():
    import torch
    factory=importlib.import_module('action_provider.create_action_provider');original_create=factory.create_action_provider
    profile=os.environ['SONIC_DYNAMICS_PROFILE']
    spec=json.loads((ROOT/('configs/heft-dynamics.json' if profile=='heft' else 'configs/sonic-dynamics.json')).read_text())

    def create(env,args):
        provider=original_create(env,args);original_step=env.sim.step;original_action=provider.get_action
        robot=env.scene['robot'];active=False;applied=False;run=Path(os.environ['ORACLE_RUN'])

        def step(*args,**kwargs):
            nonlocal applied
            if active and not applied:
                view=robot.root_physx_view
                tensors={k:getattr(view,'get_'+k)().clone() for k in FIELDS}
                before={k:array(v)[0] for k,v in tensors.items()}
                expected=intervene(before,list(robot.joint_names),list(robot.body_names),profile,spec)
                idx=torch.tensor([0],dtype=torch.int32,device='cpu')
                changed=[k for k in FIELDS if not np.array_equal(before[k],expected[k])]
                for key in changed:
                    tensor=tensors[key];tensor[0]=torch.as_tensor(expected[key],dtype=tensor.dtype,device=tensor.device)
                    getattr(view,'set_'+key)(tensor,idx)
                after={k:array(getattr(view,'get_'+k)())[0] for k in FIELDS}
                for key in FIELDS:
                    if not np.allclose(after[key],expected[key],rtol=1e-6,atol=1e-8):raise ValueError('PhysX rejected profile: '+key)
                    if key not in changed and not np.array_equal(after[key],before[key]):raise ValueError('Unintended dynamics change: '+key)
                (run/'dynamics-intervention.json').write_text(json.dumps(plain(dict(profile=profile,spec=spec,
                    applied_at_time=float(env.sim.current_time),changed_fields=changed,before=before,after=after)),indent=2))
                applied=True
            return original_step(*args,**kwargs)

        def action(current_env):
            nonlocal active
            active=True
            try:return original_action(current_env)
            finally:active=False

        env.sim.step=step;provider.get_action=action
        return provider
    factory.create_action_provider=create
