"""Audit recorded PhysX targets/parameters and steady gravity balance; never tune gains."""
import argparse
import ast
import copy
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.oracle.recorded_reference import read_rows


def pd_estimate(q, qd, target, velocity, ff, kp, kd):
    """An estimate, not the implicit solver's measured motor torque."""
    return kp*(target-q)+kd*(velocity-qd)+ff


def gravity_from_links(config, joints, transforms):
    """Independent gravity compensation from actual COMs, masses and USD joint frames.

    PhysX link transforms/COM quaternions are XYZW; USD localRot1 is WXYZ.
    This computes the external gravitational moment of all descendant links.
    It is valid for uncontacted arms; contacts are audited separately.
    """
    names=config['body_names']; index={n:i for i,n in enumerate(names)}
    state=config['active_first_physics_step']
    mass=np.asarray(state['masses']); local_com=np.asarray(state['coms'])[:,:3]
    poses=np.asarray(transforms); rotations=Rotation.from_quat(poses[:,3:7]).as_matrix()
    com=poses[:,:3]+np.einsum('nij,nj->ni',rotations,local_com)
    gravity=np.asarray(config['sim_cfg']['gravity'])
    children={}; by_name={}
    for joint in joints:
        if not joint['body0'] or not joint['body1']:continue
        parent=joint['body0'][0].rsplit('/',1)[-1];child=joint['body1'][0].rsplit('/',1)[-1]
        children.setdefault(parent,[]).append(child);by_name[joint['name']]=(child,joint['attributes'])
    def descendants(name):
        return [name]+[v for ch in children.get(name,[]) for v in descendants(ch)]
    result={}
    for name in G1_JOINT_NAMES:
        child,attrs=by_name[name];i=index[child]
        axis=np.eye(3)['XYZ'.index(attrs['physics:axis'])]
        quat=np.asarray(attrs['physics:localRot1'])
        axis=rotations[i]@Rotation.from_quat(quat[[1,2,3,0]]).apply(axis)
        local=attrs['physics:localPos1']
        if isinstance(local,str):local=ast.literal_eval(local)
        origin=poses[i,:3]+rotations[i]@np.asarray(local)
        ids=[index[n] for n in descendants(child) if n in index]
        result[name]=float(-np.sum(np.cross(com[ids]-origin,mass[ids,None]*gravity)@axis))
    return np.array([result[n] for n in G1_JOINT_NAMES])


def summarize(run):
    run=Path(run);config=json.loads((run/'actuator-config.json').read_text())
    rows=read_rows(run/'actuator-physics.jsonl');controls=read_rows(run/'actuator-controls.jsonl')
    execution=read_rows(run/'execution.jsonl');n=len(execution)
    if len(rows)!=4*n or len(controls)!=n:raise ValueError('Missing physical/control samples')
    if [(r['control'],r['substep']) for r in rows]!=[(i,j) for i in range(n) for j in range(4)]:
        raise ValueError('Physical substep order changed')
    if any(r['substeps']!=4 for r in controls):raise ValueError('Wrong control decimation')
    times=np.array([r['time_after'] for r in rows]);start=rows[0]['time_before']
    if not np.allclose(times-np.array([r['time_before'] for r in rows]),.005,atol=1e-6):
        raise ValueError('Physics dt changed')
    state=config['active_first_physics_step'];ids=config['canonical_ids']
    def prop(key):return np.asarray(state[key])[ids]
    def values(key):return np.asarray([r[key] for r in rows])
    kp=prop('dof_stiffnesses');kd=prop('dof_dampings');limit=prop('dof_max_forces')
    q=values('q_after');qd=values('qd_after');target=values('physx_position_target')
    decoded=values('decoder_target');ff=values('physx_feedforward');velocity=values('physx_velocity_target')
    actual_torque_estimate=pd_estimate(q,qd,target,velocity,ff,kp,kd)
    mask=times>=start+4-1e-7;controlmask=np.array([c['time'] for c in controls])>=start+4-1e-7
    provider=config['provider'];order=[config['sonic_ids'].index(i) for i in ids]
    provider_kp=np.asarray(provider['kp']).reshape(-1)[order];provider_kd=np.asarray(provider['kd']).reshape(-1)[order]
    output=dict(run=run.name,control_frames=n,physics_steps=len(rows),steady_physics_steps=int(mask.sum()),
        decoder_to_physx_target_max_abs_rad=float(abs(target-decoded).max()),
        provider_to_physx_kp_max_abs=float(abs(kp-provider_kp).max()),
        provider_to_physx_kd_max_abs=float(abs(kd-provider_kd).max()),
        feedforward_max_abs_nm=float(abs(ff).max()),velocity_target_max_abs=float(abs(velocity).max()),
        external_force_max_abs=float(max(np.abs(c['external_force']).max() for c in controls)),
        external_torque_max_abs=float(max(np.abs(c['external_torque']).max() for c in controls)),
        total_mass_kg=float(np.sum(state['masses'])),joints={},
        note='PD values are estimates using actual PhysX gains; projected joint force is not an isolated motor torque measurement.')
    for label,array in [('q',q),('target',target)]:
        bounds=prop('dof_limits')
        output[label+'_limit_violation_max_rad']=float(np.maximum(np.maximum(bounds[:,0]-array,array-bounds[:,1]),0).max())
    usd=json.loads((run/'actuator-usd-joints.json').read_text())
    geometric_gravity=np.array([gravity_from_links(config,usd,c['link_transforms']) for c in controls])
    camera_config=copy.deepcopy(config)
    camera_config['active_first_physics_step']['masses']=[
        m if name.endswith('hand_camera_base_link') else 0.
        for name,m in zip(config['body_names'],state['masses'])]
    camera_gravity=np.array([gravity_from_links(camera_config,usd,c['link_transforms']) for c in controls])
    after_step_indices=np.arange(3,len(rows),4)
    if 'gravity_compensation' in rows[0]:
        output['gravity_api_vs_link_geometry_max_abs_nm']=float(abs(values('gravity_compensation')[after_step_indices]-geometric_gravity).max())
    for i,name in enumerate(G1_JOINT_NAMES):
        error=q[mask,i]-target[mask,i];estimated=actual_torque_estimate[mask,i]
        geometric=geometric_gravity[controlmask,i];steady_ctrl=after_step_indices[controlmask]
        j=dict(kp=float(kp[i]),kd=float(kd[i]),force_limit_nm=float(limit[i]),
            armature=float(prop('dof_armatures')[i]),friction=float(prop('dof_friction_coefficients')[i]),
            actual_target_bias_deg=float(np.degrees(error).mean()),actual_target_std_deg=float(np.degrees(error).std()),
            actual_target_control_end_bias_deg=float(np.degrees(q[steady_ctrl,i]-target[steady_ctrl,i]).mean()),
            velocity_rms_rad_s=float(np.sqrt(np.mean(qd[mask,i]**2))),
            pd_estimate_mean_nm=float(estimated.mean()),pd_estimate_max_abs_nm=float(abs(estimated).max()),
            pd_estimate_full_run_max_limit_fraction=float(np.max(abs(actual_torque_estimate[:,i]))/limit[i]),
            geometric_gravity_mean_nm=float(geometric.mean()),
            hand_camera_gravity_mean_nm=float(camera_gravity[controlmask,i].mean()),
            gravity_predicted_bias_deg=float(np.degrees(-geometric/kp[i]).mean()),
            pd_minus_geometric_gravity_mean_nm=float((actual_torque_estimate[steady_ctrl,i]-geometric).mean()),
            gravity_bias_residual_mean_deg=float(np.degrees(q[steady_ctrl,i]-target[steady_ctrl,i]+geometric/kp[i]).mean()))
        for key in ('projected_joint_force','gravity_compensation','coriolis_compensation','estimated_isaac_torque'):
            if key in rows[0]:
                v=values(key)[mask,i];j[key+'_mean_nm']=float(v.mean());j[key+'_max_abs_nm']=float(abs(v).max())
        output['joints'][name]=j
    if 'contact_net_forces' in controls[0]:
        contacts=np.asarray([c['contact_net_forces'] for c in controls]);cnames=controls[0]['contact_body_names']
        output['contact_max_n']={name:float(np.linalg.norm(contacts[:,i],axis=-1).max()) for i,name in enumerate(cnames)}
        feet=[i for i,name in enumerate(cnames) if name.endswith('ankle_roll_link')]
        output['steady_total_foot_force_mean_n']=contacts[controlmask][:,feet].sum(axis=1).mean(axis=0).tolist()
        output['total_gravity_force_n']=(np.sum(state['masses'])*np.asarray(config['sim_cfg']['gravity'])).tolist()
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=summarize(a.run);a.output.write_text(json.dumps(result,indent=2))
    print(result['run'],result['physics_steps'],'physics steps; target max delta',result['decoder_to_physx_target_max_abs_rad'])
