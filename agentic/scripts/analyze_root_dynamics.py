"""Root/foot/posture decomposition for archived SONIC controls; no target correction."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from paths import G1_XML
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.oracle.geometry import XmlFK,rotation_wxyz
from kimodo_vla.oracle.recorded_reference import read_rows
from analyze_sonic_replay import vector_stats
FEET=['left_ankle_roll_link','right_ankle_roll_link']

def decompose(root,refroot,feet,reffoot,localfeet,reflocal,rotation,refrotation):
    # Per-foot exact identity, including independent FK residual.
    contact=feet-reffoot
    attitude=reflocal@(rotation-refrotation).T
    posture=(localfeet-reflocal)@rotation.T
    residual=feet-(root+localfeet@rotation.T)
    reconstructed=contact-attitude-posture-residual
    expected=np.broadcast_to(root-refroot,(2,3))
    if not np.allclose(reconstructed,expected,atol=1e-9,rtol=0):raise ValueError('Root-foot identity failed')
    return dict(foot_placement=contact,negative_root_rotation=-attitude,negative_leg_posture=-posture,negative_fk_residual=-residual,root_error=expected)

def analyze(run):
    run=Path(run);rows=read_rows(run/'execution.jsonl');cfg=json.loads((run/'actuator-config.json').read_text());ctrl=read_rows(run/'actuator-controls.jsonl')
    assert len(rows)==len(ctrl)==400
    fk=XmlFK(G1_XML,G1_JOINT_NAMES)
    times=np.array([r['after']['time'] for r in rows]);arrival=json.loads((run/'target.json').read_text())['arrival_time'];start=rows[0]['before']['time']
    root=np.array([r['after']['root_pos'] for r in rows]);refroot=np.array([r['reference_root_world'] for r in rows]);errors=root-refroot
    parts=[];angles=[];com=[];feet=np.array([r['after']['foot_pos'] for r in rows]);state=cfg['active_first_physics_step'];m=np.array(state['masses']);c=np.array(state['coms'])[:,:3]
    for r,k in zip(rows,ctrl):
        a=r['after'];R=rotation_wxyz(a['root_quat']);Rr=rotation_wxyz(r['reference_root_quat']);local=fk(a['joints'],np.zeros(3),np.eye(3),FEET)[0];reflocal=fk(r['reference_joints'],np.zeros(3),np.eye(3),FEET)[0]
        parts.append(decompose(np.array(a['root_pos']),np.array(r['reference_root_world']),np.array(a['foot_pos']),np.array(r['reference_foot_world']),local,reflocal,R,Rr))
        angles.append(Rotation.from_matrix(Rr.T@R).as_rotvec()*180/np.pi)
        t=np.array(k['link_transforms']);rot=Rotation.from_quat(t[:,3:]).as_matrix();world=t[:,:3]+np.einsum('nij,nj->ni',rot,c);com.append(np.sum(m[:,None]*world,axis=0)/m.sum())
    q=np.array([r['after']['joints'] for r in rows]);qr=np.array([r['reference_joints'] for r in rows]);qd=np.array([r['decoder_joint_target'] for r in rows])
    force=np.array([k['contact_net_forces'] for k in ctrl]);names=ctrl[0]['contact_body_names'];ids=[names.index(n) for n in FEET];footforce=force[:,ids,:]
    out=dict(run=run.name,total_mass_kg=float(m.sum()),initial_reference_foot_xyz_m=rows[0]['reference_foot_world'],initial_measured_foot_xyz_m=rows[0]['before']['foot_pos'],windows={})
    for name,mask in [('arrival',times>arrival+1e-6),('steady',times>=start+4)]:
        e=100*errors[mask];s=dict(frames=int(mask.sum()),root_xyz_bias_cm=e.mean(axis=0).tolist(),root_mean_cm=float(np.linalg.norm(e,axis=1).mean()),root_max_cm=float(np.linalg.norm(e,axis=1).max()),root_xy_mean_cm=float(np.linalg.norm(e[:,:2],axis=1).mean()),root_abs_z_mean_cm=float(abs(e[:,2]).mean()),root_rotvec_bias_deg=np.array(angles)[mask].mean(axis=0).tolist())
        s['decomposition']={k:vector_stats(np.array([p[k] for p in parts])[mask]) for k in parts[0]}
        s['foot_z_mean_m']=feet[mask,:,2].mean(axis=0).tolist();s['foot_horizontal_range_cm']=(100*np.ptp(feet[mask,:,:2],axis=0)).tolist()
        s['vertical_contact_force_mean_n']=footforce[mask,:,2].mean(axis=0).tolist();s['double_support_fraction']=float(np.mean(np.all(footforce[mask,:,2]>10,axis=1)))
        s['com_relative_foot_midpoint_cm']=(100*(np.array(com)[mask]-feet[mask].mean(axis=1))).mean(axis=0).tolist()
        s['leg_joints']={}
        for key,diff in [('actual_reference',q-qr),('actual_decoder',q-qd),('decoder_reference',qd-qr)]:
            d=np.degrees(diff[mask]);s['leg_joints'][key]={n:dict(bias_deg=float(d[:,i].mean()),rmse_deg=float(np.sqrt(np.mean(d[:,i]**2)))) for i,n in enumerate(G1_JOINT_NAMES) if any(x in n for x in ('hip_','knee_','ankle_'))}
        out['windows'][name]=s
    return out
