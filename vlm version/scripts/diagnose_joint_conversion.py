"""Separate raw rotation defects, scalar conversion, resampling and plan seams.

Nearest-axis and bounded projections are diagnostic counterfactuals only. No
projected pose is sent to the simulator, and no target is silently corrected.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from kimodo_vla.model.features import KimodoFeatures,G1_XML_PATH
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.motion.g1_reference import _interpolate_motion
from kimodo_vla.oracle.geometry import XmlFK


def angle(r):
    return np.rad2deg(np.arccos(np.clip((np.trace(r,axis1=-2,axis2=-1)-1)/2,-1,1)))


def project_axis(rot,axis,limits):
    skew=np.stack((rot[...,2,1]-rot[...,1,2],rot[...,0,2]-rot[...,2,0],rot[...,1,0]-rot[...,0,1]),-1)
    sine=np.einsum('...i,i->...',skew,axis)
    cosine=np.trace(rot,axis1=-2,axis2=-1)-np.einsum('i,...ij,j->...',axis,rot,axis)
    theta=np.arctan2(sine,cosine)
    candidates=np.stack([np.clip(theta+2*np.pi*k,*limits) for k in (-1,0,1)],axis=-1)
    score=sine[...,None]*np.sin(candidates)+cosine[...,None]*np.cos(candidates)
    bounded=np.take_along_axis(candidates,score.argmax(-1)[...,None],axis=-1)[...,0]
    return theta,bounded


def analyze(rot,q,codec,fk):
    nearest=np.zeros_like(q);bounded=np.zeros_like(q);offaxis=np.zeros_like(q);oldres=np.zeros_like(q);boundres=np.zeros_like(q)
    for xi,si in enumerate(codec.xml_to_skeleton.tolist()):
        ci=int(codec.canonical_to_xml[xi]);axis=codec.axes_kimodo[xi].numpy().astype(float)
        axis/=np.linalg.norm(axis)
        r=codec.rot_offsets_f2q[si].numpy()@rot[:,si]
        nearest[:,ci],bounded[:,ci]=project_axis(r,axis,fk.joint_limits[ci])
        for value,dest in [(nearest[:,ci],offaxis[:,ci]),(q[:,ci],oldres[:,ci]),(bounded[:,ci],boundres[:,ci])]:
            pr=Rotation.from_rotvec(value[:,None]*axis).as_matrix()
            dest[:]=angle(np.swapaxes(pr,-1,-2)@r)
    bad=(q<fk.joint_limits[:,0]-1e-5)|(q>fk.joint_limits[:,1]+1e-5)
    nearestbad=(nearest<fk.joint_limits[:,0]-1e-5)|(nearest>fk.joint_limits[:,1]+1e-5)
    excess=np.maximum(np.maximum(fk.joint_limits[:,0]-q,q-fk.joint_limits[:,1]),0)[bad]
    return {'frames':len(q),'scalar_limit_frames':int(bad.any(1).sum()),
        'nearest_axis_limit_frames':int(nearestbad.any(1).sum()),
        'off_axis_deg_mean':float(offaxis.mean()),'off_axis_deg_p95':float(np.quantile(offaxis,.95)),
        'off_axis_deg_max':float(offaxis.max()),'old_projection_residual_deg_mean':float(oldres.mean()),
        'bounded_projection_residual_deg_mean':float(boundres.mean()),
        'limit_by_joint':{n:int(v) for n,v in zip(G1_JOINT_NAMES,bad.sum(0)) if v},
        'limit_excess_deg_median_p95_max':np.rad2deg([np.median(excess),np.quantile(excess,.95),excess.max()]).tolist() if excess.size else [0.,0.,0.],
        'nearest_vs_old_q_abs_deg_mean':float(np.rad2deg(np.abs(nearest-q)).mean())},nearest,bounded,offaxis


def audit(run,output):
    f=KimodoFeatures(50);fk=XmlFK(G1_XML_PATH,G1_JOINT_NAMES)
    plans=[json.loads(x) for x in (run/'plans.jsonl').read_text().splitlines()]
    executions=[json.loads(x) for x in (run/'execution.jsonl').read_text().splitlines()]
    rawr=[];rawq=[];sentr=[];sentq=[];planids=[];reconstruction=[];condition=[]
    for plan in plans:
        d=np.load(run/f"plan-{plan['plan']:03d}.npz")
        dec=f.representation.inverse(torch.from_numpy(d['motion_features'])[None],is_normalized=False)
        r=dec['local_rot_mats'][0];root=dec['root_positions'][0]
        q=f.codec.encode(r,root)[:,9:38].numpy();rawr.append(r.numpy());rawq.append(q)
        t0=executions[0]['before']['time'];step=round((plan['time']-t0)*50)
        times=step*.6-plan['cut']+torch.arange(1,26)*.6
        rr,rp=_interpolate_motion(r,root,times)
        qq=f.codec.encode(rr,rp)[:,9:38].numpy()
        reconstruction.append(float(np.abs(qq-d['actions'][:,9:38]).max()))
        np.testing.assert_allclose(qq,d['actions'][:,9:38],atol=2e-5)
        sentr.append(rr.numpy());sentq.append(qq);planids.extend([plan['plan']]*25)
        mask=d['mask'];v=d['conditions'];pred=d['motion_features']
        for block,sl in f.representation.slice_dict.items():
            m=mask[:,sl]
            if m.any():condition.append({'plan':plan['plan'],'block':block,'cells':int(m.sum()),
                'mae':float(np.abs(pred[:,sl]-v[:,sl])[m].mean()),
                'max_abs':float(np.abs(pred[:,sl]-v[:,sl])[m].max())})
    rawr,rawq,sentr,sentq=map(np.concatenate,(rawr,rawq,sentr,sentq))
    raw,_,_,_=analyze(rawr,rawq,f.codec,fk)
    sent,near,bounded,offaxis=analyze(sentr,sentq,f.codec,fk)
    delta=np.abs(np.diff(sentq,axis=0));geo=angle(np.swapaxes(sentr[:-1],-1,-2)@sentr[1:])
    boundaries=np.flatnonzero(np.diff(planids)!=0)
    biggest=[]
    for flat in np.argsort(delta.ravel())[-8:][::-1]:
        i,j=np.unravel_index(flat,delta.shape)
        si=f.representation.skeleton.bone_order_names.index(G1_JOINT_NAMES[j].replace('_joint','_skel'))
        biggest.append({'step':int(i),'joint':G1_JOINT_NAMES[j],'scalar_step_deg':float(np.rad2deg(delta[i,j])),
            'raw_matrix_step_deg':float(geo[i,si]),'from_to_rad':sentq[i:i+2,j].tolist(),
            'off_axis_deg':offaxis[i:i+2,j].tolist(),'chunk_seam':bool(i in boundaries)})
    # Independent SciPy matrix log verifies the existing scalar projection math.
    independent=np.zeros_like(sentq)
    for xi,si in enumerate(f.codec.xml_to_skeleton.tolist()):
        rr=f.codec.rot_offsets_f2q[si].numpy()@sentr[:,si]
        independent[:,int(f.codec.canonical_to_xml[xi])]=Rotation.from_matrix(rr).as_rotvec()@f.codec.axes_kimodo[xi].numpy()
    seams=[]
    for boundary,plan in zip(boundaries,plans[1:]):
        actual=min(executions,key=lambda e:abs(e['before']['time']-plan['time']))['before']
        assert abs(actual['time']-plan['time'])<1e-6
        j=int(delta[boundary].argmax());measured=np.array(actual['joints'])
        seams.append({'step':int(boundary),'scalar_max_deg':float(np.rad2deg(delta[boundary].max())),
            'raw_matrix_max_deg':float(geo[boundary].max()),'largest_scalar_joint':G1_JOINT_NAMES[j],
            'new_reference_minus_measured_deg':float(np.rad2deg(sentq[boundary+1,j]-measured[j])),
            'old_reference_minus_measured_deg':float(np.rad2deg(sentq[boundary,j]-measured[j]))})
    result={'run':run.name,'raw_30hz':raw,'sent_50hz':sent,'masked_feature_residuals':condition,
        'saved_action_reconstruction_max_rad':max(reconstruction),
        'scipy_matrix_log_projection_max_difference_rad':float(np.abs(independent-sentq).max()),
        'largest_jumps':biggest,
        'seams':seams}
    output.mkdir(parents=True,exist_ok=True)
    (output/f'{run.name}.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(output/f'{run.name}.npz',sent_joint=sentq,nearest_joint=near,bounded_joint=bounded,
        off_axis_deg=offaxis,raw_matrix_step_deg=geo)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(audit(a.run,a.output),indent=2))
