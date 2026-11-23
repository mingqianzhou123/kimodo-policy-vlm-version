"""Separate world support drift, reference transport, and physical root/foot tracking."""
import argparse,json
from pathlib import Path
import numpy as np
from kimodo_vla.oracle.geometry import angle_deg,rotation_wxyz,Frame
from kimodo_vla.model.features import KimodoFeatures
from kimodo_vla.oracle.support import condition_support
import torch
from summarize_continuity import summarize as summarize_continuity
from summarize_oracle import stats


def summarize(run):
    base=summarize_continuity(run)
    read=lambda f:[json.loads(x) for x in (run/f).read_text().splitlines()]
    commands,execution=read('commands.jsonl'),read('execution.jsonl')
    start=execution[0]['before']['time']
    lookup={round((c['time']-start)*50):c for c in commands}
    target=json.loads((run/'target.json').read_text())['support_target']
    root=np.array(target['root_world']);rotation=np.array(target['root_rotation_world'])
    feet=np.array(target['ankle_world']);rows=[]
    yaw=lambda r:np.arctan2(r[1,0],r[0,0])
    yaw_error=lambda a,b:float(abs(np.degrees(np.arctan2(np.sin(yaw(a)-yaw(b)),np.cos(yaw(a)-yaw(b))))))
    for e in execution:
        step=round((e['after']['time']-start)*50)
        if step not in lookup:continue
        c=lookup[step];actual=e['after']
        row={'step':step,'elapsed':step/50}
        poses={'raw':(c['raw_root_world'],c['raw_root_rotation_world'],c['raw_model_feet']),
               'sent':(c['sent_root_world'],c['sent_root_rotation_world'],c['sent_feet']),
               'consumed':(e['reference_root_world'],rotation_wxyz(e['reference_root_quat']),e['reference_foot_world']),
               'actual':(actual['root_pos'],rotation_wxyz(actual['root_quat']),actual['foot_pos'])}
        for name,(p,r,f) in poses.items():
            delta=np.array(p)-root
            row.update({name+'_root_goal_m':float(np.linalg.norm(delta)),
                        name+'_root_xy_goal_m':float(np.linalg.norm(delta[:2])),
                        name+'_root_height_abs_m':float(abs(delta[2])),
                        name+'_root_rotation_goal_deg':angle_deg(rotation,np.array(r)),
                        name+'_root_yaw_goal_deg':yaw_error(rotation,np.array(r)),
                        name+'_foot_goal_m':np.linalg.norm(np.array(f)-feet,axis=-1).tolist()})
        p,r,f=poses['actual'];rp,rr,rf=poses['consumed']
        row.update(root_tracking_m=float(np.linalg.norm(np.array(p)-rp)),
            root_xy_tracking_m=float(np.linalg.norm(np.array(p)[:2]-np.array(rp)[:2])),
            root_rotation_tracking_deg=angle_deg(rr,r),root_yaw_tracking_deg=yaw_error(rr,r),
            foot_tracking_m=np.linalg.norm(np.array(f)-rf,axis=-1).tolist(),
            foot_xy_drift_m=np.linalg.norm(np.array(f)[:,:2]-feet[:,:2],axis=-1).tolist(),
            foot_height_abs_m=abs(np.array(f)[:,2]-feet[:,2]).tolist(),
            foot_speed_m_s=(np.linalg.norm(np.array(f)-np.array(e['before']['foot_pos']),axis=-1)/.02).tolist(),
            foot_xml_vs_sim_m=e['measured_xml_vs_sim_foot_m'])
        rows.append(row)
    result={'run':run.name,'support_constraints':json.loads((run/'run.json').read_text())['support_constraints'],
            'definition':'feet=ankle_roll_link origins; drift is displacement, not force-confirmed slip',
            'units':'metres, degrees, m/s','hand_and_joint_metrics':base}
    keys=[k for k in rows[0] if k not in ('step','elapsed')]
    for label,subset in [('all',rows),('matched_1p2_to_4s',[r for r in rows if 1.2<r['elapsed']<=4])]:
        result[label]={k:stats(subset,k) for k in keys}
    # Future prediction windows overlap: keep these 30 Hz diagnostics separate
    # from the 50 Hz actually sent/executed trajectory above.
    rep=KimodoFeatures(50).representation
    info=json.loads((run/'target.json').read_text())
    frame=Frame(info['origin'],np.array(info['heading']))
    ankle_ids=[rep.skeleton.bone_order_names.index(f'{side}_ankle_roll_skel') for side in ('left','right')]
    future=[];hand_keyframes=[]
    hand_goal=np.array(info['targets_world'])[:,0]
    for plan in read('plans.jsonl'):
        d=np.load(run/f"plan-{plan['plan']:03d}.npz")
        if result['support_constraints']!='none':
            expected=torch.zeros(50,417);known=torch.zeros_like(expected,dtype=torch.bool)
            condition_support(expected,known,torch.from_numpy(d['standing_features']),rep,
                              result['support_constraints'],keyframes=plan['keyframes'])
            known=known.numpy()
            assert d['mask'][known].all()
            np.testing.assert_allclose(d['conditions'][known],expected.numpy()[known],atol=1e-6)
        feat=d['motion_features']
        local=feat[:,rep.slice_dict['local_joints_positions']].reshape(50,34,3)[:,ankle_ids].copy()
        local[:,:,[0,2]]+=feat[:,None,rep.slice_dict['smooth_root_pos']][:,:,[0,2]]
        positions=frame.world(local)
        for k in plan['keyframes']:
            if k>0:
                hand_keyframes.append({'feature_hand_goal_m':np.linalg.norm(d['raw_positions_world'][k,:,0]-hand_goal,axis=-1).tolist(),
                    'fk_hand_goal_m':np.linalg.norm(d['kimodo_fk_world'][k,:,0]-hand_goal,axis=-1).tolist()})
        for i in range(1,50):
            future.append({'feature_ankle_goal_m':np.linalg.norm(positions[i]-feet,axis=-1).tolist(),
                'fk_ankle_goal_m':np.linalg.norm(d['raw_future_ankle_world'][i]-feet,axis=-1).tolist(),
                'root_goal_m':float(np.linalg.norm(d['raw_future_root_world'][i]-root))})
    result['overlapping_future_1_to_49']={k:stats(future,k) for k in future[0]}
    result['overlapping_hand_keyframes']={k:stats(hand_keyframes,k) for k in hand_keyframes[0]}
    (run/'support-aligned.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (run/'support-summary.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    print(json.dumps(summarize(p.parse_args().run),indent=2))
