"""Time-aligned raw generation, handoff, joint repair and execution errors."""
import argparse,json
from pathlib import Path
import numpy as np
from kimodo_vla.model.features import G1_XML_PATH
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.oracle.geometry import XmlFK
from summarize_oracle import stats


def summarize(run):
    read=lambda f:[json.loads(x) for x in (run/f).read_text().splitlines()]
    commands,execution=read('commands.jsonl'),read('execution.jsonl')
    metadata=json.loads((run/'run.json').read_text())
    target=json.loads((run/'target.json').read_text());goal=np.array(target['targets_world'])[:,0]
    limits=XmlFK(G1_XML_PATH,G1_JOINT_NAMES).joint_limits
    start=execution[0]['before']['time'];lookup={round((c['time']-start)*50):c for c in commands}
    rows=[]
    for e in execution:
        step=round((e['after']['time']-start)*50)
        if step not in lookup:
            if step>5:raise ValueError(f'No command for step {step}')
            continue  # explicit initial 0.1 s warmup
        c=lookup[step]
        assert abs(c['time']-e['after']['time'])<1e-5
        row={'step':step,'elapsed':step/50,'plan':c['plan'],'action_index':c['action_index']}
        for name in ('raw_model','raw_projected','handoff','sent'):
            row[f'{name}_target_m']=np.linalg.norm(np.array(c[f'{name}_wrist'])-goal,axis=-1).tolist()
        ref=np.array(e['reference_wrist_world']);actual=np.array(e['after']['wrist_pos'])
        q=np.array(c['sent_joints']);prev=np.array(e['before']['reference_joints'])
        row.update(transport_m=np.linalg.norm(ref-c['sent_wrist'],axis=-1).tolist(),
            tracking_m=np.linalg.norm(actual-ref,axis=-1).tolist(),
            actual_target_m=np.linalg.norm(actual-goal,axis=-1).tolist(),
            handoff_change_m=np.linalg.norm(np.array(c['handoff_wrist'])-c['raw_projected_wrist'],axis=-1).tolist(),
            bounded_change_m=np.linalg.norm(np.array(c.get('pre_ik_wrist',c['sent_wrist']))-c['handoff_wrist'],axis=-1).tolist(),
            pre_ik_target_m=np.linalg.norm(np.array(c.get('pre_ik_wrist',c['sent_wrist']))-goal,axis=-1).tolist(),
            projection_change_m=np.linalg.norm(np.array(c['sent_wrist'])-c.get('pre_ik_wrist',c['sent_wrist']),axis=-1).tolist(),
            sent_joint_step_deg=float(np.rad2deg(abs(q-prev).max())),
            joint_limit_violation=bool(np.any((q<limits[:,0]-1e-5)|(q>limits[:,1]+1e-5))),
            joint_transport_max_rad=float(abs(q-np.array(e['reference_joints'])).max()),
            root_position_error_m=float(np.linalg.norm(np.array(e['after']['root_pos'])-c['sent_root_world'])),
            posture_wrist_m=e['posture_wrist_m'])
        rows.append(row)
    result={'run':run.name,'variant':metadata['variant'],'mode':metadata['mode'],
        'sim_steps':len(execution),'aligned_steps':len(rows),'warmup_steps':len(execution)-len(rows),
        'units':'metres except angles in degrees; error norms do not add',
        'grasp_attempted':False,'joint_speed_rad_s':metadata['joint_speed_rad_s']}
    keys=[k for k in rows[0] if k not in ('step','elapsed','plan','action_index','joint_limit_violation')]
    for name,subset in [('all',rows),('matched_1p2_to_4s',[r for r in rows if 1.2<r['elapsed']<=4]),
                        ('steady_4_to_8s',[r for r in rows if 4<r['elapsed']<=8])]:
        result[name]={k:stats(subset,k) for k in keys}
        result[name]['joint_limit_frames']=sum(r['joint_limit_violation'] for r in subset)
        result[name]['both_wrists_within_3cm_fraction']=float(np.mean([max(r['actual_target_m'])<=.03 for r in subset])) if subset else None
    result['seams']=[{'step':r['step'],'plan':r['plan'],'max_joint_step_deg':r['sent_joint_step_deg']}
                     for r in rows if r['action_index']==0]
    if metadata['variant']=='bounded':
        assert result['all']['joint_limit_frames']==0
        assert result['all']['sent_joint_step_deg']['max']<=np.rad2deg(metadata['joint_speed_rad_s']/50)+1e-4
    (run/'continuity-aligned.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (run/'continuity-summary.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    print(json.dumps(summarize(p.parse_args().run),indent=2))
