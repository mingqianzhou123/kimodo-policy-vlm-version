"""Aligned fixed-reference error budget; root-pose and joint-posture vectors are separate."""
import argparse,json
from pathlib import Path
import numpy as np
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.oracle.geometry import rotation_wxyz,angle_deg
from summarize_oracle import stats


def summarize(run):
    read=lambda n:[json.loads(x) for x in (run/n).read_text().splitlines()]
    commands,executions=read('commands.jsonl'),read('execution.jsonl')
    target=json.loads((run/'target.json').read_text())
    start=executions[0]['before']['time']
    by_step={round((c['time']-start)*50):c for c in commands}
    arms=[i for i,n in enumerate(G1_JOINT_NAMES) if any(k in n for k in ('shoulder','elbow','wrist'))]
    legs=[i for i,n in enumerate(G1_JOINT_NAMES) if any(k in n for k in ('hip','knee','ankle'))]
    rows=[]
    for e in executions:
        step=round((e['after']['time']-start)*50);c=by_step[step]
        if abs(c['time']-e['after']['time'])>1e-5: raise ValueError('Reference/actual timestamps differ')
        actual=np.array(e['after']['wrist_pos']);ref=np.array(e['reference_wrist_world'])
        planned=np.array(c['wrist_pos']);goal=np.array(target['targets_world'])[:,0]
        q=np.array(e['after']['joints']);refq=np.array(e['reference_joints'])
        decoder=np.array(e['decoder_joint_target'])
        row={k:e[k] for k in ('posture_wrist_m','root_pose_wrist_contribution_m',
            'decoder_actual_joint_rmse_rad','reference_actual_joint_rmse_rad','measured_xml_vs_sim_wrist_m')}
        row.update(step=step,elapsed=step/50,
            prescribed_target_m=np.linalg.norm(planned-goal,axis=-1).tolist(),
            transport_wrist_m=np.linalg.norm(ref-planned,axis=-1).tolist(),
            tracking_wrist_m=np.linalg.norm(actual-ref,axis=-1).tolist(),
            actual_target_m=np.linalg.norm(actual-goal,axis=-1).tolist(),
            root_position_error_m=float(np.linalg.norm(np.array(e['after']['root_pos'])-c['root_pos'])),
            root_rotation_error_deg=angle_deg(rotation_wxyz(e['after']['root_quat']),np.array(c['root_rotation'])),
            reference_arm_rmse_rad=float(np.sqrt(np.mean((q[arms]-refq[arms])**2))),
            reference_leg_rmse_rad=float(np.sqrt(np.mean((q[legs]-refq[legs])**2))),
            decoder_arm_rmse_rad=float(np.sqrt(np.mean((q[arms]-decoder[arms])**2))),
            decoder_leg_rmse_rad=float(np.sqrt(np.mean((q[legs]-decoder[legs])**2))))
        # Exact vector identity; the norms alone do not add.
        rooted=np.array(e['root_anchored_reference_wrist_world'])
        np.testing.assert_allclose(actual-ref,(actual-rooted)+(rooted-ref),atol=1e-10)
        rows.append(row)
    keys=[k for k in rows[0] if k not in ('step','elapsed')]
    result={'run':run.name,'source':'fixed','mode':target['mode'],'steps':len(rows),
        'generator_loaded':False,'units':{'position':'m','joint':'rad','root_rotation':'deg'},
        'goal_displacement_m':target['goal_displacement_m']}
    for name,subset in [('all',rows),('matched_1p2_to_4s',[r for r in rows if 1.2<r['elapsed']<=4.]),
                        ('steady_4_to_8s',[r for r in rows if 4.<r['elapsed']<=8.])]:
        result[name]={k:stats(subset,k) for k in keys}
        result[name]['both_wrists_within_3cm_fraction']=float(np.mean([
            max(r['actual_target_m'])<=.03 for r in subset])) if subset else None
    (run/'fixed-aligned.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    (run/'fixed-summary.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    print(json.dumps(summarize(p.parse_args().run),indent=2))
