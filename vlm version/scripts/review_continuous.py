"""Build compact comparison and verify controls from saved experimental records."""
import json
from pathlib import Path
import numpy as np
from kimodo_vla.core.types import G1_JOINT_NAMES
from summarize_continuous import summarize


def review(runs):
    summaries=[summarize(p) for p in runs]
    read=lambda p,f:json.loads((p/f).read_text())
    initial=lambda p:json.loads((p/'execution.jsonl').read_text().splitlines()[0])['before']
    anchor=runs[0];meta=read(anchor,'run.json');pose=initial(anchor)
    checks={};comparison=[]
    for p,s in zip(runs,summaries):
        m=read(p,'run.json');a=initial(p)
        for field in ('checkpoint_sha256','tracker_sha256','arena_sha256'):assert m[field]==meta[field],(p,field)
        for field in ('root_pos','root_quat','joints','wrist_pos','foot_pos'):np.testing.assert_array_equal(a[field],pose[field])
        row={'run':p.name,'mode':m['mode'],'hand_constraints':m['hand_constraints'],
             'projection':m.get('trajectory_projection','none'),'windows':s['windows']}
        metrics=s['support']['hand_and_joint_metrics']
        for interval in ('matched_1p2_to_4s','steady_4_to_8s'):
            if interval not in s['windows']:continue
            h=metrics[interval]
            row[interval]={k:h[k] for k in ('raw_model_target_m','pre_ik_target_m','sent_target_m','transport_m',
                'tracking_m','actual_target_m','projection_change_m','root_position_error_m','sent_joint_step_deg','joint_limit_frames')}
        row['support_1p2_to_4s']=s['support']['matched_1p2_to_4s']
        if m.get('trajectory_projection')=='ik':
            dense=Path('runs')/f"continuous-dense-{m['mode']}-20261117-r1"
            a0=np.load(p/'plan-000.npz');b0=np.load(dense/'plan-000.npz')
            for field in ('motion_features','conditions','mask','raw_actions'):
                np.testing.assert_array_equal(a0[field],b0[field])
            checks[p.name]={'first_native_prediction_identical_to_dense':True}
            plans=[json.loads(x) for x in (p/'plans.jsonl').read_text().splitlines()]
            frames=[f for plan in plans for f in plan['projection']['frames']]
            row['ik']={'plans':len(plans),'frames':len(frames),'converged':sum(x['success'] for x in frames),
                'max_wrist_error_m':np.max([x['wrist_error_m'] for x in frames],axis=0).tolist(),
                'max_foot_error_m':np.max([x['foot_error_m'] for x in frames],axis=0).tolist(),
                'mean_projection_s':float(np.mean([x['projection']['elapsed_s'] for x in plans])),
                'max_projection_s':max(x['projection']['elapsed_s'] for x in plans),
                'mean_native_inference_s':float(np.mean([x['inference_s'] for x in plans])),
                'mean_prior_joint_rmse_rad':float(np.mean([x['prior_joint_rmse_rad'] for x in frames]))}
            execution=[json.loads(x) for x in (p/'execution.jsonl').read_text().splitlines()]
            t0=execution[0]['before']['time']
            matched=[e for e in execution if 60<round((e['after']['time']-t0)*50)<=200]
            ranges=np.rad2deg(np.ptp([e['reference_joints'] for e in matched],axis=0))
            row['ik']['post_deadline_joint_range_deg']=dict(zip(G1_JOINT_NAMES,ranges.tolist()))
            row['ik']['posture_wrist_mean_m']=np.mean([e['posture_wrist_m'] for e in matched],axis=0).tolist()
            row['ik']['root_pose_wrist_mean_m']=np.mean([e['root_pose_wrist_contribution_m'] for e in matched],axis=0).tolist()
        if m['hand_constraints']=='keyframes':
            old=Path('runs')/f"support-root-{m['mode']}-20261117-r1"
            diffs={field:max(float(np.max(abs(np.load(f)[field]-np.load(old/f.name)[field]))) for f in p.glob('plan-*.npz'))
                   for field in ('motion_features','conditions','actions')}
            for f in p.glob('plan-*.npz'):
                np.testing.assert_array_equal(np.load(f)['mask'],np.load(old/f.name)['mask'])
            assert max(diffs.values())==0
            checks[p.name]={'previous_baseline_max_differences':diffs}
        comparison.append(row)
    checks['same_checkpoint_tracker_arena']=True;checks['same_initial_pose_and_targets_source']=True
    out=Path('runs/continuous-review');out.mkdir(exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(comparison,indent=2))
    (out/'validation.json').write_text(json.dumps(checks,indent=2))
    return comparison

if __name__=='__main__':
    result=review(sorted(Path('runs').glob('continuous-*-20261117-r1')))
    for row in result:print(row['run'],row['windows']['matched_1p2_to_4s'],row.get('ik',''))
