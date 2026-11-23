"""Audit held-goal reference motion and strict bilateral execution accuracy."""
import argparse,json
from pathlib import Path
import numpy as np
from kimodo_vla.core.types import G1_JOINT_NAMES
from summarize_continuous import summarize


def review(run):
    s=summarize(run)
    read=lambda f:[json.loads(x) for x in (run/f).read_text().splitlines()]
    execution,commands,plans=read('execution.jsonl'),read('commands.jsonl'),read('plans.jsonl')
    meta=json.loads((run/'run.json').read_text());mode=meta['mode'];t0=execution[0]['before']['time']
    old=Path('runs')/f'continuous-ik-{mode}-20261117-r1';oldmeta=json.loads((old/'run.json').read_text())
    for key in ('checkpoint_sha256','tracker_sha256','arena_sha256'):
        assert meta[key]==oldmeta[key],key
    oldinitial=json.loads((old/'execution.jsonl').read_text().splitlines()[0])['before']
    for key in ('root_pos','root_quat','joints','wrist_pos','foot_pos'):
        np.testing.assert_array_equal(execution[0]['before'][key],oldinitial[key])
    first=np.load(run/'plan-000.npz');original=np.load(old/'plan-000.npz')
    for key in ('motion_features','conditions','mask','raw_actions'):
        np.testing.assert_array_equal(first[key],original[key])
    baseline=meta['trajectory_projection']=='ik'
    if baseline:
        for f in run.glob('plan-*.npz'):
            a=np.load(f);b=np.load(old/f.name)
            for key in ('motion_features','conditions','mask','actions'):
                np.testing.assert_array_equal(a[key],b[key])
    lookup={round((c['time']-t0)*50):c for c in commands}
    rows=[]
    for e in execution:
        step=round((e['after']['time']-t0)*50)
        if step not in lookup:continue
        c=lookup[step];q=np.array(c['sent_joints']);previous=np.array(e['before']['reference_joints'])
        diag=plans[c['plan']]['projection']['frames'][c['action_index']]
        rows.append({'step':step,'elapsed':step/50,'q':q,'dq':q-previous,'held':diag.get('holding_target',False),
            'reused':diag.get('reused_reference',False),'actual_q':e['after']['joints'],
            'posture_wrist_m':e['posture_wrist_m'],'root_pose_wrist_contribution_m':e['root_pose_wrist_contribution_m']})
    result={'run':run.name,'projection':meta['trajectory_projection'],'mode':mode,
        'checks':{'same_initial_state_weights_tracker_arena':True,'same_first_native_prediction':True,
                  'baseline_reproduces_prior_run':baseline},'windows':s['windows'],
        'support':s['support'],'joint_motion':{}}
    for label,subset in [('all',[r for r in rows]),('after_deadline',[r for r in rows if r['elapsed']>1.2]),
                         ('steady_4_to_8s',[r for r in rows if 4<r['elapsed']<=8])]:
        if not subset:continue
        q=np.array([r['q'] for r in subset]);dq=np.array([r['dq'] for r in subset])
        result['joint_motion'][label]={'frames':len(subset),
            'reference_range_deg':dict(zip(G1_JOINT_NAMES,np.rad2deg(np.ptp(q,axis=0)).tolist())),
            'reference_max_step_deg':float(np.rad2deg(abs(dq).max())),
            'reference_max_speed_rad_s':float(abs(dq).max()*50),
            'reference_total_travel_deg':float(np.rad2deg(abs(dq).sum())),
            'reference_q_actual_rmse_rad':float(np.sqrt(np.mean((q-np.array([r['actual_q'] for r in subset]))**2))),
            'held_frames':sum(r['held'] for r in subset),'reused_reference_frames':sum(r['reused'] for r in subset),
            'posture_wrist_mean_m':np.mean([r['posture_wrist_m'] for r in subset],axis=0).tolist(),
            'root_pose_wrist_mean_m':np.mean([r['root_pose_wrist_contribution_m'] for r in subset],axis=0).tolist()}
    if not baseline:
        held=[r for r in rows if r['reused']]
        assert held
        assert max(abs(r['dq']).max() for r in held)<1e-7
        if run.name.endswith('-r2'):
            after=[r for r in rows if r['elapsed']>1.2]
            assert all(r['held'] and r['reused'] for r in after)
            assert max(result['joint_motion']['after_deadline']['reference_range_deg'].values())<1e-5
    frames=[f for p in plans for f in p['projection']['frames']]
    result['projection_audit']={'planned_frames':len(frames),'converged':sum(f['success'] for f in frames),
        'held':sum(f.get('holding_target',False) for f in frames),'reused':sum(f.get('reused_reference',False) for f in frames),
        'max_wrist_error_m':np.max([f['wrist_error_m'] for f in frames],axis=0).tolist(),
        'max_foot_error_m':np.max([f['foot_error_m'] for f in frames],axis=0).tolist()}
    (run/'ik-hold-summary.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',type=Path,nargs='+');a=p.parse_args()
    results=[review(r) for r in a.runs];out=Path('runs/ik-hold-review');out.mkdir(exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(results,indent=2))
    for r in results:print(r['run'],json.dumps(r['windows']), 'joint hold range deg',max(r['joint_motion']['after_deadline']['reference_range_deg'].values()))
