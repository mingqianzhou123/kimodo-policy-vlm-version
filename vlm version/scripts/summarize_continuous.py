"""Audit actual execution coverage and strict bilateral 3 cm acceptance."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from kimodo_vla.oracle.policy import OraclePolicy
from kimodo_vla.oracle.geometry import Frame,rotation_wxyz
from kimodo_vla.oracle.hand_path import execution_frames
from kimodo_vla.model.features import KimodoFeatures
from kimodo_vla.model.sparse_conditions import hand_keyframe
from summarize_support import summarize as support_summary


def summarize(run):
    support=support_summary(run)
    read=lambda f:[json.loads(x) for x in (run/f).read_text().splitlines()]
    meta=json.loads((run/'run.json').read_text());target=json.loads((run/'target.json').read_text())
    sample=read('execution.jsonl')[0]['before']
    p=OraclePolicy.__new__(OraclePolicy);p.features=KimodoFeatures(50)
    p.frame=Frame(target['origin'],np.array(target['heading']));p.initial_root=np.array(sample['root_pos'])
    p.initial_rotation=rotation_wxyz(sample['root_quat']);p.initial_joints=np.array(sample['joints'])
    p.goal_joints=p.initial_joints if target['witness_joint_pos'] is None else np.array(target['witness_joint_pos'])
    rep=p.features.representation;names=rep.skeleton.bone_order_names
    p.ids=[[names.index(f'{s}_{j}_skel') for j in ('wrist_yaw','hand_roll')] for s in ('left','right')]
    checks=[]
    for plan in read('plans.jsonl'):
        if meta.get('hand_constraints')!='continuous':continue
        frames=execution_frames(round((plan['time']-sample['time'])*50),plan['cut'])
        assert plan['dense_hand_frames']==frames
        pos,rot=p.path_goals((plan['cut']+np.array(frames))/30)
        d=np.load(run/f"plan-{plan['plan']:03d}.npz")
        for k,pp,rr in zip(frames,pos,rot):
            v,m=hand_keyframe(rep,frames=50,frame=k,smooth_root_xz=p.frame.points(p.initial_root)[[0,2]],
                hand_positions=dict(zip(('left','right'),pp)),wrist_rotations=dict(zip(('left','right'),rr)))
            assert d['mask'][m.numpy()].all()
            np.testing.assert_allclose(d['conditions'][m.numpy()],v.numpy()[m.numpy()],atol=2e-6)
        checks.append({'plan':plan['plan'],'frames':frames,'verified':True})
    rows=read('continuity-aligned.jsonl');end=rows[-1]['elapsed']
    windows={}
    for label,subset in [('matched_1p2_to_4s',[r for r in rows if 1.2<r['elapsed']<=4]),
                         ('all_after_deadline',[r for r in rows if r['elapsed']>1.2]),
                         ('steady_4_to_8s',[r for r in rows if 4<r['elapsed']<=8])]:
        if not subset:continue
        error=np.array([r['actual_target_m'] for r in subset]);passed=np.all(error<=.03,axis=1)
        windows[label]={'frames':len(error),'both_within_3cm_fraction':float(passed.mean()),
            'stable_both_3cm':bool(passed.all()),'actual_mean_cm':(error.mean(0)*100).tolist(),
            'actual_p95_cm':(np.percentile(error,95,axis=0)*100).tolist(),'actual_max_cm':(error.max(0)*100).tolist()}
    result={'run':run.name,'hand_constraints':meta.get('hand_constraints','keyframes'),
        'trajectory_projection':meta.get('trajectory_projection','none'),'end_s':end,
        'criterion':'every executed frame after 1.2 s has BOTH actual world wrists <= 0.03 m',
        'windows':windows,'continuous_condition_audit':checks,'support':support}
    (run/'continuous-summary.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',nargs='+',type=Path)
    for run in p.parse_args().runs:
        r=summarize(run);print(run.name,json.dumps(r['windows']))
