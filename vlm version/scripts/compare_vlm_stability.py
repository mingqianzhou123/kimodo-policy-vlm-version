"""Paired nine-condition comparison; rejected cases remain in the denominator."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from kimodo_vla.vlm.stability import stats
from kimodo_vla.vlm.geometry import load_capture,backproject

ROOT=Path(__file__).resolve().parents[1]


def compare(baseline,refined):
    old=json.loads((baseline/'summary.json').read_text())
    new=json.loads((refined/'summary.json').read_text())
    suite=json.loads((refined/'suite.json').read_text())
    assert suite['status']=='completed'
    for name,sha in suite['visual_source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    key=lambda c:(c['reach_rad'],c['camera_lateral_m'])
    before={key(c):c for c in old['cases']};after={key(c):c for c in new['cases']}
    assert before.keys()==after.keys() and len(after)==9
    pairs=[]
    for k,b in before.items():
        n=after[k];br=ROOT/'runs'/b['run'];nr=ROOT/'runs'/n['run']
        np.testing.assert_allclose(n['truth_world'],b['truth_world'],atol=1e-6,rtol=0)
        np.testing.assert_allclose(n['camera_position'],b['camera_position'],atol=1e-6,rtol=0)
        np.testing.assert_allclose(n['initial_wrist_world'],b['initial_wrist_world'],atol=1e-6,rtol=0)
        bm=json.loads((br/'run.json').read_text());nm=json.loads((nr/'run.json').read_text())
        for field in ('checkpoint_sha256','tracker_sha256','arena_sha256','variant','handoff_seconds',
                      'joint_speed_rad_s','support_constraints','hand_constraints','trajectory_projection','reference_source'):
            assert nm[field]==bm[field],(k,field)
        for name,sha in suite['visual_source_sha256'].items():
            assert nm['source_sha256'][name]==sha,(k,name)
        with np.load(br/'vlm-capture.npz') as bc,np.load(nr/'vlm-capture.npz') as nc:
            identical_rgb=np.array_equal(bc['rgb'],nc['rgb'])
            max_rgb_diff=int(np.abs(bc['rgb'].astype(int)-nc['rgb'].astype(int)).max())
            mean_rgb_diff=float(np.abs(bc['rgb'].astype(int)-nc['rgb'].astype(int)).mean())
        old_intent=json.loads((br/'vlm-intent.json').read_text())
        new_intents=[json.loads(p.read_text()) for p in [nr/'vlm-intent.json',*sorted(nr.glob('vlm-repeat-*.json'))]]
        coarse_matches=old_intent.get('intent')==new_intents[0].get('coarse_intent')
        all_calls_identical=len({json.dumps([c['raw_output'] for c in d['calls']]) for d in new_intents})==1
        capture=load_capture(nr/'vlm-capture.npz');coarse_errors=[];coarse_rejections=[]
        for hand,side in enumerate(('left','right')):
            try:
                point=new_intents[0]['coarse_intent'][side]
                world,_=backproject(point,capture['depth'],capture['K'],capture['camera_position'],capture['camera_rotation'])
                coarse_errors.append(float(np.linalg.norm(world-np.array(n['truth_world'])[hand])*100))
                coarse_rejections.append(None)
            except (ValueError,KeyError,TypeError) as exc:
                coarse_errors.append(None);coarse_rejections.append(str(exc))
        pair={'reach_rad':k[0],'camera_lateral_m':k[1],'same_rgb':identical_rgb,'max_rgb_channel_diff':max_rgb_diff,
              'mean_rgb_channel_diff':mean_rgb_diff,'same_frame_coarse_visual_cm':coarse_errors,
              'same_frame_coarse_rejections':coarse_rejections,
              'coarse_matches_baseline':coarse_matches,'all_refinement_raw_calls_identical':all_calls_identical,
              'model_generate_calls':sum(len(d['calls']) for d in new_intents)}
        for label,c in [('baseline',b),('refined',n)]:
            first=c['intents'][0]
            pair[label]={'accepted':first['accepted'],'complete':c['complete'],'frames':c['executed_frames'],
                         'rejection':first.get('rejection'),'visual_cm':[next((h['world_error_cm'] for h in first.get('hands',[])
                             if h['hand']==side and 'world_error_cm' in h),None) for side in ('left','right')],
                         'pixel_error':first.get('pixel_error'),'inference_s':first['inference_s'],
                         'tracking_mean_cm':c.get('error_cm',{}).get('tracking',{}).get('mean'),
                         'total_mean_cm':c.get('error_cm',{}).get('total',{}).get('mean'),
                         'both_within_3cm_fraction':c.get('both_true_target_within_3cm_fraction')}
        pairs.append(pair)
    common=[p for p in pairs if p['baseline']['complete'] and p['refined']['complete']]
    grouped={}
    for group,items in [('common_executed',common),('all_conditions',pairs)]:
        grouped[group]={}
        for label in ('baseline','refined'):
            completed=[p[label] for p in items if p[label]['complete']]
            grouped[group][label]={'conditions':len(items),'completed':len(completed),
                'visual_by_hand_cm':{s:stats([p[label]['visual_cm'][i] for p in items if p[label]['visual_cm'][i] is not None])
                                    for i,s in enumerate(('left','right'))},
                'tracking_mean_cm':stats([c['tracking_mean_cm'] for c in completed]),
                'total_mean_cm':stats([c['total_mean_cm'] for c in completed]),
                'all_postarrival_frames_both_within_3cm':sum(c['both_within_3cm_fraction']==1 for c in completed),
                'inference_s':stats([p[label]['inference_s'] for p in items])}
    result={'paired_conditions':9,'common_executed_conditions':len(common),'groups':grouped,'cases':pairs,
            'same_scene_and_backend_verified':True,'refined_visual_source_verified':True,
            'same_rgb_conditions':sum(p['same_rgb'] for p in pairs),
            'coarse_matches_baseline_conditions':sum(p['coarse_matches_baseline'] for p in pairs),
            'all_raw_calls_repeat_identically_conditions':sum(p['all_refinement_raw_calls_identical'] for p in pairs),
            'model_generate_calls':sum(p['model_generate_calls'] for p in pairs),
            'same_frame_coarse_visual_by_hand_cm':{s:stats([p['same_frame_coarse_visual_cm'][i] for p in pairs
                 if p['same_frame_coarse_visual_cm'][i] is not None]) for i,s in enumerate(('left','right'))},
            'same_frame_common_valid_visual_by_hand_cm':{s:{
                 'coarse':stats([p['same_frame_coarse_visual_cm'][i] for p in pairs if p['same_frame_coarse_visual_cm'][i] is not None and p['refined']['visual_cm'][i] is not None]),
                 'refined':stats([p['refined']['visual_cm'][i] for p in pairs if p['same_frame_coarse_visual_cm'][i] is not None and p['refined']['visual_cm'][i] is not None])}
                 for i,s in enumerate(('left','right'))}}
    (refined/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path);p.add_argument('refined',type=Path)
    a=p.parse_args();compare(a.baseline,a.refined)
