"""Evaluator-only error accounting. Never imported by the execution policy."""
import hashlib
import json
from pathlib import Path
import numpy as np
from kimodo_vla.vlm.geometry import backproject,load_capture


def stats(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return None
    return {'n':len(a),'mean':a.mean(0).tolist(),'median':np.median(a,axis=0).tolist(),'p95':np.percentile(a,95,axis=0).tolist(),
            'max':a.max(0).tolist()}


def decompose(actual,reference,visual,truth):
    """Signed vectors add exactly; Euclidean error magnitudes generally do not."""
    a,r,v,t=map(np.asarray,(actual,reference,visual,truth))
    components={'visual':v-t,'reference':r-v,'tracking':a-r,'total':a-t}
    residual=components['total']-components['visual']-components['reference']-components['tracking']
    assert np.max(np.abs(residual))<1e-9
    return {k:stats(np.linalg.norm(x,axis=-1)*100) for k,x in components.items() if k!='visual'}


def inspect_intent(path,capture,truth,initial):
    result=json.loads(path.read_text())
    if result['capture_sha256']!=hashlib.sha256(capture.read_bytes()).hexdigest():
        raise ValueError('Intent/capture hash mismatch')
    data=load_capture(capture)
    optical=(truth-data['camera_position'])@data['camera_rotation']
    uv=optical@data['K'].T;uv=uv[:,:2]/uv[:,2:]
    h,w=data['depth'].shape
    out={'file':path.name,'load_s':result['load_s'],'inference_s':result['inference_s'],
         'truth_pixels':uv.tolist(),'truth_in_image':((optical[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)).tolist(),
         'valid_protocol':'intent' in result or 'candidate_intent' in result,'accepted':False,'raw_output':result['raw_output']}
    # Evaluator-only calibration/visibility check at the TRUE projected center.
    # This is never substituted for the model's pixel or goal.
    out['truth_center_depth_check']=[]
    for hand,point in enumerate(uv/np.array([w-1,h-1])*1000):
        try:
            calibrated,info=backproject(point.tolist(),data['depth'],data['K'],data['camera_position'],data['camera_rotation'])
            out['truth_center_depth_check'].append({'error_cm':float(np.linalg.norm(calibrated-truth[hand])*100),**info})
        except ValueError as exc:out['truth_center_depth_check'].append({'error':str(exc)})
    if 'intent' not in result and 'candidate_intent' not in result:
        out['rejection']=result.get('error','Invalid protocol');return out
    intent=result.get('intent',result.get('candidate_intent'))
    quality_rejection=result.get('error') if 'intent' not in result else None
    points=np.array([intent[side] for side in ('left','right')])
    pixels=points/1000*np.array([w-1,h-1]);out.update(pixels=pixels.tolist(),pixel_error=np.linalg.norm(pixels-uv,axis=-1).tolist())
    world=[];hands=[]
    for hand,point in enumerate(points):
        detail={'hand':('left','right')[hand],'pixel_error':float(np.linalg.norm(pixels[hand]-uv[hand]))}
        try:
            value,depth_info=backproject(point.tolist(),data['depth'],data['K'],data['camera_position'],data['camera_rotation'])
            world.append(value);detail.update(world=value.tolist(),depth_info=depth_info,
                world_error_cm=float(np.linalg.norm(value-truth[hand])*100))
        except ValueError as exc:detail['rejection']=str(exc)
        hands.append(detail)
    out['hands']=hands
    if any('rejection' in hand for hand in hands):
        out['rejection']='; '.join(hand['hand']+': '+hand['rejection'] for hand in hands if 'rejection' in hand)
        return out
    world=np.array(world)
    out.update(world=world.tolist(),world_error_cm=(np.linalg.norm(world-truth,axis=-1)*100).tolist(),
               world_signed_error_cm=((world-truth)*100).tolist())
    # Match the unchanged policy's workspace gate. Never execute evaluator truth.
    if np.any(np.linalg.norm(world-initial,axis=-1)>.15):out['rejection']='Visual target exceeds 15 cm workspace'
    elif quality_rejection:out['rejection']=quality_rejection
    else:out['accepted']=True
    return out


def review_case(run):
    meta=json.loads((run/'run.json').read_text())
    out={'run':run.name,'status':meta['status'],'error':meta.get('error'),
         'reach_rad':meta['marker_reach_rad'],'camera_lateral_m':meta['camera_lateral_m'],
         'intents':[],'executed_frames':0,'complete':False}
    if not (run/'vlm-capture.npz').exists():return out
    data=load_capture(run/'vlm-capture.npz')
    fixture=json.loads((run/'visual-evaluation-truth.json').read_text())
    truth=np.array(fixture['targets_world'])
    target=json.loads((run/'target.json').read_text())
    # Oracle initialize always records the starting wrist goals before VLM failure.
    if (run/'execution.jsonl').exists():
        execution=[json.loads(x) for x in (run/'execution.jsonl').read_text().splitlines()]
    else:execution=[]
    initial=np.array(target['initial_wrist_world']) if 'initial_wrist_world' in target else None
    if initial is None:
        # Separate fixture initial wrists are measurement-only evaluator metadata.
        initial=np.array(fixture['initial_wrist_world'])
    if not np.allclose(data['camera_position'],fixture['expected_camera_position'],atol=1e-5):
        raise ValueError('Camera did not reach its requested view')
    out.update(truth_world=truth.tolist(),camera_position=data['camera_position'].tolist(),
        initial_wrist_world=initial.tolist(),goal_displacement_cm=(np.linalg.norm(truth-initial,axis=-1)*100).tolist())
    for path in [run/'vlm-intent.json',*sorted(run.glob('vlm-repeat-*.json'))]:
        if path.exists():out['intents'].append(inspect_intent(path,run/'vlm-capture.npz',truth,initial))
    valid=[i for i in out['intents'] if 'world' in i]
    if valid:
        out['repeat_world_span_cm']=(np.ptp([i['world'] for i in valid],axis=0)*100).tolist()
    pixels=[i['pixels'] for i in out['intents'] if 'pixels' in i]
    if pixels:out['repeat_pixel_span']=np.ptp(pixels,axis=0).tolist()
    out['repeat_raw_outputs_identical']=bool(out['intents']) and len({i['raw_output'] for i in out['intents']})==1
    out['executed_frames']=len(execution)
    requested=json.loads((run/'episodes.json').read_text())['episodes'][0]['max_steps']
    out['complete']=meta['status']=='completed' and len(execution)==requested
    if execution:
        assert all(e['after']['gripper_command']==[0,0] for e in execution)
        assert np.allclose(np.diff([e['after']['time'] for e in execution]),.02,atol=1e-6)
    post=[e for e in execution if e['after']['time']>target['arrival_time']+1e-6]
    if post:
        actual=np.array([e['after']['wrist_pos'] for e in post]);ref=np.array([e['reference_wrist_world'] for e in post])
        goal=np.array(target['targets_world'])[:,0]
        out['postarrival_frames']=len(post)
        out['error_cm']=decompose(actual,ref,goal,truth)
        out['both_true_target_within_3cm_fraction']=float(np.all(np.linalg.norm(actual-truth,axis=-1)<=.03,axis=-1).mean())
        out['reference_range_deg']=float(np.rad2deg(np.ptp([e['reference_joints'] for e in post],axis=0)).max())
        out['root_translation_cm']=stats([np.linalg.norm(np.array(e['after']['root_pos'])-e['reference_root_world'])*100 for e in post])
    return out


def review_suite(folder):
    suite=json.loads((folder/'suite.json').read_text())
    cases=[]
    for spec in suite['cases']:
        run=folder.parent/spec['run_id']
        if (run/'run.json').exists():cases.append(review_case(run))
    first=[c['intents'][0] for c in cases if c['intents']]
    completed=[c for c in cases if c['complete']]
    result={'suite':folder.name,'requested_cases':len(suite['cases']),'observed_cases':len(cases),
        'completed_cases':len(completed),'first_intent_accepted':sum(x['accepted'] for x in first),
        'first_intent_count':len(first),'all_intent_count':sum(len(c['intents']) for c in cases),
        'visual_first_world_error_cm':stats([x['world_error_cm'] for x in first if 'world_error_cm' in x]),
        'accepted_visual_world_error_cm':stats([x['world_error_cm'] for x in first if x['accepted']]),
        'visual_first_pixel_error':stats([x['pixel_error'] for x in first if 'pixel_error' in x]),
        'visual_by_hand_cm':{side:stats([h['world_error_cm'] for x in first for h in x.get('hands',[])
            if h['hand']==side and 'world_error_cm' in h]) for side in ('left','right')},
        'episode_mean_tracking_cm':stats([c['error_cm']['tracking']['mean'] for c in completed]),
        'episode_mean_total_cm':stats([c['error_cm']['total']['mean'] for c in completed]),
        'independent_conditions':len(cases),'repeated_same_images_are_not_independent_samples':True,
        'cases':cases}
    (folder/'summary.json').write_text(json.dumps(result,indent=2))
    return result
