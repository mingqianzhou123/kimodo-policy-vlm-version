"""Separate visual localization, projection and physical execution errors."""
import argparse
import json
from pathlib import Path
import numpy as np
from kimodo_vla.vlm.geometry import load_capture


def summary(error):
    error=np.asarray(error)
    return {'frames':len(error),'mean_cm':(error.mean(0)*100).tolist(),
        'max_cm':(error.max(0)*100).tolist(),
        'both_within_3cm_fraction':float(np.all(error<=.03,axis=-1).mean())}


def review(run):
    meta=json.loads((run/'run.json').read_text())
    if meta['status']!='completed':raise ValueError('Review requires a completed run')
    target=json.loads((run/'target.json').read_text())
    truth=np.array(json.loads((run/'visual-evaluation-truth.json').read_text())['targets_world'])
    vision=load_capture(run/'vlm-capture.npz')
    intent=json.loads((run/'vlm-intent.json').read_text())
    goal=np.array(target['targets_world'])[:,0]
    optical=(truth-vision['camera_position'])@vision['camera_rotation']
    projected=optical@vision['K'].T;projected=projected[:,:2]/projected[:,2:]
    h,w=vision['depth'].shape
    predicted=np.array([intent['intent'][s] for s in ('left','right')])/1000*np.array([w-1,h-1])
    result={'run':run.name,'camera':str(vision['camera_name']),
        'visual_world_error_cm':(np.linalg.norm(goal-truth,axis=-1)*100).tolist(),
        'visual_pixel_error':np.linalg.norm(predicted-projected,axis=-1).tolist(),
        'predicted_pixels':predicted.tolist(),'truth_projected_pixels':projected.tolist(),
        'inference_s':intent['inference_s'],'load_s':intent['load_s'],
        'motion_text':intent['intent']['motion_text'],'text_conditioning':target['text_conditioning'],
        'grasp_attempted':False,'fixture_not_object_grasp':True}
    commands=[json.loads(x) for x in (run/'commands.jsonl').read_text().splitlines()]
    commands=[x for x in commands if x['time']>target['arrival_time']+1e-6]
    # Commands include the predicted tail; restrict to frames actually executed.
    execution=[json.loads(x) for x in (run/'execution.jsonl').read_text().splitlines()]
    requested=json.loads((run/'episodes.json').read_text())['episodes'][0]['max_steps']
    result.update(execution_frames=len(execution),requested_frames=requested,
        completed_full_episode=len(execution)==requested)
    end=execution[-1]['after']['time'];commands=[x for x in commands if x['time']<=end+1e-6]
    for key in ('raw_projected_wrist','pre_ik_wrist','sent_wrist'):
        if commands:result[key+'_vs_visual_goal']=summary([np.linalg.norm(np.array(x[key])-goal,axis=-1) for x in commands])
    execution=[x for x in execution if x['after']['time']>target['arrival_time']+1e-6]
    if execution:
        actual=np.array([x['after']['wrist_pos'] for x in execution])
        refs=np.array([x['reference_wrist_world'] for x in execution])
        result['actual_vs_visual_goal']=summary(np.linalg.norm(actual-goal,axis=-1))
        result['actual_vs_true_marker']=summary(np.linalg.norm(actual-truth,axis=-1))
        result['tracking_vs_reference']=summary(np.linalg.norm(actual-refs,axis=-1))
        result['reference_range_deg']=float(np.rad2deg(np.ptp([x['reference_joints'] for x in execution],axis=0)).max())
    (run/'vlm-summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    review(parser.parse_args().run)
