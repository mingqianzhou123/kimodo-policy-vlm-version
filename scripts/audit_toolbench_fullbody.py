"""Recompute raw training conversions before admitting the fullbody recipe."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from kimodo_vla.benchmarks.toolbench import demos
from kimodo_vla.benchmarks.toolbench.adapter import ToolBenchAdapter
from kimodo_vla.data.build import training_episode
from kimodo_vla.data.episodes import validate_episode
from kimodo_vla.model.features import KimodoFeatures

def main():
 p=argparse.ArgumentParser();p.add_argument('--toolbook',type=Path,required=True);p.add_argument('--prepared',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(1)
 root=a.toolbook;entries=json.loads((root/'manifest.json').read_text())['episodes']
 selected={}
 for e in entries:selected.setdefault(e['condition'],e)
 f=KimodoFeatures(50);counts={};checked=[]
 for task,e in selected.items():
  data=root/e['data'];rec=data.parents[2];idx=int(data.stem.split('_')[-1]);raw=demos.read_recording(rec,idx)
  demo=demos.ToolBenchDemo(raw,demos.instruction(task));demo.images=lambda steps:np.zeros((len(steps),1,8,8,3),np.uint8)
  record,images=training_episode(demo,f,task=task,episode_id='audit')
  assert record['observed_mask'].sum()==206 # heading 2 + all 34 skeleton rotations 204
  assert not any(k.startswith('current_observed_') for k in record)
  obs=demo.observations();states=demos.policy_state(demo.state,demo.previous_height)
  adapter=ToolBenchAdapter();adapter._expected_states=len(states)
  actual=adapter.to_observations({'instruction':demo.instruction,'image':{'rgb_head_stereo_left':np.zeros((360,640,3),np.uint8)},'state':{'states':states[-1:],'state_history':states,'base_pose_history':demo.base_pose,'body_q_history':demo.state[:,:29]}})
  for x,y in zip(obs,actual):
   np.testing.assert_array_equal(x.joint_pos,y.joint_pos);np.testing.assert_array_equal(x.hand,y.hand);np.testing.assert_array_equal(x.root_rot,y.root_rot)
  assert len(record['target_motion_padded'])==len(record['target_motion'])+49
  assert record['target_motion_padded'][-20:,311:413].abs().max()<1e-5
  checked.append({'condition':task,'raw_steps':len(demo),'model_frames':len(record['target_motion'])})
 if a.prepared:
  complete=json.loads((a.prepared/'preparation_complete.json').read_text());assert complete['passed'] and complete['episodes']==len(entries)
  paths=list(a.prepared.glob('*/*.pt'));assert len(paths)==len(entries)
  for path in paths:
   r=torch.load(path,map_location='cpu',weights_only=True);validate_episode(r,path.with_suffix('.frames.npy'))
   assert r['history_source']==demos.HISTORY_SOURCE and r['image_preprocessing']==demos.IMAGE_PREPROCESSING
   assert r['current_observation_source']==demos.CURRENT_OBSERVATION_SOURCE
   assert r.get('target_padding')=='pose_before_features_with_last_command_v1'
   assert r['observed_mask'].sum()==206
   assert not any(k.startswith('current_observed_') for k in r)
   assert r['target_motion_padded'][-20:,311:413].abs().max()<1e-5
   counts[r['task']]=counts.get(r['task'],0)+1
 result={'passed':True,'raw_alignment_checks':checked,'prepared_counts':counts,'prepared_episodes':sum(counts.values())}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
