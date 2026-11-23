"""Replay one identical saved condition/noise to audit native CFG independently of physics."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from kimodo_vla.model.native_policy import NativeKimodoPolicy
from kimodo_vla.model.planner import _episode_noise_seed
from kimodo_vla.oracle.geometry import Frame

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path);p.add_argument('--checkpoint',required=True)
a=p.parse_args()
data=np.load(a.run/'plan-000.npz')
plan=json.loads((a.run/'plans.jsonl').read_text().splitlines()[0])
target=json.loads((a.run/'target.json').read_text())
frame=Frame(target['origin'],np.array(target['heading']))
goals=np.array(target['targets_world'])[:,0]
model=NativeKimodoPolicy(a.checkpoint,constraint_cfg=2).cuda().eval()
ids=[model.representation.skeleton.bone_order_names.index(f'{s}_wrist_yaw_skel') for s in ('left','right')]
rows=[]
for noisy in (False,True):
    for cfg in (1.,2.):
        model.native_noisy_history=noisy
        model.denoiser.weight=cfg
        start=time.perf_counter()
        out=model.predict_future('oracle',torch.zeros(1,3,1,1),torch.from_numpy(data['history']),
            history_feature_mask=torch.ones_like(torch.from_numpy(data['history']),dtype=torch.bool),
            text_feat=torch.zeros(1,4096,device='cuda'),
            future_condition=torch.from_numpy(data['conditions']),
            future_condition_mask=torch.from_numpy(data['mask']),diffusion_steps=50,
            generator=torch.Generator(device='cuda').manual_seed(_episode_noise_seed(1000,0)))
        pos=frame.world(out['posed_joints'][plan['keyframes'][0],ids].cpu().numpy())
        rows.append({'native_noisy_history':noisy,'constraint_cfg':cfg,'diffusion_steps':50,
            'wrist_fk_error_m':np.linalg.norm(pos-goals,axis=-1).tolist(),'seconds':time.perf_counter()-start})
print(json.dumps(rows,indent=2));(a.run/'sampler-audit.json').write_text(json.dumps(rows,indent=2))
