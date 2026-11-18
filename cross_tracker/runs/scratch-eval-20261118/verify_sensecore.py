"""Read-only validation of the three completed full-budget scratch finals."""
from pathlib import Path
import hashlib, json, torch, time
root=Path('/data/NTU_slab/mingqianzhou/humanoid_vla/kimodo-policy-vlm-version/cross_tracker')
torch.set_num_threads(2)
rows=[]
for method in ['sonic','twist2','mix']:
    ident=f'box-aligned-{method}-scratch-s42-20261118-r1'
    run=root/'runs'/ident
    checkpoint=run/'training/checkpoint_200000'
    state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
    assert state['global_step']==200000, (method,state['global_step'])
    count=[0,0]
    def check(value):
        if isinstance(value,torch.Tensor):
            count[0]+=1; count[1]+=value.numel()
            if value.is_floating_point(): assert torch.isfinite(value).all(), method
        elif isinstance(value,dict):
            for v in value.values():check(v)
        elif isinstance(value,(tuple,list)):
            for v in value:check(v)
    check(state)
    files={}
    for p in sorted(checkpoint.iterdir()):
        if p.is_file():
            with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            files[p.name]={'bytes':p.stat().st_size,'sha256':digest}
    row=dict(method=method,run_id=ident,checkpoint=str(checkpoint),global_step=state['global_step'],state_keys=list(state),tensor_count=count[0],tensor_elements=count[1],all_tensors_finite=True,files=files)
    for name in ['launch.json','scratch-check.json','resolved-scratch-config.json','source-files.json','plan.json','submission.json']:
        row[name]=json.loads((run/name).read_text())
    rows.append(row)
print(json.dumps({'checked_unix':time.time(),'models':rows},indent=2))
