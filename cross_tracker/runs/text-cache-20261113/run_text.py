"""Generate real task embeddings only after all weights are verified and GPU 0 is idle."""
from pathlib import Path
import hashlib,json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[2]
RUN=Path(__file__).resolve().parent
record=json.loads((RUN/'run.json').read_text())
assert record['status']=='planned' and record['stage']=='diagnostic'
record.update(status='waiting_for_weights',formal_training_started=False)
(RUN/'run.json').write_text(json.dumps(record,indent=2))
deadline=time.monotonic()+3*3600
while not (ROOT/'runs/readiness-20261113-r2/weights-complete.json').exists():
 if time.monotonic()>deadline:raise TimeoutError('Verified weight import not complete')
 time.sleep(15)
verified=json.loads((ROOT/'runs/readiness-20261113-r2/weights-complete.json').read_text())
assert len(verified)==21 and all(verified.values())
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
row=[line.split(',') for line in gpu.splitlines() if line.split(',')[0].strip()=='0'][0]
assert int(row[1])<100 and int(row[2])<5,'GPU 0 is occupied; do not start encoder'
env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':str(ROOT/'data/cache/python'),
     'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4'}
command=[str(ROOT.parent/'.venv/bin/python'),'-u',str(ROOT/'scripts/prepare_text.py'),'--device','cuda']
record.update(status='running',command=command,runtime_environment={'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1'})
(RUN/'run.json').write_text(json.dumps(record,indent=2))
with (RUN/'encoder.log').open('x') as log:
 result=subprocess.run(command,cwd=ROOT.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
receipt={'exit_code':result.returncode,'gpu_inventory_before':gpu,'embedding_files':{}}
if result.returncode==0:
 import torch
 files=sorted((ROOT/'data/cache/text').glob('*.pt'))
 assert {p.stem for p in files}=={'HOI_pp_box','HSI_boxing','HSI_vision_navi'}
 for p in files:
  x=torch.load(p,map_location='cpu',weights_only=True);v=x['embedding']
  assert v.shape==(1,4096) and torch.isfinite(v).all()
  receipt['embedding_files'][p.name]={'instruction':x['instruction'],'norm':float(v.float().norm()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
record['status']='completed' if result.returncode==0 else 'failed'
(RUN/'result.json').write_text(json.dumps(receipt,indent=2))
(RUN/'run.json').write_text(json.dumps(record,indent=2))
print('TEXT_CACHE',record['status'],flush=True)
raise SystemExit(result.returncode)
