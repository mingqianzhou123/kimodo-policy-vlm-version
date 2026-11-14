from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,subprocess
r=Path(__file__).resolve().parent/'public-file-lists';r.mkdir(exist_ok=True)
urls={'github-main':'https://api.github.com/repos/William-wAng618/HumanoidArena/git/trees/main?recursive=1',
      'hf-raw':'https://huggingface.co/api/datasets/WilliamWang16/HumanoidArena_raw',
      'hf-v3_1':'https://huggingface.co/api/datasets/WilliamWang16/HumanoidArena_dataset_v3_1',
      'hf-models':'https://huggingface.co/api/models/WilliamWang16/HumanoidArena_models'}
def get(item):
 name,url=item;p=r/(name+'.json')
 subprocess.run(['curl','-fsSL','--retry','2','--connect-timeout','15','--max-time','90','-o',str(p),url],check=True)
 m=json.loads(p.read_text());paths=[x.get('path',x.get('rfilename')) for x in m.get('tree',m.get('siblings',[]))]
 return name,{'source':url,'revision':m.get('sha'),'entries':len(paths),'truncated':m.get('truncated',False),
              'weight_matches':[x for x in paths if x and (x.endswith('.onnx') or 'xua2csee' in x)],
              'checkpoint_paths':[x for x in paths if x and '/checkpoints/' in x][:30]}
result=dict(ThreadPoolExecutor(4).map(get,urls.items()))
(r.parent/'public-weight-search.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
