from pathlib import Path
import hashlib,json,shutil,time
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/readiness-20261113-r2'
meta=json.loads((RUN/'remaining-text.json').read_text())
full=json.loads((RUN/'weight-upload-bundles.json').read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for name,item in meta.items():
 p=ROOT/name
 assert p.stat().st_size==item['prefix_bytes'] and sha(p)==item['prefix_sha256'],name
 for chunk in item['chunks']:
  q=ROOT/chunk['name'];deadline=time.monotonic()+3*3600
  while not q.exists() or q.stat().st_size!=chunk['bytes']:
   if time.monotonic()>deadline:raise TimeoutError(q)
   time.sleep(10)
  assert sha(q)==chunk['sha256'],q
 target=p.with_suffix('.zip.assembling')
 with target.open('xb') as out:
  for source in [p,*[ROOT / c['name'] for c in item['chunks']]]:
   with source.open('rb') as inp:shutil.copyfileobj(inp,out,16*1024**2)
 assert target.stat().st_size==full[name]['bytes'] and sha(target)==full[name]['sha256'],name
 target.replace(p)
 saved=ROOT/'data/cache/setup/remaining-text';saved.mkdir(parents=True,exist_ok=True)
 for chunk in item['chunks']:(ROOT/chunk['name']).rename(saved/chunk['name'])
 print('RESTORED',name,flush=True)
print('ALL_UPLOADS_RESTORED',flush=True)
