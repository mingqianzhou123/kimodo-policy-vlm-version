"""Import verified public simulation assets without modifying the shared checkout."""
from pathlib import Path
import hashlib,json,shutil,time,zipfile
ROOT=Path(__file__).resolve().parents[2]
RUN=Path(__file__).resolve().parent
meta=json.loads((RUN/'scene-parts.json').read_text())
store=ROOT/'data/cache/setup/sim-assets';store.mkdir(parents=True,exist_ok=True)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def wait(p,size,digest):
 deadline=time.monotonic()+3*3600
 while not p.exists() or p.stat().st_size!=size:
  if time.monotonic()>deadline:raise TimeoutError(p)
  time.sleep(10)
 assert sha(p)==digest,p
for item in meta['parts']:wait(ROOT/item['name'],item['bytes'],item['sha256'])
archive=store/'humanoidarena-assets.zip'
with archive.open('xb') as out:
 for item in meta['parts']:
  with (ROOT/item['name']).open('rb') as f:shutil.copyfileobj(f,out,16*1024**2)
wait(archive,meta['bytes'],meta['sha256'])
scene=ROOT/'data/cache/sim-assets/arena/assets';scene.mkdir(parents=True,exist_ok=False)
with zipfile.ZipFile(archive) as z:
 for name in z.namelist():assert (scene/name).resolve().is_relative_to(scene.resolve()),name
 z.extractall(scene)
for item in meta['parts']:(ROOT/item['name']).rename(store/item['name'])
sonic=ROOT/'sonic-weights.zip'
wait(sonic,91004593,'1c3fbdfc8c575366f1795e45aead003fa8ac61e83ee25155ac735cc7ea881cfb')
with zipfile.ZipFile(sonic) as z:z.extractall(ROOT/'data/cache/sim-assets')
for name,item in json.loads((ROOT/'data/cache/sim-assets/sonic/manifest.json').read_text())['files'].items():
 wait(ROOT/'data/cache/sim-assets/sonic'/name,item['bytes'],item['sha256'])
sonic.rename(store/sonic.name)
receipt={'scene_archive_sha256':meta['sha256'],'scene_files':len(list(scene.rglob('*'))),
         'robots_present':(scene/'robots').is_dir(),'objects_present':(scene/'objects').is_dir(),
         'sonic_verified':True,'mimic_lite_verified':False,'simulation_executed':False}
(RUN/'assets-import.json').write_text(json.dumps(receipt,indent=2))
print('SIM_ASSETS_IMPORTED',json.dumps(receipt),flush=True)
