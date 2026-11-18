"""Keep the failed R1 intact; include non-Python skeleton assets in R2."""
from pathlib import Path
import hashlib,json,shutil,time
root=Path('/root/workspace/humanoidVLA')
old=root/'cross_tracker/runs/scratch-eval-20261118'
run=root/'cross_tracker/runs/scratch-eval-20261118-r2'
state=json.loads((old/'batch.json').read_text())
assert state['status']!='running',state['status']
run.mkdir(exist_ok=False)
code=run/'code'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
sources=json.loads((old/'source-files.json').read_text())
for name,digest in sources.items():
    src=old/'code'/name;dst=code/name
    assert sha(src)==digest
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for src in (root/'kimodo_vla/skeleton/assets').rglob('*'):
    if src.is_file() and src.suffix!='.pyc':
        name=str(src.relative_to(root));dst=code/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert sha(src)==sha(dst)
        else:shutil.copy2(src,dst)
        sources[name]=sha(dst)
(code/'cross_tracker/data').mkdir()
(code/'cross_tracker/data/cache').symlink_to(root/'cross_tracker/data/cache',target_is_directory=True)
(code/'checkpoints').symlink_to((root/'checkpoints').resolve(),target_is_directory=True)
for name in ['sonic-manifest.json','twist2-manifest.json','mix-manifest.json','local-verification.json','shared-eval-assets.json']:
    shutil.copy2(old/name,run/name)
(run/'run_batch.py').write_text((old/'run_batch.py').read_text().replace('-20261118-r1','-20261118-r2'))
(run/'source-files.json').write_text(json.dumps(sources,indent=2))
receipt=json.loads((old/'prepare.json').read_text())
receipt.update(code=str(code),source_files=len(sources),finished_unix=time.time(),fix='Include skeleton .p and XML assets; no policy/protocol change',previous_failed_batch=str(old))
(run/'prepare.json').write_text(json.dumps(receipt,indent=2))
print('PREPARED_R2',len(sources),str(run),flush=True)
