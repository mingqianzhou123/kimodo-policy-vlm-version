"""Import verified scratch finals and freeze a private evaluation source tree."""
from pathlib import Path
import hashlib,json,shutil,subprocess,zipfile,time
ROOT=Path('/root/workspace/humanoidVLA')
RUN=ROOT/'cross_tracker/runs/scratch-eval-20261118'
CACHE=ROOT/'cross_tracker/data/cache'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
data=json.loads((RUN/'local-verification.json').read_text())
for m in data['models']:
    bundle=ROOT.parent/f"scratch-{m['method']}-200k-20261118.zip"
    assert zipfile.is_zipfile(bundle), f'Incomplete upload: {bundle}'
    checkpoint=CACHE/'eval-checkpoints'/m['run_id']/'checkpoint_200000'
    with zipfile.ZipFile(bundle) as archive:
        expected={f"{m['run_id']}/checkpoint_200000/{n}" for n in m['files']}
        assert set(archive.namelist())==expected
        for name,item in m['files'].items():
            target=checkpoint/name
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(f"{m['run_id']}/checkpoint_200000/{name}") as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
            assert target.stat().st_size==item['bytes'] and sha(target)==item['sha256'],target
    manifest={**m,'checkpoint':str(checkpoint),'step':200000}
    (RUN/f"{m['method']}-manifest.json").write_text(json.dumps(manifest,indent=2))
    print('CHECKPOINT_VERIFIED',m['method'],m['global_step'],flush=True)
code=RUN/'code'
assert not code.exists(), 'Evaluation snapshot already exists'
code.mkdir()
files=[]
for directory in ['kimodo_vla','scripts','configs','cross_tracker/src','cross_tracker/scripts']:
    files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.json'])
files.append(ROOT/'cross_tracker/runs/eval-setup-20261116/evaluation-plan.json')
sources={}
for src in files:
    name=src.relative_to(ROOT);dst=code/name
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    sources[str(name)]=sha(dst)
(code/'cross_tracker/data').mkdir()
(code/'cross_tracker/data/cache').symlink_to(CACHE,target_is_directory=True)
(code/'checkpoints').symlink_to((ROOT/'checkpoints').resolve(),target_is_directory=True)
# Freeze the simulation code/configuration too; binary assets remain hash-recorded shared files.
arena=CACHE/'sources/HumanoidArena-44a72229f11bab47a145e12db39ed891b677e783'
assets={str(p.relative_to(CACHE)):sha(p) for p in arena.rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.onnx']}
for p in (CACHE/'text-eval').glob('*.pt'):assets[str(p.relative_to(CACHE))]=sha(p)
for p in (CACHE/'sim-assets/sonic').glob('*.onnx'):assets[str(p.relative_to(CACHE))]=sha(p)
robot=CACHE/'sim-assets/arena/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd'
assets[str(robot.relative_to(CACHE))]=sha(robot)
(RUN/'source-files.json').write_text(json.dumps(sources,indent=2))
(RUN/'shared-eval-assets.json').write_text(json.dumps(assets,indent=2))
(RUN/'prepare.json').write_text(json.dumps(dict(status='complete',source_files=len(sources),shared_assets=len(assets),code=str(code),platform='Baidu T2M-PLUS dv-e21ddf3a87',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),finished_unix=time.time()),indent=2))
print('PREPARED',len(sources),len(assets),flush=True)
