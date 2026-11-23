"""Read frozen asset collision geometry using the installed USD libraries only."""
import os,sys,subprocess,json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/root-dynamics-20261120'

def main():
    OUT.mkdir(exist_ok=True)
    if '--usd' not in sys.argv:
        m=json.loads((ROOT/'dynamics-final-manifest.json').read_text());dst=ROOT/'snapshots/dynamics-before-root-20261120';dst.mkdir(exist_ok=True)
        for n,h in m.items():
            p=ROOT/n;q=dst/n;q.parent.mkdir(parents=True,exist_ok=True)
            if not q.exists():
                assert hashlib.sha256(p.read_bytes()).hexdigest()==h,n
                shutil.copy2(p,q)
            assert hashlib.sha256(q.read_bytes()).hexdigest()==h,n
        (dst/'baseline-manifest.json').write_text(json.dumps(m,indent=2));print('SNAPSHOT',len(m),flush=True)
        envroot=ROOT.parent/'cross_tracker/data/cache/envs/isaaclab-2.2';site=envroot/'lib/python3.11/site-packages/isaacsim'
        usd=list(site.glob('**/pxr/Usd/__init__.py'));assert len(usd)==1,usd
        libs=usd[0].parents[2];env=dict(os.environ,PYTHONPATH=str(libs),LD_LIBRARY_PATH=str((envroot/'bin/python').resolve().parents[1]/'lib')+':'+':'.join(sorted({str(x.parent) for x in site.glob('**/libusd*.so')}))+':'+os.environ.get('LD_LIBRARY_PATH',''))
        subprocess.run([str(envroot/'bin/python'),str(Path(__file__).resolve()),'--usd'],env=env,check=True);return
    from pxr import Usd,UsdGeom,UsdPhysics
    c=json.loads((ROOT/'runs/dynamics-actuator-20261119-r1-base-center/actuator-config.json').read_text());p=Path(c['robot_asset']['path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest()==c['robot_asset']['sha256']
    stage=Usd.Stage.Open(str(p));cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy]);out=[]
    for body in stage.Traverse():
        if body.GetName() not in ('left_ankle_roll_link','right_ankle_roll_link'):continue
        for prim in Usd.PrimRange(body):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            bounds=cache.ComputeRelativeBound(prim,body).ComputeAlignedRange()
            out.append(dict(body=body.GetName(),path=str(prim.GetPath()),type=prim.GetTypeName(),min=list(bounds.GetMin()),max=list(bounds.GetMax()),attributes={a.GetName():str(a.Get()) for a in prim.GetAttributes()}))
    result=dict(robot_asset=c['robot_asset'],foot_colliders=out)
    (OUT/'foot-collision-geometry.json').write_text(json.dumps(result,indent=2))
    print('FOOT COLLIDERS',len(out));print(*[(x['body'],x['type'],x['min'],x['max']) for x in out],sep=chr(10))
if __name__=='__main__':main()
