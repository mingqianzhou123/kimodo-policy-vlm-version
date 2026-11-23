"""Verify the final code, raw evidence, preserved baseline and untouched model paths."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def check(manifest,base):
    data=json.loads(manifest.read_text())
    bad=[p for p,h in data.items() if not (base/p).is_file() or hashlib.sha256((base/p).read_bytes()).hexdigest()!=h]
    if bad:raise ValueError((str(manifest),bad))
    print('PASS',manifest.name,len(data),flush=True)
    return data

def main():
    check(ROOT/'root-final-manifest.json',ROOT)
    check(ROOT/'root-evidence-manifest.json',ROOT)
    snap=ROOT/'snapshots/dynamics-before-root-20261120';old=check(snap/'baseline-manifest.json',snap)
    fixed=[p for p in old if p.startswith('kimodo_vla/') or p in ('configs/sonic-dynamics.json','arena/oracle_bridge.py')]
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==old[p] for p in fixed)
    print('PASS unchanged VLM/Kimodo/IK/backend',len(fixed),flush=True)
    d=json.loads((ROOT/'runs/root-dynamics-20261120/comparison.json').read_text())
    for label,cs in d['cases'].items():
        assert len(cs)==5
        assert all(r['invariants']['passed'] and r['invariants']['decoder_target_delta_rad']==0 for r in cs.values())
        a=d['aggregate'][label]
        print(label,'root_cm',round(a['root_mean_cm'],3),'wrist_LR_cm',[round(x,3) for x in a['wrist_mean_cm']],'both3cm',str(a['both_3cm_passes'])+'/5',flush=True)
    print('COMPLETE: 15 new cases + 3 identical controls; original ground is not fixed',flush=True)
if __name__=='__main__':main()
