"""Read the exact recorded USD asset without launching or editing a simulation.

Run with the already-installed Isaac Sim USD Python/lib directories on
PYTHONPATH/LD_LIBRARY_PATH; no new dependency or shared-environment change is needed.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    from pxr import Usd
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();config=json.loads(a.config.read_text());asset=config['robot_asset'];path=Path(asset['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=asset['sha256']:raise ValueError('Robot asset changed')
    stage=Usd.Stage.Open(str(path));bodies=[]
    for prim in stage.Traverse():
        schemas=prim.GetAppliedSchemas()
        if prim.GetName() not in config['body_names'] or 'PhysicsRigidBodyAPI' not in schemas:continue
        bodies.append(dict(name=prim.GetName(),path=str(prim.GetPath()),schemas=schemas,
            attributes={x.GetName():dict(value=str(x.Get()),authored=x.HasAuthoredValueOpinion())
                for x in prim.GetAttributes() if x.GetName().startswith(('physics:','physxRigidBody:'))}))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(robot_asset=asset,bodies=bodies),indent=2))
    print('Verified asset SHA256; inspected',len(bodies),'rigid bodies')
    for b in bodies:
        if b['name'] in ('imu_in_pelvis','imu_in_torso','d435_link','mid360_link'):
            print(b['name'],b['attributes']['physics:mass'])


if __name__=='__main__':main()
