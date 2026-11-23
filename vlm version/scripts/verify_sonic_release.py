"""Verify archived run weights against pinned HF LFS and release config Git blobs."""
import argparse
import hashlib
import json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]


def verify(run):
    official=ROOT/'runs/actuator-audit-20261119/official';evidence=ROOT/'runs/dynamics-20261119'
    files={x['path']:x for x in json.loads((official/'hf-tree.json').read_text())}
    weights=json.loads((Path(run)/'run.json').read_text())['tracker_sha256']
    if set(weights)!={'model_encoder.onnx','model_decoder.onnx'}:raise ValueError('Incomplete SONIC weight pair')
    for name,sha in weights.items():
        if files[name]['lfs']['oid']!=sha:raise ValueError('Not the pinned default SONIC weight: '+name)
    for path,local in [('config.json','hf-config.json'),('sonic_release/config.yaml','release-training.yaml')]:
        data=(official/local).read_bytes()
        oid=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if files[path]['oid']!=oid:raise ValueError('Release file differs: '+path)
    manifest=json.loads((official/'hf-config.json').read_text());training=yaml.safe_load((official/'release-training.yaml').read_text())
    variant=manifest['variants']['default']
    if variant['training_files']!=['sonic_release/last.pt','sonic_release/config.yaml']:raise ValueError('Release pairing differs')
    if training['manager_env']['config']['robot']['type']!='g1_model_12_dex':raise ValueError('Robot type differs')
    for name,source in json.loads((evidence/'published-source.json').read_text()).items():
        if hashlib.sha256((evidence/name).read_bytes()).hexdigest()!=source['sha256']:raise ValueError('Published source changed')
    spec=json.loads((ROOT/'configs/sonic-dynamics.json').read_text())
    if hashlib.sha256((evidence/'published-g1.py').read_bytes()).hexdigest()!=spec['source_sha256']:raise ValueError('Actuator profile source differs')
    return dict(passed=True,weight_sha256=weights,variant='default',robot_type='g1_model_12_dex',
        training_file='sonic_release/config.yaml',historical_runtime_recovered=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    print(json.dumps(verify(a.run),indent=2))
