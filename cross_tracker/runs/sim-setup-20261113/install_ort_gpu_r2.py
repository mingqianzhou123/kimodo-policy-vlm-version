"""Install the relayed official wheel after verifying its fixed SHA256."""
from pathlib import Path
import hashlib
import json
import subprocess

root = Path(__file__).resolve().parents[2]
run = Path(__file__).resolve().parent
python = root / 'data/cache/envs/isaaclab-2.2/bin/python'
name = 'onnxruntime_gpu-1.22.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
p = root / name
assert hashlib.sha256(p.read_bytes()).hexdigest() == 'd30c1512f22b1f01bacb4f177d49cbefd23e0f4bef56066f1282992d133e6ff8'
wheel = root / 'data/cache/sim-wheels' / name
p.rename(wheel)
commands = [
    ['uv','pip','install','--python',str(python),'--no-deps',str(wheel)],
    ['uv','pip','check','--python',str(python)],
]
with (run / 'ort-gpu-install-r2.log').open('x') as log:
    for command in commands:
        print('COMMAND',command,file=log,flush=True)
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
(run / 'ort-gpu-install-r2.json').write_text(json.dumps({'passed':True,'commands':commands},indent=2))
print('ORT_GPU_INSTALLED')
