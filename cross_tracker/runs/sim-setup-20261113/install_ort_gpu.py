"""Use one pinned ONNX Runtime GPU distribution in the simulation environment."""
from pathlib import Path
import json
import os
import subprocess

root = Path(__file__).resolve().parents[2]
run = Path(__file__).resolve().parent
python = root / 'data/cache/envs/isaaclab-2.2/bin/python'
url = 'https://files.pythonhosted.org/packages/4a/10/cd3e7e289f7b46eb93e38b5c90139f735bf1ea7f03d4b17ceb0e998e5bb6/onnxruntime_gpu-1.22.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=d30c1512f22b1f01bacb4f177d49cbefd23e0f4bef56066f1282992d133e6ff8'
env = {**os.environ, 'UV_CACHE_DIR':str(root / 'data/cache/uv')}
env['NO_PROXY'] = env['no_proxy'] = 'localhost,127.0.0.1,files.pythonhosted.org,pypi.org'
commands = [
    ['uv','pip','uninstall','--python',str(python),'onnxruntime'],
    ['uv','pip','install','--python',str(python),'--no-deps',url],
    ['uv','pip','check','--python',str(python)],
]
record = {'commands':commands,'passed':False}
try:
    with (run / 'ort-gpu-install.log').open('x') as log:
        for command in commands:
            print('COMMAND',command,file=log,flush=True)
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
    record['passed'] = True
finally:
    (run / 'ort-gpu-install.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
