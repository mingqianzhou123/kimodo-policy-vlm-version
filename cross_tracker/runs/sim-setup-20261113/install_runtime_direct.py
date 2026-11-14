"""Install the pinned simulator packages into the dedicated environment only."""
from pathlib import Path
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
PYTHON = ROOT / 'data/cache/envs/isaaclab-2.2/bin/python'
commands = [
    ('torch', ['torch==2.7.0', 'torchvision==0.22.0', 'torchaudio==2.7.0',
               '--index-url', 'https://download.pytorch.org/whl/cu128']),
    ('isaacsim', ['isaacsim[all,extscache]==5.0.0',
                  '--default-index', 'https://pypi.org/simple',
                  '--index', 'https://pypi.nvidia.com',
                  '--index-strategy', 'unsafe-best-match']),
]
env = {**os.environ, 'UV_CACHE_DIR': str(ROOT / 'data/cache/uv'),
       'UV_HTTP_TIMEOUT': '30', 'UV_HTTP_RETRIES': '3', 'UV_CONCURRENT_DOWNLOADS': '4'}
for key in list(env):
    if key.lower() in {'https_proxy', 'http_proxy', 'all_proxy'}:
        env.pop(key)
record = json.loads((RUN / 'run.json').read_text())
record.update(status='running', training_started=False, formal_training_started=False)
(RUN / 'run.json').write_text(json.dumps(record, indent=2))
for name, packages in commands:
    command = ['uv', 'pip', 'install', '--python', str(PYTHON), *packages]
    start = time.time()
    with (RUN / f'install-direct-{name}.log').open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    receipt = {'command': command, 'exit_code': result.returncode,
               'started_unix': start, 'finished_unix': time.time()}
    (RUN / f'install-direct-{name}.json').write_text(json.dumps(receipt, indent=2))
    print(name, result.returncode, flush=True)
    if result.returncode:
        record.update(status='failed', failed_stage=name)
        (RUN / 'run.json').write_text(json.dumps(record, indent=2))
        raise SystemExit(result.returncode)
record.update(status='packages_installed', simulation_executed=False)
(RUN / 'run.json').write_text(json.dumps(record, indent=2))
print('PACKAGES_INSTALLED; simulator not launched', flush=True)
