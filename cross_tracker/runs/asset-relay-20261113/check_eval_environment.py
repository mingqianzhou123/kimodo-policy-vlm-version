"""Read-only inventory; missing runtime is not a simulator rollout test."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

root = Path('cross_tracker')
arena = Path('third_party/HumanoidArena')
modules = {}
for name in ('isaacsim', 'isaaclab', 'omni', 'onnxruntime'):
    try:
        spec = importlib.util.find_spec(name)
        modules[name] = str(spec.origin) if spec else None
    except (ImportError, ModuleNotFoundError):
        modules[name] = None
controllers = {}
source = Path('kimodo_vla/benchmarks/arena/policy.py')
for node in ast.parse(source.read_text()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ROOT_BOUNDARY' for t in node.targets):
        controllers = ast.literal_eval(node.value)
patterns = {'mimic_lite': 'policy-xua2csee-4000.*', 'twist2': 'twist2*.onnx',
            'sonic_release': '**/gear_sonic_deploy/policy/release/*',
            'robot_assets': '**/assets/robots/*', 'object_assets': '**/assets/objects/*'}
assets = {key: [str(p) for p in arena.glob('**/'+pattern) if p.is_file()][:30]
          for key, pattern in patterns.items()}
report = {'python': sys.executable, 'modules': modules, 'assets': assets,
          'kimodo_controller_support': controllers, 'simulation_executed': False,
          'formal_training_started': False,
          'blockers': []}
if not modules['isaacsim'] or not modules['isaaclab']:
    report['blockers'].append('No Isaac Sim/Isaac Lab in the active environment')
for key, paths in assets.items():
    if not paths:
        report['blockers'].append('No files at expected asset path: '+key)
if 'mimic_lite' not in controllers:
    report['blockers'].append('Kimodo serving entry has no MimicLite root-boundary mapping')
result = subprocess.run(['git', '-C', str(arena), 'rev-parse', 'HEAD'], capture_output=True, text=True)
report['arena_revision'] = result.stdout.strip() if result.returncode == 0 else None
output = root / 'runs/readiness-20261113-r2/eval-environment.json'
output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
