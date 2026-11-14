"""Verify real pretrained model construction and GPU loading; no placeholder text input."""
import json
import os
from pathlib import Path
import sys

root = Path('cross_tracker').resolve()
run = root / 'runs/gpu-load-20261113'
sys.path[:0] = [str(root / 'scripts'), str(root.parent / 'scripts'), str(root.parent)]
import torch
from omegaconf import OmegaConf
from fetch_assets import matches
from run_config import load_yaml
from train import build_model, check_initialization, initialize_from

record = json.loads((run / 'run.json').read_text())
assert record['stage'] == 'diagnostic' and record['status'] == 'planned'
record.update(status='running', training_started=False)
(run / 'run.json').write_text(json.dumps(record, indent=2))
try:
    protocol = json.loads((run / 'config.json').read_text())
    rel = protocol['initial_checkpoint']['path']
    checkpoint = root / 'data/cache/weights' / rel
    assets = json.loads((root / 'data/manifests/weights-v1.json').read_text())['files']
    for name in ('config.json', 'training_state.pt'):
        assert matches(checkpoint / name, assets[f'{rel}/{name}'])
    shared = json.loads((root / 'data/manifests/shared-weights-v1.json').read_text())['files']
    assert all(matches(root.parent / name, expected) for name, expected in shared.items())
    config = load_yaml(root / 'configs/kimodo_pilot.yaml')
    config.initialization = {'checkpoint': str(checkpoint)}
    check_initialization(config)
    (run / 'model-config.yaml').write_text(OmegaConf.to_yaml(config))
    torch.manual_seed(record['seed'])
    model = build_model(config)
    initialize_from(model, config)
    model.cuda()
    torch.cuda.synchronize()
    result = {'loaded': True, 'forward_executed': False, 'training_started': False,
              'parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'peak_memory_bytes': torch.cuda.max_memory_allocated()}
    (run / 'result.json').write_text(json.dumps(result, indent=2))
    record['status'] = 'completed'
    print(json.dumps(result), flush=True)
except BaseException as error:
    record.update(status='failed', error_type=type(error).__name__)
    raise
finally:
    (run / 'run.json').write_text(json.dumps(record, indent=2))
