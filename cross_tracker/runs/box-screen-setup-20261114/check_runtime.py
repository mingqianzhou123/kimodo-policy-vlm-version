"""CPU acceptance of the exact three training datasets and effective DDP recipe."""
import json
import importlib.metadata
import platform
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent), str(ROOT.parent / 'scripts')]
from sampling import TrainingDataset
from kimodo_vla.model.planner import load_text_embeddings
from run_config import load_yaml

text = load_text_embeddings(ROOT / 'data/cache/text')
config = load_yaml(ROOT / 'configs/box_screen.yaml')
assert config.main.batch_size == 8 and config.main.micro_batch_size == 1
assert config.main.seed == 42 and config.main.max_steps == 10000
results = []
for sources in [['sonic'], ['twist2'], ['sonic', 'twist2']]:
    roots = [ROOT / 'data/cache/prepared-v1/train/HOI_pp_box' / s for s in sources]
    data = TrainingDataset(roots, action_history=100, action_chunk=50, text_embeddings=text, cameras=['head'])
    assert len(data.episodes) == 43 * len(sources)
    assert {e['task'] for e in data.episodes} == {'HOI_pp_box'}
    assert {e['tracker'] for e in data.episodes} == set(sources)
    assert all(e['instruction'] in text for e in data.episodes)
    for index in [0, len(data)//2, len(data)-1]:
        sample = data[index]
        assert sample['gt_motion'].shape == (150, 417)
        assert sample['text_embedding'].shape[-1] == 4096
    results.append({'sources': sources, 'episodes': len(data.episodes), 'windows': len(data)})
report = {'passed': True, 'world_size': 8, 'global_batch': 8, 'datasets': results,
          'python': platform.python_version(),
          'packages': {name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'transformers', 'numpy', 'pyarrow', 'av', 'omegaconf']},
          'gpu_execution_verified': False, 'note': 'CPU acceptance; actual DDP starts only after ACP allocation'}
(ROOT / 'runs/box-screen-setup-20261114/runtime-check.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
