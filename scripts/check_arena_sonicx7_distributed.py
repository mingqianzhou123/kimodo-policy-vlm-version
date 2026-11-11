"""Audit completed four-GPU stage-transition/resume/stop smoke receipts."""
import argparse
import json
import math
from pathlib import Path

from omegaconf import OmegaConf
import torch

from groot_experiment import atomic_json, fingerprint
from run_config import REPO, load_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'configs/arena_sonicx7_groot.yaml')
    parser.add_argument('--run', type=Path, default=REPO / 'outputs/sonicx7_ddp_check/train')
    parser.add_argument('--gpus', default='4,5,6,7')
    parser.add_argument('--out', type=Path, default=REPO / 'outputs/sonicx7_preflight/distributed.json')
    args = parser.parse_args()
    config = load_yaml(args.config)
    tested = OmegaConf.create(json.loads((args.run / 'config.json').read_text()))
    if tested.training.stages.adapter_start != 3:
        raise ValueError('The smoke must exercise unfreezing after three updates')
    tested.training.stages.adapter_start = config.training.stages.adapter_start
    if fingerprint(tested) != fingerprint(config):
        raise ValueError('The compiled smoke differs from the production model/data/optimizer recipe')
    for key in ['num_workers', 'episode_cache_size', 'micro_batch_size', 'batch_size']:
        if tested.main[key] != config.main[key]:
            raise ValueError(f'Smoke used different main.{key}')
    audits = [json.loads((args.run / f'distributed_audit_{step}.json').read_text()) for step in [6, 7, 10]]
    for audit in audits:
        if not audit['passed'] or audit['world_size'] != 4 or len(set(audit['sha256_by_rank'])) != 1:
            raise ValueError('Four ranks have diverging saved weights')
    rows = [json.loads(line) for line in (args.run / 'metrics.jsonl').read_text().splitlines()]
    if [r['step'] for r in rows] != list(range(1, 11)):
        raise ValueError('Resume skipped or repeated optimizer updates')
    for row in rows:
        if row['stage'] != ('injection' if row['step'] <= 3 else 'adapter'):
            raise ValueError('Unexpected stage boundary')
        if not all(math.isfinite(row[k]) for k in ['loss', 'grad_control', 'grad_hand']):
            raise ValueError('Non-finite loss or gradient')
    state = torch.load(args.run / 'resume.pt', map_location='cpu', weights_only=True)
    if state['global_step'] != 10 or len(state['rng_states']) != 4:
        raise ValueError('Resume lacks complete global update/RNG state')
    steps = {}
    for group in state['optimizer']['param_groups']:
        counts = sorted({int(state['optimizer']['state'][key]['step']) for key in group['params']})
        expected = 7 if group['name'].startswith('vl_adapter') else 10
        if counts != [expected]:
            raise ValueError(f'Adam moments reset or were not restored: {group["name"]}: {counts}')
        steps[group['name']] = counts
    stop_log = (args.run.parent / 'resume_stop.log').read_text()
    if 'checkpoint-and-stop request completed at step 7' not in stop_log:
        raise ValueError('Safe checkpoint-and-stop was not exercised')
    atomic_json(args.out, {'passed': True, 'fingerprint': fingerprint(config),
        'gpus': [int(v) for v in args.gpus.split(',')], 'world_size': 4,
        'global_batch': int(config.main.batch_size),
        'micro_batch': int(config.main.micro_batch_size),
        'workers_per_gpu': int(config.main.num_workers), 'compiled': True,
        'test_overrides': {'adapter_start': 3, 'stop_after': [6, 10], 'require_preflight': False, 'log_every': 1},
        'audits': audits, 'optimizer_steps_by_group': steps, 'metrics': rows,
        'safe_stop_and_resume': True, 'production_initialization': 'fresh, not this smoke checkpoint'})
    print('PASS: compiled four-GPU training, stage transition, Adam/RNG resume, safe stop, all-rank weight equality')


if __name__ == '__main__':
    main()
