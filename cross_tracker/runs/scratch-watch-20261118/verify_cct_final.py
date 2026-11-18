"""Validate the stable CCT 200000 checkpoint on CPU and independently back it up."""
from pathlib import Path
import hashlib
import json
import time

ROOT = Path('/data/NTU_slab/mingqianzhou/humanoid_vla/kimodo-policy-vlm-version/cross_tracker')
RUN = ROOT / 'runs/box-aligned-cct-scratch-s42-20261118-r1'
CHECKPOINT = RUN / 'training/checkpoint_200000'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    metrics = (RUN / 'training/metrics.jsonl').read_text().splitlines()
    report = dict(checked_unix=time.time(), latest_metric=json.loads(metrics[-1]),
                  run_id=RUN.name, checkpoint=str(CHECKPOINT), status='waiting_for_final')
    names = ['config.json', 'training_state.pt']
    if not all((CHECKPOINT / n).is_file() for n in names):
        print(json.dumps(report))
        return
    before = {n: digest(CHECKPOINT / n) for n in names}
    import torch
    torch.set_num_threads(2)
    state = torch.load(CHECKPOINT / 'training_state.pt', map_location='cpu', weights_only=False)
    assert state['global_step'] == 200000
    counts = [0, 0]

    def check(value):
        if isinstance(value, torch.Tensor):
            counts[0] += 1
            counts[1] += value.numel()
            if value.is_floating_point():
                assert torch.isfinite(value).all(), 'Non-finite checkpoint tensor'
        elif isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check(child)

    check(state)
    encoder = [k for k in state['model'] if k.removeprefix('_orig_mod.').startswith('cct_encoder.')]
    assert encoder, 'CCT encoder weights absent'
    assert all(digest(CHECKPOINT / n) == before[n] for n in names), 'Checkpoint still changing'
    import importlib.util
    spec = importlib.util.spec_from_file_location('backup_weights', ROOT / 'scripts/backup_weights.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = ROOT.parents[1] / 'cross_tracker_backups/weights' / RUN.name / 'checkpoint_200000'
    backup = module.backup(CHECKPOINT, destination)
    report.update(status='verified_final', global_step=200000, all_tensors_finite=True,
                  tensor_count=counts[0], tensor_elements=counts[1], cct_encoder_keys=len(encoder),
                  files={n: dict(sha256=before[n], bytes=(CHECKPOINT / n).stat().st_size) for n in names},
                  backup=backup)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
