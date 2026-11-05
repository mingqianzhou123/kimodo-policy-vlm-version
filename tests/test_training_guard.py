"""The recovery guard must not update weights on anomalous batches."""

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train import guarded_optimizer_step, make_optimizer, restore_training_rng
import pytest


def test_guard_skips_large_update_and_keeps_scheduler_in_step():
    weight = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([weight], lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config = OmegaConf.create({"max_update_loss": 0.2, "max_update_grad_norm": 10.0})

    weight.grad = torch.tensor(0.1)
    assert not guarded_optimizer_step(optimizer, scheduler, 0.04, 0.1, 0.1, config)
    baseline = weight.detach().clone()
    previous_epoch = scheduler.last_epoch

    weight.grad = torch.tensor(100.0)
    assert guarded_optimizer_step(optimizer, scheduler, 0.7, 300.0, 0.1, config)
    assert torch.equal(weight, baseline)
    assert weight.grad is None
    assert scheduler.last_epoch == previous_epoch + 1

    weight.grad = torch.tensor(0.1)
    assert not guarded_optimizer_step(optimizer, scheduler, 0.04, 0.1, 0.1, config)
    assert weight.item() < baseline.item()


def test_new_vlm_group_warms_up_after_unfreezing():
    model = torch.nn.Module()
    model.controlnet = torch.nn.Linear(2, 2)
    model.image_encoder = torch.nn.Module()
    model.image_encoder.adapter = torch.nn.Linear(2, 2)
    config = OmegaConf.create({'main': {'max_steps': 500000}, 'training': {
        'stages': {'adapter_start': 10000},
        'scheduler': {'warmup_steps': 10000, 'adapter_warmup_steps': 5000, 'min_lr_ratio': .1},
        'optimizer': {'control_backbone_lr': 3e-5, 'control_adapter_lr': 2e-5,
                      'hand_lr': 2e-5, 'vl_adapter_lr': 5e-6,
                      'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .01}}})
    groups, optimizer, scheduler = make_optimizer(model, config, torch.device('cpu'), step=10000)
    assert all(g['lr'] == 0 for g in groups if g['name'].startswith('vl_adapter'))
    assert all(g['lr'] > 0 for g in groups if g['name'].startswith('adapter'))
    for _ in range(2500):
        optimizer.step()
        scheduler.step()
    assert all(2.49e-6 < g['lr'] <= 2.5e-6 for g in groups if g['name'].startswith('vl_adapter'))


def test_world_size_migration_is_explicit_and_restartable():
    cfg = OmegaConf.create({'main': {'seed': 42, 'original_ordinal_stream': True}})
    state = {'rng_states': [{'cpu': torch.get_rng_state()} for _ in range(4)]}
    with pytest.raises(RuntimeError, match='explicit migration'):
        restore_training_rng(state, cfg, torch.device('cpu'), 8, 6, 20000)
    cfg.main.allow_world_size_change = True
    restore_training_rng(state, cfg, torch.device('cpu'), 8, 6, 20000)
    migrated = torch.get_rng_state().clone()
    expected = torch.rand(5)
    resumed = {'rng_states': [{'cpu': migrated} for _ in range(8)]}
    restore_training_rng(resumed, cfg, torch.device('cpu'), 8, 6, 20001)
    assert torch.equal(torch.rand(5), expected)
