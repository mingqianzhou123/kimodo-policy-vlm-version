"""The recovery guard must not update weights on anomalous batches."""

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train import guarded_optimizer_step


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
