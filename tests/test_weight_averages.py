"""The exponential moving averages train.py keeps of the trainable weights."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train import WeightAverages  # noqa: E402


def model_with(value: float) -> torch.nn.Module:
    model = torch.nn.Linear(2, 1, bias=False)
    frozen = torch.nn.Linear(1, 1)
    frozen.requires_grad_(False)
    model.frozen = frozen
    with torch.no_grad():
        model.weight.fill_(value)
    return model


def test_average_follows_the_update_rule_and_skips_frozen_weights():
    model = model_with(0.0)
    averages = WeightAverages(model, [0.9])
    assert averages.names == ["weight"]
    expected = torch.zeros(1, 2)
    for step in range(1, 30):
        with torch.no_grad():
            model.weight.fill_(float(step))
        averages.update(step)
        rate = min(0.9, (1 + step) / (10 + step))
        expected = rate * expected + (1 - rate) * float(step)
    torch.testing.assert_close(averages.state()["0.9"]["weight"], expected)


def test_saved_state_resumes_the_same_average():
    model = model_with(1.0)
    averages = WeightAverages(model, [0.99, 0.999])
    for step in range(1, 6):
        with torch.no_grad():
            model.weight.add_(1.0)
        averages.update(step)
    resumed = WeightAverages(model, [0.99, 0.999], averages.state())
    for decay in ("0.99", "0.999"):
        torch.testing.assert_close(resumed.state()[decay]["weight"], averages.state()[decay]["weight"])
