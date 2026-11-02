"""Exercise false-alarm boundaries and truncated append-only metric records."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from monitor_groot_training import health_alerts, historical_spikes, read_metrics, rolling


def records(count, stage="adapter", loss=.02):
    return [{"step": 10*(i+1), "stage": stage, "loss": loss,
             "grad_control": .1, "grad_hand": .2} for i in range(count)]


def test_stall_and_exit_alerts_respect_pipeline_phase():
    rows = records(1)
    assert "metrics_stalled" in health_alerts(rows, "training", True, 301)
    assert "pipeline_missing" in health_alerts(rows, "training", False, 0)
    assert not health_alerts(rows, "evaluation", True, 900)
    assert not health_alerts(rows, "complete", False, 900)
    assert "pipeline_failed" in health_alerts(rows, "failed", False, 0)
    rows[0]["loss"] = float("nan")
    assert "nonfinite_metrics" in health_alerts(rows, "training", True, 0)


def test_sustained_spike_requires_enough_records_in_the_same_stage():
    rows = records(500, stage="injection") + records(100, stage="adapter", loss=1.)
    assert not health_alerts(rows, "training", True, 0)
    rows = records(500) + records(100, loss=1.)
    assert "sustained_loss_spike" in health_alerts(rows, "training", True, 0)


def test_partial_final_append_is_ignored_and_smoothing_ignores_nonfinite_values(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(records(1)[0]) + '\n{"step": 20')
    rows, invalid = read_metrics(path)
    assert len(rows) == 1 and invalid == []
    path.write_text('{bad json}\n' + json.dumps(records(1)[0]) + '\n')
    rows, invalid = read_metrics(path)
    assert len(rows) == 1 and invalid == [1]
    np.testing.assert_allclose(rolling([1., np.nan, 3., 5.], width=2), [1., 1., 3., 4.])


def test_single_severe_spike_is_retained_after_recovery():
    rows = records(600)
    rows[-1].update(loss=3., grad_control=1000.)
    alerts = health_alerts(rows, "training", True, 0)
    assert "loss_outlier" in alerts and "grad_control_outlier" in alerts
    assert "sustained_loss_spike" not in alerts
    assert len(historical_spikes(rows)) == 1
    recovered = rows + records(600)
    assert not health_alerts(recovered, "training", True, 0)
    assert len(historical_spikes(recovered)) == 1
