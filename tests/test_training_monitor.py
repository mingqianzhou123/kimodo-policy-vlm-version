"""Exercise false-alarm boundaries and truncated append-only metric records."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from monitor_groot_training import (evaluation_progress, health_alerts, historical_spikes,
                                    read_metrics, recovery_progress, rolling)


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


def test_recovery_download_is_not_extraction_and_parts_are_not_double_counted(tmp_path):
    backup = tmp_path / 'kimodo-lct'
    backup.mkdir()
    entry = {'filename': 'sim.tar.zst', 'size': 10, 'sha256': 'verified',
             'parts': [{'name': 'part000', 'size': 10}]}
    (backup / 'manifest.json').write_text(json.dumps({'components': {'sim': entry}}))
    archives = tmp_path / 'workspace-archives'
    parts = archives / '.download-parts/sim'
    parts.mkdir(parents=True)
    (parts / 'part000').write_bytes(b'1234')
    status = {'restoring_components': ['sim']}
    assert recovery_progress(status, tmp_path)['downloaded_bytes'] == 4
    (archives / 'sim.tar.zst').write_bytes(b'1234567890')
    current = recovery_progress(status, tmp_path)
    assert current['downloaded_bytes'] == 10
    assert not current['components'][0]['restored']
    (archives / 'additive_restore_receipt.json').write_text(json.dumps(
        {'components': {'sim': {'archive_sha256': 'verified'}}}))
    assert recovery_progress(status, tmp_path)['components'][0]['restored']


def test_formal_progress_excludes_smoke_and_requires_complete_validated_inference(tmp_path):
    smoke = tmp_path / 'restored_sim_smoke_any/episodes'
    smoke.mkdir(parents=True)
    (smoke / 'smoke.json').write_text(json.dumps({'success': True}))
    abandoned = tmp_path / 'groot_football_formal_incomplete_old/episodes'
    abandoned.mkdir(parents=True)
    (abandoned / 'old.json').write_text(json.dumps({'success': True}))
    status = {'phase': 'evaluation', 'evaluation_task': 'football'}
    assert evaluation_progress(tmp_path, status)['episodes'] == 0
    out = tmp_path / 'groot_football_formal'
    (out / 'episodes').mkdir(parents=True)
    for i in range(60):
        (out / 'episodes' / f'{i}.json').write_text(json.dumps({'success': i < 21, 'seed': i // 20}))
    (out / 'episodes/broken.json').write_text('{"success":')
    (out / 'summary.json').write_text(json.dumps({'total_episodes': 60, 'total_successes': 21}))
    current = evaluation_progress(tmp_path, status)
    assert current['episodes'] == 60 and current['validated_tasks'] == 0
    assert current['tasks'][0]['success_rate'] == .35
    assert current['tasks'][0]['delta_vs_paper'] is None
    (out / 'validation.json').write_text(json.dumps(
        {'valid_inference': True, 'smoke_only': True, 'episodes': 60}))
    assert evaluation_progress(tmp_path, status)['validated_tasks'] == 0
    (out / 'validation.json').write_text(json.dumps(
        {'valid_inference': True, 'smoke_only': False, 'episodes': 60}))
    current = evaluation_progress(tmp_path, status)
    assert current['validated_tasks'] == 1 and current['tasks'][0]['delta_vs_paper'] == 0
    assert current['tasks'][0]['per_seed']['0'] == {'episodes': 20, 'successes': 20}
