import functools
import http.server
import json
import threading
import urllib.request

import numpy as np
import pytest

import overview
from paths import RUNS
from report_settled import wrist_stats
from analyze_sonic_replay import vector_stats

ARRIVAL = .6


def logged(offset_cm, frames=100, marker_offset_cm=0.):
    """A run holding still: wrists a fixed distance from the reference, plus noise of a millimetre or less."""
    rng = np.random.default_rng(0)
    time = .02*np.arange(1, frames+1)
    reference = rng.normal(size=(2, 3))+rng.normal(scale=1e-3, size=(frames, 2, 3))
    wrist = reference+[offset_cm/100, 0, 0]+rng.normal(scale=1e-4, size=(frames, 2, 3))
    reference_root = rng.normal(size=(frames, 3))
    root = reference_root+rng.normal(scale=.01, size=(frames, 3))
    rows = [{'before': {'time': t-.02}, 'after': {'time': t, 'wrist_pos': w.tolist(), 'root_pos': r.tolist()},
             'reference_wrist_world': ref.tolist(), 'reference_root_world': rr.tolist()}
            for t, w, r, ref, rr in zip(time, wrist, root, reference, reference_root)]
    late = time > ARRIVAL+1e-6
    marker = wrist[late].mean(axis=0)+[0, marker_offset_cm/100, 0]
    return rows, dict(late=late, wrist=wrist, reference=reference, root=root, reference_root=reference_root, marker=marker)


def save(folder, rows=None, meta=None, target=None, truth=None):
    folder.mkdir(parents=True)
    if meta is not None:
        (folder/'run.json').write_text(json.dumps(meta))
    if rows is not None:
        (folder/'execution.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    if target is not None:
        (folder/'target.json').write_text(json.dumps(target))
    if truth is not None:
        (folder/'visual-evaluation-truth.json').write_text(json.dumps(truth))
    return folder


@pytest.mark.parametrize('offset_cm, marker_offset_cm, passes', [(1, 0, True), (3.5, 0, False), (1, 4, False)])
def test_numbers_follow_the_settled_report(tmp_path, offset_cm, marker_offset_cm, passes):
    rows, a = logged(offset_cm, marker_offset_cm=marker_offset_cm)
    run = save(tmp_path/'run', rows, {}, {'arrival_time': ARRIVAL}, {'targets_world': a['marker'].tolist()})
    m = overview.measure(run)['metrics']
    late = a['late']
    tracking = wrist_stats((a['wrist']-a['reference'])[late])
    marker = wrist_stats(a['wrist'][late]-a['marker'])
    assert m['late_frames'] == late.sum()
    assert m['wrist_max_cm'] == pytest.approx(tracking['max_norm'], abs=1e-3)
    assert m['wrist_mean_cm'] == pytest.approx(tracking['mean_norm'], abs=1e-3)
    assert m['marker_max_cm'] == pytest.approx(marker['max_norm'], abs=1e-3)
    assert m['root_mean_cm'] == pytest.approx(vector_stats((a['root']-a['reference_root'])[late])['mean_norm'], abs=1e-3)
    assert m['pass'] == (tracking['strict_pass'] and marker['strict_pass']) == passes


def test_numbers_match_the_stored_comparison():
    stored = RUNS/'settled-20261123-r1/comparison.json'
    if not stored.exists():
        pytest.skip('the settled suite is not on this machine')
    for case, result in json.loads(stored.read_text()).items():
        for side, suite in (('new', 'settled-20261123-r1'), ('old', 'dynamics-actuator-20261119-r1')):
            m, want = overview.measure(RUNS/f'{suite}-{case}')['metrics'], result[side]
            assert m['wrist_max_cm'] == pytest.approx(want['tracking']['max_norm'], abs=1e-3)
            assert m['marker_max_cm'] == pytest.approx(want['actual_marker']['max_norm'], abs=1e-3)
            assert m['root_mean_cm'] == pytest.approx(want['root_cm']['mean_norm'], abs=1e-3)
            assert m['pass'] == (want['tracking']['strict_pass'] and want['actual_marker']['strict_pass'])


def test_old_runs_with_missing_pieces_still_show(tmp_path):
    rows, _ = logged(1)
    runs = tmp_path/'runs'
    save(runs/'empty-20261117-r1', meta={})
    save(runs/'broken-20261117-r1').joinpath('run.json').write_text('{"status": "comp')
    save(runs/'no-reference-20261117-r1', [{'after': {'time': 1, 'wrist_pos': [[0, 0, 0], [0, 0, 0]]}}], {'status': 'failed', 'error': 'boom'})
    no_target = save(runs/'no-target-20261117-r1', rows, {'status': 'completed', 'source_sha256': {'a.py': 'x'}})
    with (no_target/'execution.jsonl').open('a') as log:
        log.write('\n{"after": {"time": 2.0')
    save(runs/'wrists-only-20261117-r1', [{k: v for k, v in r.items() if k != 'reference_root_world'} for r in rows],
         {'status': 'completed', 'git_commit': 'abcdef0123456789', 'git_dirty': False}, {'arrival_time': ARRIVAL})
    (runs/'review').mkdir()

    data = overview.collect(runs, runs/'overview', {'experiments': []})
    page = overview.write(data, runs/'overview')

    by_name = data['runs']
    assert set(by_name) == {'empty-20261117-r1', 'broken-20261117-r1', 'no-reference-20261117-r1', 'no-target-20261117-r1', 'wrists-only-20261117-r1'}
    assert data['other_folders'] == 1
    for name in ('empty-20261117-r1', 'broken-20261117-r1', 'no-reference-20261117-r1'):
        assert by_name[name]['m'] is None and by_name[name]['s'] is None and by_name[name]['video'] is None
    assert by_name['empty-20261117-r1']['status'] == by_name['broken-20261117-r1']['status'] == '未知'
    assert by_name['no-reference-20261117-r1']['error'] == 'boom'
    untimed = by_name['no-target-20261117-r1']
    assert untimed['m'] == {'frames': 100, 'late_frames': 0} and len(untimed['s']['t']) == 100
    assert set(untimed['version']) == {'fingerprint'} and untimed['date'] == '2026-11-17'
    partial = by_name['wrists-only-20261117-r1']
    assert partial['version'] == {'commit': 'abcdef01', 'dirty': False}
    assert partial['m']['pass'] and 'root_mean_cm' not in partial['m'] and 'root' not in partial['s']
    assert 'wrists-only-20261117-r1' in page.read_text()


def record(name, status='完成', wins=None, episodes=1):
    return {'name': name, 'status': status, 'wins': wins, 'episodes': episodes, 'error': None, 'version': None, 'm': {'frames': 400}}


def test_progress_counts_what_has_been_run():
    plan = {'experiments': [{'id': 'one', 'title': 'One', 'basis': ['old-*'], 'steps': [
        {'name': 'static', 'runs': 'one-static-*', 'planned': 4},
        {'name': 'extra', 'runs': 'one-extra-*', 'planned': 2, 'optional': True},
        {'name': 'typed in', 'done': 3, 'planned': 2},
        {'name': 'not yet', 'done': 0}]}]}
    runs = [record('one-static-a', wins=1), record('one-static-b', wins=0), record('one-static-c', '失败'),
            record('one-static-many', wins=2, episodes=3), record('old-a'), record('two-static-a', wins=1)]
    experiment, = overview.progress(plan, runs)
    static, extra, typed, later = experiment['steps']
    assert (static['done'], static['share']) == (5, 1) and len(static['runs']) == 4
    assert static['summary']['success']['wins'] == 3 and static['summary']['success']['total'] == 5
    assert static['summary']['notes'] == ['one-static-c 失败']
    assert extra['done'] == 0 and typed['share'] == 1 and later['share'] == 0
    assert (experiment['steps_done'], experiment['steps_counted']) == (2, 3)
    assert experiment['progress'] == pytest.approx(2/3)
    assert experiment['basis'] == ['old-a']
    experiment, = overview.progress(plan, runs+[record('one-extra-a', wins=1)])
    assert experiment['steps_counted'] == 4 and experiment['progress'] == pytest.approx((1+.5+1+0)/4)


def test_success_is_only_counted_where_the_task_was_attempted(tmp_path):
    def wins(name, meta, episode):
        run = save(tmp_path/name, meta=meta)
        (run/'episode.json').write_text(json.dumps(episode))
        return overview.describe(run, tmp_path/'overview')['wins']
    assert wins('reach-only', {'status': 'completed', 'success_rate_evaluation': False}, {'success': False}) is None
    assert wins('lifted', {'status': 'completed', 'success_rate_evaluation': True}, {'success': True}) == 1
    assert wins('dropped', {'status': 'completed'}, {'success': False}) == 0
    assert wins('unknown', {'status': 'completed'}, {}) is None


def test_rate_reports_how_little_five_runs_prove():
    five = overview.rate(5, 5)
    assert (five['low'], five['high']) == (pytest.approx(.5655, abs=1e-3), 1)
    half = overview.rate(15, 30)
    assert half['low'] == pytest.approx(.332, abs=1e-3) and half['high'] == pytest.approx(.668, abs=1e-3)
    assert overview.rate(None, 5) is None and overview.rate(0, 0) is None


def test_h264_video_is_linked_where_it_is(tmp_path):
    video = tmp_path/'runs/some-run/videos/episode.mp4'
    video.parent.mkdir(parents=True)
    video.write_bytes(b'....ftypisom....avc1....')
    assert overview.playable(video, tmp_path/'runs/overview') == {'url': '../some-run/videos/episode.mp4', 'plays': True}
    assert not (tmp_path/'runs/overview').exists()


def test_server_answers_byte_ranges(tmp_path):
    (tmp_path/'clip.mp4').write_bytes(bytes(range(100)))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(overview.Ranged, directory=str(tmp_path)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f'http://127.0.0.1:{server.server_port}/clip.mp4'
        for asked, first, last in (('bytes=10-19', 10, 19), ('bytes=90-', 90, 99), ('bytes=-5', 95, 99)):
            with urllib.request.urlopen(urllib.request.Request(url, headers={'Range': asked})) as reply:
                assert reply.status == 206 and reply.headers['Content-Range'] == f'bytes {first}-{last}/100'
                assert reply.read() == bytes(range(first, last+1))
        with urllib.request.urlopen(url) as reply:
            assert reply.status == 200 and len(reply.read()) == 100
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize('verdict', [True, False])
def test_a_moving_goal_run_is_judged_by_its_own_result_file(tmp_path, verdict):
    rows, _ = logged(8)  # 8 cm from the reference: fails the standing 3 cm rule either way
    run = save(tmp_path/'run', rows, {'status': 'completed'}, {'arrival_time': ARRIVAL})
    assert overview.describe(run, tmp_path)['m']['pass'] is False
    (run/'motion-result.json').write_text(json.dumps({'pass': verdict}))
    assert overview.describe(run, tmp_path)['m']['pass'] is verdict
