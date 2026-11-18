"""Audit the complete 480-episode queue before calculating the final eight cells."""
from pathlib import Path
import argparse
import hashlib
import json
import time

TASK = 'Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby'
METHODS = ['sonic', 'twist2', 'mix', 'cct']
TRACKERS = ['sonic', 'twist2']


def seed_for(group, repeat):
    return int.from_bytes(hashlib.sha256(f'{TASK}|{group}|{repeat}'.encode()).digest()[-4:], 'big') & 0x7fffffff


def summarize(batch, reports, manifests, require_complete=True):
    expected = {(m, c, g, e) for m in METHODS for c in TRACKERS for g in range(3) for e in range(20)}
    seen, episodes = set(), []
    if require_complete:
        assert batch['status'] == 'completed' and len(batch['completed']) == 480
        assert set(manifests) == set(METHODS)
    for item in batch['completed']:
        key = tuple(item[n] for n in ['method', 'controller', 'group', 'repeat'])
        assert key in expected and key not in seen, key
        seen.add(key)
        method, controller, group, repeat = key
        seed = seed_for(group, repeat)
        report = reports[item['run_id']]
        manifest = manifests[method]
        assert manifest.get('step', manifest.get('global_step')) == 200000
        assert report['status'] == 'completed' and report['valid_for_success_rate']
        assert report['model'] == manifest['run_id'] and report['controller'] == controller
        assert report['checkpoint_files'] == manifest['files']
        assert report['simulator_exit_code'] == 0 and report['max_steps'] == 1450
        assert report['fresh_sim'] and report['deterministic'] and report['post_termination_steps'] == 10
        assert report['group_seed'] == group and report['repeat_index'] == repeat
        command = report['server_command']
        assert command[command.index('--timing') + 1] == 'original'
        if method == 'cct':
            assert command[2].endswith('/serve_cct.py') and '--deterministic' in command
        else:
            assert command[2].endswith('/serve_arena_deterministic.py')
            for flag, value in [('--dtype', 'fp32'), ('--execution-frames', '15'),
                                ('--diffusion-steps', '10'), ('--rtc-overlap-frames', '0')]:
                assert command[command.index(flag) + 1] == value
        assert len(report['episodes']) == 1
        episode = report['episodes'][0]
        assert episode['seed'] == item['seed'] == seed
        assert episode['object_seed'] == seed ^ 0x9E3779B185EBCA87
        assert isinstance(episode['success'], bool)
        assert 0 < episode['steps'] <= 1450 and episode['video_frames'] > 0
        assert len(episode['video_sha256']) == 64
        assert all(item[k] == v for k, v in episode.items())
        if controller == 'sonic':
            assert report['refresh_video'] and report['robot_usd'].endswith('_m2.usd')
        else:
            assert report['robot_usd'] is None and report['twist2_physics_timing'] == 'recorded'
        episodes.append(dict(method=method, controller=controller, group=group, repeat=repeat,
                             run_id=item['run_id'], **episode))
    if require_complete:
        assert seen == expected
    cells = []
    for method in METHODS:
        for controller in TRACKERS:
            values = [x for x in episodes if x['method'] == method and x['controller'] == controller]
            count, successes = len(values), sum(x['success'] for x in values)
            scope = 'seen_source' if method in ('mix', 'cct') else ('same_source' if method == controller else 'cross_source')
            groups = []
            for group in range(3):
                part = [x for x in values if x['group'] == group]
                groups.append(dict(group=group, completed=len(part), successes=sum(x['success'] for x in part)))
            cells.append(dict(method=method, controller=controller, scope=scope, successes=successes,
                completed=count, target=60, success_rate=successes / count if count else None, groups=groups))
    return dict(complete=seen == expected and batch['status'] == 'completed', valid_episodes=len(episodes),
                target_episodes=480, cells=cells, episodes=episodes, protocol=batch['protocol'],
                infrastructure_failed_attempts=len(batch['failed']),
                interpretation='Both dual-source targets are seen-source. TWIST2 uses the frozen adapted recording-compatible protocol.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    batch = json.loads((args.run / 'batch.json').read_text())
    manifests = {m: json.loads((args.run / f'{m}-manifest.json').read_text()) for m in METHODS}
    reports, hashes = {}, {}
    for item in batch['completed']:
        path = Path(item['command'][2]).parents[1] / 'runs' / item['run_id'] / 'run.json'
        data = path.read_bytes()
        reports[item['run_id']] = json.loads(data)
        hashes[item['run_id']] = hashlib.sha256(data).hexdigest()
    result = summarize(batch, reports, manifests)
    result.update(checked_unix=time.time(), checkpoints=manifests, receipt_sha256=hashes)
    path = args.run / time.strftime('final-report-%Y%m%d-%H%M%S.json')
    with path.open('x') as output:
        json.dump(result, output, indent=2)
    print('FINAL_REPORT', path, hashlib.sha256(path.read_bytes()).hexdigest())
    for cell in result['cells']:
        print(cell['method'], cell['controller'], cell['scope'], f"{cell['successes']}/60", f"{cell['success_rate']:.2%}")


if __name__ == '__main__':
    main()
