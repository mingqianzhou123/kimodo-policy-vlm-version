"""Run frozen evaluation episodes on explicitly enabled, disjoint GPU pairs."""
from pathlib import Path
import fcntl
import json
import os
import signal
import subprocess
import time
import episode_protocol as protocol
from parallel_state import assert_resumable, gpu_pairs, pending

RUN, CODE, CACHE = protocol.RUN, protocol.CODE, protocol.CACHE


def gpu_idle(pair):
    output = subprocess.check_output(['nvidia-smi', '-i', ','.join(map(str, pair)),
        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    rows = [list(map(int, line.split(','))) for line in output.splitlines()]
    assert len(rows) == 2
    return all(memory < 100 and utilization == 0 for memory, utilization in rows)


def command(item):
    result = protocol.command(item)
    for flag, gpu in zip(('--sim-gpu', '--policy-gpu'), item['gpu_pair']):
        result[result.index(flag) + 1] = str(gpu)
    return result


def save(state, active):
    state['active'] = [entry['item'] for entry in active.values()]
    state['current'] = state['active'][0] if state['active'] else None
    state['heartbeat_unix'] = time.time()
    protocol.save(state)


def check_sources(sources, assets, runners):
    for root, files in ((CODE, sources), (CACHE, assets), (RUN, runners)):
        for name, digest in files.items():
            assert protocol.sha(root / name) == digest, name


def start(item, pair, state, active):
    item.update(gpu_pair=list(pair), started_unix=time.time(),
        run_id=f"scratch200k-{item['method']}-to-{item['controller']}-g{item['group']}-e{item['repeat']}-20261119-r5-a{item['attempt']}")
    item['command'] = command(item)
    entry = dict(item=item)
    active[pair] = entry
    save(state, active)  # Reserve before spawning; crashes require explicit reconciliation.
    log = (RUN / (item['run_id'] + '.log')).open('x')
    try:
        process = subprocess.Popen(item['command'], cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env={**os.environ, 'PYTHONPATH': str(CODE)}, start_new_session=True)
    except BaseException:
        log.close()
        raise
    entry.update(process=process, log=log)
    item['pid'] = process.pid
    save(state, active)
    print('START', item['run_id'], pair, flush=True)


def poll(entry):
    item, process = entry['item'], entry['process']
    if process.poll() is not None:
        entry['log'].close()
        item['exit_code'] = process.returncode
        try:
            result = protocol.result_for(item)
        except (OSError, ValueError, AssertionError) as error:
            item['error'] = f'Invalid receipt: {error!r}'
            result = None
        if item['exit_code'] == 0 and result is not None:
            item.update(result)
            return 'completed'
        item.setdefault('error', 'Invalid or missing receipt; preserved logs require review')
        return 'failed'
    if entry.get('kill_at'):
        if time.time() >= entry['kill_at']:
            os.killpg(process.pid, signal.SIGKILL)
            entry['kill_at'] = time.time() + 20
        return None
    folder = CODE / 'cross_tracker/runs' / item['run_id']
    if folder.exists():
        entry.setdefault('simulation_observed_unix', time.time())
        logs = [RUN / (item['run_id'] + '.log'), folder / 'server.log', folder / 'simulation.log']
        newest = max(p.stat().st_mtime for p in logs if p.exists())
        stalled = time.time() - newest > 600
        overdue = time.time() - entry['simulation_observed_unix'] > 2400
        if stalled or overdue:
            item['error'] = 'No logs for 600s' if stalled else 'Episode exceeded 2400s infrastructure limit'
            assert os.getpgid(process.pid) == process.pid
            os.killpg(process.pid, signal.SIGTERM)
            entry['kill_at'] = time.time() + 20
    return None


def main():
    assert not os.environ.get('ISAAC_MULTI_IMAGE_SHM_NAME'), 'Each simulator must create its own image shared memory'
    lock = (RUN / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((RUN / 'prepare.json').read_text())['status'] == 'complete'
    if (RUN / 'batch.json').exists():
        state = json.loads((RUN / 'batch.json').read_text())
        assert_resumable(state, Path(f"/proc/{state['pid']}").exists())
    else:
        previous = json.loads((RUN / 'previous-batch.json').read_text())
        state = dict(completed=previous['completed'], failed=previous['failed'], active=[], current=None,
            total_episodes=480, started_unix=time.time(), protocol=previous['protocol'])
    state.update(pid=os.getpid(), status='running', platform='Baidu T2M-PLUS dv-e21ddf3a87')
    sources = json.loads((RUN / 'source-files.json').read_text())
    assets = json.loads((RUN / 'shared-eval-assets.json').read_text())
    runners = json.loads((RUN / 'runner-sources.json').read_text())
    check_sources(sources, assets, runners)
    active, cooldown = {}, {}
    tasks = list(protocol.tasks())
    try:
        while len(state['completed']) < 480:
            for pair, entry in list(active.items()):
                outcome = poll(entry)
                if outcome:
                    state[outcome].append(entry['item'])
                    del active[pair]
                    cooldown[pair] = time.time() + 10
                    print(outcome.upper(), entry['item']['run_id'], flush=True)
                    save(state, active)
            pairs = gpu_pairs(json.loads((RUN / 'enabled-pairs.json').read_text()))
            assert all(pair in pairs for pair in active), 'Drain a GPU pair before disabling it'
            for pair in pairs:
                if pair in active or time.time() < cooldown.get(pair, 0):
                    continue
                methods = {m for m in ('sonic', 'twist2', 'mix', 'cct') if (RUN / f'{m}-manifest.json').exists()}
                ready, blocked = pending(state, tasks, methods, lambda item: (RUN /
                    f"retry-approved-{item['method']}-{item['controller']}-{item['group']}-{item['repeat']}-{item['attempt']}").exists())
                state['attention'] = blocked
                try:
                    idle = gpu_idle(pair) if ready else False
                    state.setdefault('resource_errors', {}).pop(str(pair), None)
                except (subprocess.SubprocessError, ValueError, AssertionError) as error:
                    state.setdefault('resource_errors', {})[str(pair)] = repr(error)
                    idle = False
                if ready and idle:
                    check_sources(sources, assets, runners)
                    start(ready[0], pair, state, active)
            state['status'] = 'running' if active else ('needs_attention' if state.get('attention') else 'waiting_for_checkpoint_or_idle_gpus')
            save(state, active)
            time.sleep(5)
        state.update(status='completed', finished_unix=time.time())
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        save(state, active)


if __name__ == '__main__':
    main()
