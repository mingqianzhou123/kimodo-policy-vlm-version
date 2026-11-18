"""Four scratch finals, fixed 480 scenes; resume valid receipts and retry infrastructure errors."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import time

RUN = Path(__file__).resolve().parent
ROOT = Path('/root/workspace/humanoidVLA')
CACHE = ROOT / 'cross_tracker/data/cache'
CODE = RUN / 'code'
PY = CACHE / 'envs/policy/bin/python'
TASK = 'Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby'
COMBINATIONS = [('sonic', 'sonic'), ('twist2', 'twist2'), ('mix', 'sonic'),
                ('mix', 'twist2'), ('sonic', 'twist2'), ('twist2', 'sonic'),
                ('cct', 'sonic'), ('cct', 'twist2')]


def save(state):
    temp = RUN / 'batch.tmp'
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(RUN / 'batch.json')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def key(item):
    return tuple(item[n] for n in ('method', 'controller', 'group', 'repeat'))


def tasks():
    for group in range(3):
        for repeat in range(20):
            seed = int.from_bytes(hashlib.sha256(f'{TASK}|{group}|{repeat}'.encode()).digest()[-4:], 'big') & 0x7fffffff
            for method, controller in COMBINATIONS:
                yield dict(method=method, controller=controller, group=group, repeat=repeat, seed=seed)


def command(item):
    alias = {'sonic': 'sonic', 'twist2': 'twist2', 'mix': 'sonic-twist2', 'cct': 'sonic-twist2'}
    cmd = [str(PY), '-u', str(CODE / 'cross_tracker/scripts/evaluate_box.py'),
           alias[item['method']], item['controller'], '--run-id', item['run_id'],
           '--sim-gpu', '6', '--policy-gpu', '7', '--model-manifest', str(RUN / f"{item['method']}-manifest.json"),
           '--scene-seeds', str(item['seed']), '--group-seed', str(item['group']),
           '--repeat-index', str(item['repeat']), '--timing', 'original', '--fresh-sim',
           '--deterministic', '--env-preset', 'official', '--post-termination-steps', '10']
    if item['controller'] == 'sonic':
        cmd += ['--refresh-video', '--robot-usd', str(CACHE / 'sim-assets/arena/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd')]
    else:
        cmd += ['--sim-device', 'cpu', '--twist2-physics-timing', 'recorded']
    if item['method'] == 'cct':
        cmd += ['--cct']
    return cmd


def result_for(item):
    report = CODE / 'cross_tracker/runs' / item['run_id'] / 'run.json'
    result = json.loads(report.read_text()) if report.exists() else {}
    if result.get('status') == 'completed' and result.get('valid_for_success_rate'):
        assert len(result['episodes']) == 1 and result['episodes'][0]['seed'] == item['seed']
        assert result['group_seed'] == item['group'] and result['repeat_index'] == item['repeat']
        return result['episodes'][0]
    return None


def run_episode(item, state):
    item.update(command=command(item), started_unix=time.time())
    state.update(status='running', current=item)
    save(state)
    print('START', item['run_id'], flush=True)
    folder = CODE / 'cross_tracker/runs' / item['run_id']
    outer = RUN / f"{item['run_id']}.log"
    with outer.open('x') as log:
        process = subprocess.Popen(item['command'], cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'PYTHONPATH': str(CODE)}, start_new_session=True)
        item['pid'] = process.pid
        save(state)
        while process.poll() is None:
            state['heartbeat_unix'] = time.time()
            save(state)
            logs = [p for p in [outer, folder / 'server.log', folder / 'simulation.log'] if p.exists()]
            # Idle-GPU waits have no run folder and are not a stalled episode.
            stalled = folder.exists() and time.time() - max(p.stat().st_mtime for p in logs) > 600
            overdue = folder.exists() and time.time() - folder.stat().st_ctime > 2400
            if stalled or overdue:
                item['error'] = 'No episode logs for 600s' if stalled else 'Episode exceeded 2400s infrastructure limit'
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(5)
        item['exit_code'] = process.wait()
    result = result_for(item)
    if item['exit_code'] == 0 and result is not None:
        item.update(result)
        state['completed'].append(item)
        print('DONE', item['run_id'], item['success'], item['steps'], flush=True)
    else:
        item.setdefault('error', 'Invalid or missing receipt; inspect preserved logs')
        state['failed'].append(item)
        print('INFRA_FAILURE', item['run_id'], item['error'], flush=True)
    state['current'] = None
    save(state)
    time.sleep(10)


def main():
    assert json.loads((RUN / 'prepare.json').read_text())['status'] == 'complete'
    if (RUN / 'batch.json').exists():
        state = json.loads((RUN / 'batch.json').read_text())
        assert not Path(f"/proc/{state['pid']}").exists(), 'Previous scheduler still alive'
        assert state['current'] is None, 'Reconcile interrupted child before restart'
    else:
        state = dict(completed=json.loads((RUN / 'carried-completed.json').read_text()), failed=[],
                     current=None, total_episodes=480, started_unix=time.time(),
                     protocol=json.loads((RUN / 'protocol.json').read_text()))
    state.update(pid=os.getpid(), status='running', platform='Baidu T2M-PLUS dv-e21ddf3a87', sim_gpu=6, policy_gpu=7)
    sources = json.loads((RUN / 'source-files.json').read_text())
    assets = json.loads((RUN / 'shared-eval-assets.json').read_text())
    try:
        while len(state['completed']) < 480:
            done = {key(x) for x in state['completed']}
            assert len(done) == len(state['completed'])
            pending = [x for x in tasks() if key(x) not in done and (RUN / f"{x['method']}-manifest.json").exists()]
            if not pending:
                state.update(status='waiting_for_final_checkpoint', heartbeat_unix=time.time())
                save(state)
                time.sleep(15)
                continue
            item = pending[0]
            attempts = sum(key(x) == key(item) for x in state['failed'])
            if attempts >= 3 and not (RUN / f"retry-approved-{item['method']}-{item['controller']}-{item['group']}-{item['repeat']}-{attempts}").exists():
                state.update(status='needs_attention', heartbeat_unix=time.time(), attention=item)
                save(state)
                time.sleep(15)
                continue
            for name, digest in sources.items():
                assert sha(CODE / name) == digest, name
            for name, digest in assets.items():
                assert sha(CACHE / name) == digest, name
            item['run_id'] = f"scratch200k-{item['method']}-to-{item['controller']}-g{item['group']}-e{item['repeat']}-20261118-r4-a{attempts}"
            run_episode(item, state)
        state.update(status='completed', finished_unix=time.time())
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        save(state)


if __name__ == '__main__':
    main()
