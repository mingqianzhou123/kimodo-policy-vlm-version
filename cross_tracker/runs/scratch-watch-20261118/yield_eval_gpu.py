"""Temporarily yield the verified idle GPU utility when R5 has waiting work."""
import fcntl
import json
import sys
import time

from await_parallel_handoff import RUN, WATCH, scheduler_alive, yield_idle_utility
from monitor_parallel import worker
from parallel_state import gpu_pairs, pending


def blocked_pairs(batch, pairs, ready):
    active = {tuple(item['gpu_pair']): item for item in batch['active']}
    for pair in pairs:
        if pair in active:
            try:
                status = worker(RUN, active[pair])
            except FileNotFoundError:
                continue
            if status['waiting_for_gpu']:
                yield pair
        elif ready:
            yield pair


def main():
    sys.path.append(str(RUN))
    from episode_protocol import tasks

    lock = (WATCH / 'yield-eval-gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {}
    scenes = list(tasks())
    print('WATCHING_R5_GPU_WAITS', flush=True)
    while True:
        batch = json.loads((RUN / 'batch.json').read_text())
        if batch['status'] in ('completed', 'failed') or not scheduler_alive():
            return
        now = time.time()
        heartbeat_fresh = now - batch['heartbeat_unix'] < 45
        cooldown_elapsed = now - state.get('last_idle_utility_pause', 0) >= 300
        if heartbeat_fresh and cooldown_elapsed:
            pairs = gpu_pairs(json.loads((RUN / 'enabled-pairs.json').read_text()))
            methods = {method for method in ('sonic', 'twist2', 'mix', 'cct')
                       if (RUN / f'{method}-manifest.json').exists()}
            ready, _ = pending(
                batch, scenes, methods,
                lambda item: (RUN / (
                    f"retry-approved-{item['method']}-{item['controller']}-"
                    f"{item['group']}-{item['repeat']}-{item['attempt']}"
                )).exists(),
            )
            for pair in blocked_pairs(batch, pairs, ready):
                if yield_idle_utility(pair, state):
                    print('TEMPORARY_GPU_YIELD', json.dumps(state['temporary_yields'][-1]), flush=True)
                    break
        time.sleep(15)


if __name__ == '__main__':
    main()
