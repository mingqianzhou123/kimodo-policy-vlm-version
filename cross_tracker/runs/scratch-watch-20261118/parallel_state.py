"""Reservations for the same fixed 480 episodes on disjoint GPU pairs."""
from collections import Counter


def key(item):
    return tuple(item[n] for n in ('method', 'controller', 'group', 'repeat'))


def gpu_pairs(value):
    assert isinstance(value, list) and value
    pairs = [tuple(pair) for pair in value]
    assert all(pair in ((6, 7), (0, 1), (2, 3)) for pair in pairs)
    devices = [gpu for pair in pairs for gpu in pair]
    assert len(set(devices)) == len(devices), 'GPU pairs must not overlap'
    return pairs


def pending(state, tasks, available_methods, approved):
    done = [key(item) for item in state['completed']]
    reserved = [key(item) for item in state.get('active', [])]
    assert len(set(done + reserved)) == len(done + reserved), 'Duplicate episode reservation'
    attempts = Counter(key(item) for item in state['failed'])
    occupied = set(done + reserved)
    ready, blocked = [], []
    for task in tasks:
        if key(task) in occupied or task['method'] not in available_methods:
            continue
        item = dict(task, attempt=attempts[key(task)])
        if item['attempt'] >= 3 and not approved(item):
            blocked.append(item)
        else:
            ready.append(item)
    return ready, blocked


def assert_resumable(state, previous_alive):
    assert not previous_alive, 'Previous scheduler still alive'
    assert not state.get('active') and state.get('current') is None, 'Reconcile interrupted children first'
