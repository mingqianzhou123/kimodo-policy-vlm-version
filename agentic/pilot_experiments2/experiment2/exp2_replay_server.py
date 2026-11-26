"""Experiment 2 replay server: one official recording's actions on the measured clock (exp2_replay.py); CPU only."""
import argparse

from exp2_replay import EpisodeReference
from kimodo_vla.benchmarks.server import make_server

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', required=True)
p.add_argument('--episode', required=True)
p.add_argument('--port', type=int, required=True)
a = p.parse_args()
policy = EpisodeReference(a.run, a.episode)
make_server({('GET', '/health'): lambda _: (200, {'status': 'ok', 'native': False, 'source': 'official-raw'}),
             ('POST', '/reset'): lambda r: (200, policy.reset(r.get('seed'))),
             ('POST', '/infer'): lambda r: (200, policy(r))}, '127.0.0.1', a.port).serve_forever()
