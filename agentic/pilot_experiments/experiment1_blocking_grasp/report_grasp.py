"""Summarize a set of blocking grasp runs: success by episode, attempts by label, contact and slip rates.

    python agentic/pilot_experiments/experiment1_blocking_grasp/report_grasp.py 'exp1-static-20261124-r1-*'
"""
import argparse
from collections import Counter
import fnmatch
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))  # agentic/scripts: paths.py and the launcher
from paths import RUNS


def summarize(results):
    """``results``: one finished grasp-result per episode. Rates use the plan's denominators."""
    attempts = [a for r in results for a in r['attempts']]
    held = [a for a in attempts if a['held']]
    wins = [r for r in results if r['success']]
    share = lambda part, whole: None if not whole else part / whole
    return {'episodes': len(results), 'successes': len(wins), 'success_rate': share(len(wins), len(results)),
            'first_attempt_successes': sum(r['first_attempt_success'] for r in results),
            'terminations': dict(Counter(r['termination'] for r in results)),
            'attempts': len(attempts), 'attempt_labels': dict(Counter(a['label'] for a in attempts)),
            'contact_rate': share(sum(a['touched'] for a in attempts), len(attempts)),
            'slip_rate': share(sum(a['label'] == 'slipped' for a in held), len(held)),
            'attempts_to_success': sorted(r['attempts_to_success'] for r in wins),
            'completion_s': sorted(round(r['completion_s'], 2) for r in wins),
            'attempts_per_episode': sorted(r['attempt_count'] for r in results)}


def load(pattern):
    results, unfinished = [], []
    for run in sorted(RUNS.iterdir()):
        if not fnmatch.fnmatch(run.name, pattern) or not run.is_dir():
            continue
        try:
            result = json.loads((run/'grasp-result.json').read_text())
        except (OSError, ValueError):
            result = {}
        if result.get('termination'):
            results.append({'run': run.name, **result})
        else:
            unfinished.append(run.name)
    return results, unfinished


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('pattern')
    p.add_argument('--json', action='store_true')
    a = p.parse_args()
    results, unfinished = load(a.pattern)
    summary = summarize(results)
    if a.json:
        print(json.dumps({'summary': summary, 'unfinished': unfinished,
                          'episodes': [{k: r[k] for k in ('run', 'seed', 'success', 'termination', 'episode_steps',
                                                           'attempt_count', 'completion_s')} |
                                       {'labels': [x['label'] for x in r['attempts']]} for r in results]}, indent=1))
    else:
        for r in results:
            labels = ' / '.join(x['label'] for x in r['attempts'])
            print(f"{r['run']}  seed {r['seed']}  {r['termination']:8s} steps {r['episode_steps']:4d}  {labels}")
        print(json.dumps(summary, indent=1, ensure_ascii=False))
        if unfinished:
            print('no verdict:', ', '.join(unfinished))
