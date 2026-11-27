"""P&PBox, both readings of "placed" for every run (no rerun needed whichever the team adopts):

  Arena       the time Arena's own position check first passes, and whether both hands were still commanded
              closed then (Arena counts a box still held in the region)
  released    the box 2 s and 5 s after the hands first open after the grasp: Arena's check over +-10 logged steps
              around that moment, box height, and both hands off the box

    python agentic/pilot_experiments2/experiment3/exp3_release_check.py '<run-folder glob under $KVLA_AGENTIC_RUNS>' ...

Needs runs played with ``run_oracle.py --after-success-steps`` (otherwise Arena ends the episode at its first success).
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'agentic/scripts'))
from paths import RUNS  # noqa: E402


def readings(run):
    loop = Path(run)/'loop'
    commands = [json.loads(line) for line in (loop/'commands.jsonl').read_text().splitlines() if line.strip()]
    steps = [json.loads(line) for line in (loop/'steps.jsonl').read_text().splitlines() if line.strip()]
    times = np.array([r['time'] for r in steps])
    box = lambda r: next(o for o in r['objects'] if o['name'] == 'box')['position']
    first = next((r for r in steps if r.get('arena_success')), None)
    starts = [c for c in commands if c['kind'] == 'start']
    closed_at = next((c['time'] for c in starts if c['type'] == 'hand' and c['command'].get('left') == 'close'), None)
    opened = next((c['time'] for c in starts if c['type'] == 'hand' and c['command'].get('left') == 'open'
                   and closed_at is not None and c['time'] > closed_at), None)

    def after(seconds):
        if opened is None:
            return None
        i = int(np.searchsorted(times, opened+seconds))
        if i >= len(steps):
            return None
        window = [r for r in steps[max(0, i-10):i+10] if r.get('arena_success') is not None]
        return {'on_shelf': bool(window and all(r['arena_success'] for r in window)), 'z': round(box(steps[i])[2], 3),
                'hands_off': bool(steps[i]['box_contact']['left_n'] < 1 and steps[i]['box_contact']['right_n'] < 1)}
    return {'run': Path(run).name, 'arena_first_s': None if first is None else round(first['time'], 1),
            'held_at_arena_first': None if first is None else bool(all(first['hands_closed'])),
            'opened_s': None if opened is None else round(opened, 1), 'after_2s': after(2.), 'after_5s': after(5.),
            'end_z': round(box(steps[-1])[2], 3)}


def main():
    for pattern in sys.argv[1:]:
        for run in sorted(glob.glob(str(RUNS/pattern))):
            if os.path.isdir(run):
                print(json.dumps(readings(run)))


if __name__ == '__main__':
    main()
