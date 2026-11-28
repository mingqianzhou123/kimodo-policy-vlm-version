"""The per-episode record of the command loop (plan section 1 item 8, plus what link 3 needs).

Everything lives in ``<run>/loop/``; the launcher's ``run.json`` (layout id = seed, condition,
code version) and Arena's ``episode.json`` (final verdict) sit one level up. Every file is
written as the episode runs, so a crashed episode keeps what happened up to the crash.

  meta.json        task, condition, layout id, commander, planner and its settings, thresholds file
  scene.json       (simulator) objects at the start: names, poses, sizes, support surfaces, frame
  vlm.jsonl        one row per commander call: the raw reply text, the observation it saw (and the
                   image file), latency, tokens, the parsed command or the validation error
  commands.jsonl   two rows per executed command: kind 'start' (the command, its source call, the
                   commanded target in world terms, the planner's own plan, measured and reference
                   state at the start) and kind 'end' (measured and reference state at the end,
                   status, errors of measured and reference against the target)
  sent.jsonl       every chunk of 40-d reference sent to the tracker: time of its first row,
                   command index, the rows
  steps.jsonl      (simulator) one row per control step: measured root and wrists, hand command,
                   object poses, contact forces (each hand on the object, rest of the robot on it,
                   hip / pelvis bodies on anything, other robot bodies touching anything), Arena's
                   success check, fallen flag
  summary.json     why the loop ended, counts, budgets used (absent when Arena ended the episode first)
  verdict.json     (written by the judge afterwards) success, failed link, every check, reach margins and
                   the per-stage, per-link 'arrived' flags

Plan v3 section 2 item 9, field by field: episode id = run.json (seed / group seed, episode_seed,
repeat_idx) and meta.json; condition = run.json and meta.json; raw VLM output = vlm.jsonl 'raw'; command
targets = commands.jsonl 'target'; the 40-d sequence sent = sent.jsonl; root and wrist trajectories and
contact forces and the object trajectory = steps.jsonl; per-link arrival = verdict.json 'arrived';
outcome = episode.json; fall = steps.jsonl 'fallen' and episode.json failure_reason 'fall'.

The measured trajectory at full detail (joints, reference, decoder target) stays in the existing
``execution.jsonl`` one level up; ``steps.jsonl`` repeats what the judge needs so that the
judge reads ``loop/`` and ``episode.json`` only.
"""
import json
import time
from pathlib import Path

import numpy as np

FILES = ('meta.json', 'scene.json', 'vlm.jsonl', 'commands.jsonl', 'sent.jsonl', 'steps.jsonl', 'summary.json')


def plain(value):
    """JSON-safe copy: numpy arrays and scalars to lists and floats."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class EpisodeLog:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def write(self, name, value):
        (self.folder/name).write_text(json.dumps(plain(value), indent=1))

    def append(self, name, row):
        with (self.folder/name).open('a') as f:
            f.write(json.dumps(plain(row))+'\n')

    def meta(self, value):
        self.write('meta.json', {**value, 'written_unix': time.time()})

    def vlm(self, row):
        self.append('vlm.jsonl', row)

    def command(self, row):
        self.append('commands.jsonl', row)

    def sent(self, time_s, index, rows):
        self.append('sent.jsonl', {'time': time_s, 'command': index, 'rows': np.round(np.asarray(rows), 6)})

    def summary(self, value):
        self.write('summary.json', value)


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:   # a line cut by a crash
                break
    return rows


def load_episode(run):
    """Everything the judge reads, from one run folder (the one holding ``loop/``)."""
    run = Path(run)
    loop = run/'loop'
    read = lambda p: json.loads(p.read_text()) if p.is_file() else None
    commands = read_jsonl(loop/'commands.jsonl')
    starts = {r['index']: r for r in commands if r['kind'] == 'start'}
    ends = {r['index']: r for r in commands if r['kind'] == 'end'}
    merged = [{**starts[i], 'end': ends.get(i)} for i in sorted(starts)]
    return {'run': str(run), 'meta': read(loop/'meta.json'), 'scene': read(loop/'scene.json'),
            'vlm': read_jsonl(loop/'vlm.jsonl'), 'commands': merged, 'steps': read_jsonl(loop/'steps.jsonl'),
            'summary': read(loop/'summary.json'), 'episode': read(run/'episode.json'), 'manifest': read(run/'run.json')}
