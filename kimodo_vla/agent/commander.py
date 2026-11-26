"""Where commands come from: a scripted sequence (the acceptance test) or a VLM.

A commander has ``needs_image`` and ``next(observation, image, error)``, which returns one record:
``{'source': ..., 'raw': <reply text or the scripted object>, 'command': <validated dict> or None,
'error': <code: message> or None, ...}``. ``observation`` is the dict ``observation_text`` renders;
``error`` is the validation error of the previous reply, if that one was refused.

The VLM's prompt is the template in the frozen VLM configuration page (``load_prompt``): its first
fenced block tagged ``prompt``, with ``{observation}`` replaced by the observation JSON and
``{task}`` by the task instruction, ``{error}`` by the refusal note (empty when there is none).
"""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

from .commands import CommandError, parse, validate


def rounded(value, digits=3):
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v, digits) for v in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    return value


def observation_text(observation):
    return json.dumps(rounded(observation), separators=(', ', ': '))


def load_prompt(page):
    text = Path(page).read_text()
    match = re.search(r'^```prompt\n(.*?)^```', text, flags=re.S | re.M)
    if not match:
        raise ValueError(f'No ```prompt block in {page}')
    return match.group(1)


def fill(template, observation, error=None):
    note = '' if not error else f'Your previous reply was refused ({error}). Reply with one valid command.\n'
    task = observation.get('task', '')
    shown = {k: v for k, v in observation.items() if k != 'task'}
    return template.replace('{task}', task).replace('{observation}', observation_text(shown)).replace('{error}', note)


class ScriptedCommander:
    """Commands from a Python script: ``script.next(observation)`` returns a command dict, a JSON
    string (parsed like a model reply), or None to end. Scripts live with their experiment."""
    needs_image = False

    def __init__(self, script, limits=None):
        self.script, self.limits = script, limits

    @classmethod
    def from_file(cls, path, name, options=None, limits=None):
        spec = importlib.util.spec_from_file_location(f'scripted_{Path(path).stem}', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return cls(getattr(module, name)(**(options or {})), limits)

    def next(self, observation, image=None, error=None):
        raw = self.script.next(observation)
        record = {'source': 'scripted', 'raw': raw, 'command': None, 'error': None}
        if raw is None:
            record['command'] = {'type': 'done', 'reason': 'script finished'}
            return record
        try:
            pelvis = observation['robot']['pelvis']
            record['command'] = parse(raw, self.limits, pelvis) if isinstance(raw, str) else validate(raw, self.limits, pelvis)
        except CommandError as exc:
            record['error'] = str(exc)
        return record


class VLMCommander:
    """One model reply per command. ``backend`` has ``generate(image, prompt, max_tokens)`` (``vlm.backends``)."""
    needs_image = True

    def __init__(self, backend, template, limits=None, max_tokens=256):
        self.backend, self.template, self.limits, self.max_tokens = backend, template, limits, int(max_tokens)

    def next(self, observation, image=None, error=None):
        prompt = fill(self.template, observation, error)
        reply = self.backend.generate(image, prompt, self.max_tokens)
        record = {'source': getattr(self.backend, 'name', 'vlm'), 'raw': reply.get('text', ''), 'command': None, 'error': None,
                  **{k: v for k, v in reply.items() if k not in ('text', 'prompt', 'response')}}
        record['prompt_chars'] = len(prompt)
        try:
            record['command'] = parse(record['raw'], self.limits, observation['robot']['pelvis'])
        except CommandError as exc:
            record['error'] = str(exc)
        return record
