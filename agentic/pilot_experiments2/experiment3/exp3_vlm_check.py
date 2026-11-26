"""One commander call outside the simulator: the real prompt, an observation from a recorded episode, one image.

    python agentic/pilot_experiments2/experiment3/exp3_vlm_check.py --commander qwen --vlm-model <weights> --episode <run folder>
    python agentic/pilot_experiments2/experiment3/exp3_vlm_check.py --commander gpt --episode <run folder>   # one paid call

The observation is rebuilt from the episode's first command start (robot state and scene truth);
the image is that episode's first capture if it has one, else a grey placeholder (said so in the
output). Prints the raw reply and whether it parses into a valid command, and appends the record to
--out (outside the repository). For a hosted model this is the single connectivity check; anything
beyond it needs the user's go-ahead.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'scripts'))
import paths  # noqa: F401  (puts this checkout first on sys.path)

from kimodo_vla.agent.commander import VLMCommander, load_prompt
from kimodo_vla.agent.episode_log import load_episode
from kimodo_vla.oracle.geometry import SIDES


def observation_from(episode, task_text):
    first = episode['commands'][0] if episode['commands'] else None
    meta = episode['meta'] or {}
    truth = meta.get('scene_truth_at_start') or {'objects': (episode['scene'] or {}).get('objects', [])}
    state = first['measured'] if first else None
    robot = None if state is None else {
        'pelvis': state['root'], 'heading_deg': state['yaw_deg'], 'torso_pitch_deg': state['torso_pitch_deg'],
        'wrists': {s: {k: state['wrists'][s][k] for k in ('position', 'palm', 'fingers')} for s in SIDES},
        'hands': {s: 'open' for s in SIDES}}
    return {'task': task_text, 'time_s': 0., 'budget': {'commands_left': 40, 'time_left_s': None}, 'robot': robot,
            'objects': truth.get('objects', []), 'last_command': None, 'recent_commands': []}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--commander', required=True)
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--vlm-model', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    config = json.loads((HERE/'exp3_harness.json').read_text())
    spec = config['commanders'][a.commander]
    if spec['kind'] != 'vlm':
        raise SystemExit('Not a VLM commander')
    episode = load_episode(a.episode)
    task = (episode['meta'] or {}).get('task')
    observation = observation_from(episode, config['tasks'][task]['task_text'])
    from PIL import Image
    captures = sorted((a.episode/'loop').glob('capture-*.npz'))
    image = (Image.fromarray(np.load(captures[0])['rgb'].astype(np.uint8)) if captures
             else Image.new('RGB', (640, 480), (128, 128, 128)))
    if spec['backend'] == 'qwen':
        from kimodo_vla.vlm.backends import QwenBackend
        backend = QwenBackend(a.vlm_model)
    else:
        from kimodo_vla.vlm.backends import API_BACKENDS
        backend = API_BACKENDS[spec['backend']](spec['api'])
    commander = VLMCommander(backend, load_prompt(HERE/spec['page']), config.get('limits'), spec.get('max_tokens', 256))
    record = commander.next(observation, image)
    record.update(image=str(captures[0]) if captures else 'grey placeholder', episode=str(a.episode), commander=a.commander)
    print('RAW:', record['raw'])
    print('COMMAND:', record['command'], 'ERROR:', record['error'], 'LATENCY_S:', round(record.get('latency_s', 0.), 2))
    if a.out:
        with a.out.open('a') as f:
            f.write(json.dumps(record, default=str)+'\n')


if __name__ == '__main__':
    main()
