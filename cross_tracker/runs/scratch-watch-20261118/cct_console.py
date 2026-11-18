"""Display CCT training progress without exiting on a partially written JSONL tail."""
from pathlib import Path
import json
import time

TRAINING = Path('/data/NTU_slab/mingqianzhou/humanoid_vla/kimodo-policy-vlm-version/'
                'cross_tracker/runs/box-aligned-cct-scratch-s42-20261118-r1/training')


def latest_metric(text):
    lines = text.splitlines()
    for line in reversed(lines[-20:]):
        try:
            value = json.loads(line)
            if isinstance(value, dict) and 'step' in value and 's_per_step' in value:
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError('No complete metric in the last 20 lines')


def main():
    while True:
        try:
            path = TRAINING / 'metrics.jsonl'
            value = latest_metric(path.read_text())
            print('\033[2J\033[H', time.strftime('%F %T'), 'CCT scratch', value['step'],
                  '/200000', 'metric_age_s', round(time.time() - path.stat().st_mtime),
                  's_per_step', round(value['s_per_step'], 3),
                  'final_exists', (TRAINING / 'checkpoint_200000/training_state.pt').is_file(),
                  flush=True)
        except (OSError, ValueError, KeyError) as error:
            print(time.strftime('%F %T'), 'MONITOR_READ_ERROR', repr(error), flush=True)
        time.sleep(60)


if __name__ == '__main__':
    main()
