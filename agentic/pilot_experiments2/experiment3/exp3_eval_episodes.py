"""The 60 official evaluation episodes of a HumanoidArena task, as Arena's evaluation suite defines them.

    python agentic/pilot_experiments2/experiment3/exp3_eval_episodes.py --task pp_box --out <file outside the repository>

Arena's test configs (`tasks/common_test_config/base_test/<task>_sonic_test.yaml`) set seeds [0, 1, 2]
and 20 repeats per seed; `script/eval_scripts/sonic/eval_vla_suite.py` runs each episode with
seed = its group seed and episode_seed = `_derive_episode_seed(task_name, group_seed, repeat_idx)`,
copied below (HumanoidArena 44a7222). This is the list only. Whether anything beyond these two
numbers is needed to restore an evaluation episode exactly is experiment 2's record to settle; the
runner (`exp3_run.py --episodes`) passes exactly these two numbers and nothing else.
"""
import argparse
import hashlib
import json
from pathlib import Path

TASK_NAMES = {'pp_box': 'Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody',
              'sit_sofa': 'Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody',
              'vision_navi': 'Isaac-Move-VisionNavi-G129-Dex3-Wholebody'}
GROUP_SEEDS, REPEATS = (0, 1, 2), 20


def derive_episode_seed(task_name, group_seed, repeat_idx):
    """Arena's `_derive_episode_seed` (eval_vla_suite.py), unchanged."""
    payload = f'{task_name}|{int(group_seed)}|{int(repeat_idx)}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[-4:], 'big', signed=False) & 0x7FFFFFFF


def official_episodes(task):
    name = TASK_NAMES[task]
    return [{'task': task, 'task_name': name, 'group_seed': g, 'repeat_idx': r,
             'episode_seed': derive_episode_seed(name, g, r)} for g in GROUP_SEEDS for r in range(REPEATS)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True, choices=sorted(TASK_NAMES))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(json.dumps(official_episodes(a.task), indent=1))
    print(f'{a.out}: {len(official_episodes(a.task))} episodes')


if __name__ == '__main__':
    main()
