"""The 60 official evaluation episodes of a HumanoidArena task, as Arena's evaluation suite defines them.

    python agentic/pilot_experiments2/experiment3/exp3_eval_episodes.py --task pp_box --out <file outside the repository>

Arena's test configs (`tasks/common_test_config/base_test/<task>_sonic_test.yaml`) set seeds [0, 1, 2]
and 20 repeats per seed; `script/eval_scripts/sonic/eval_vla_suite.py` runs each episode with
seed = its group seed and episode_seed = `_derive_episode_seed(task_name, group_seed, repeat_idx)`,
copied below (HumanoidArena 44a7222). It is the same function as experiment 2's
`exp2_replay.official_episode_seed` (branch claude/exp2-tracker-base, PR #20, not merged yet; once it is,
import that one instead of this copy); the test pins experiment 2's checked value for P&PBox (0, 0).

Restoring one (experiment 2's record, section 4): run Arena's evaluation with --seed <group> and
--episode_seed <derived seed> and the episode batch entry {seed, episode_seed, repeat_idx}; Arena's
own reset places the objects. That is exactly what the runner (`exp3_run.py --episodes`) passes, and
nothing else. Two caveats from that record: Arena runs all 60 in one persistent simulator, we run one
fresh process per episode (layouts depend only on the episode seed, not compared against an official
log); and determinism was shown for open-loop commands only. The command loop's inputs are on the
simulation clock (observation 'time' is env.sim.current_time, the simulator pauses while the commander
thinks; wall-clock time is only logged as latency), so closed-loop runs should repeat too - not tested.
"""
import argparse
import hashlib
import json
from pathlib import Path

TASK_NAMES = {'pp_box': 'Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody',
              'sit_sofa': 'Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody',
              'vision_navi': 'Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody'}
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
