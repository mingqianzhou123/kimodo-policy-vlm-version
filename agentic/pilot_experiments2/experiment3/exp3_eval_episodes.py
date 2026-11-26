"""The 60 official evaluation episodes of a HumanoidArena task, as Arena's evaluation suite defines them.

    python agentic/pilot_experiments2/experiment3/exp3_eval_episodes.py --task pp_box --out <file outside the repository>

Arena's test configs (`tasks/common_test_config/base_test/<task>_sonic_test.yaml`) set seeds [0, 1, 2]
and 20 repeats per seed; `script/eval_scripts/sonic/eval_vla_suite.py` runs each episode with
seed = its group seed and episode_seed = `_derive_episode_seed(task_name, group_seed, repeat_idx)`,
computed by experiment 2's `exp2_replay.official_episode_seed` (its record, section 4).

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
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiment2'))
from exp2_replay import OFFICIAL_GROUPS as GROUP_SEEDS, OFFICIAL_REPEATS as REPEATS, official_episode_seed  # noqa: E402

TASK_NAMES = {'pp_box': 'Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody',
              'sit_sofa': 'Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody',
              'vision_navi': 'Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody'}
derive_episode_seed = official_episode_seed


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
