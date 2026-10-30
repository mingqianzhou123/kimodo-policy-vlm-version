"""Append per-scenario percentages without changing the registered experiment code."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from toolbench_results import table_values


MARKER = '## 各阶段分场景成功率汇总'
SCENARIOS = ('G1BallMove', 'G1BallRetrieve', 'G1IceBreak')


def percentage(successes, episodes):
    return f'{100 * successes / episodes:.2f}'.rstrip('0').rstrip('.')


def scenario_table(trials):
    lines = [MARKER, '',
             '单元格为三个场景各自的成功率，固定顺序为 **BallMove / BallRetrieve / IceBreak**，'
             '**单位均为百分比（%）**；例如 `17/5/19` 表示 17%、5%、19%，不是成功次数。', '',
             '“评估回合/条件”表示每个场景、难度和 S/D 条件的回合数：50 对应全18共900回合，'
             '100 对应全18共1800回合。D 对应官方 R；screen 为开发筛选，confirm 为开发确认。'
             'L1/L2 与全18均值沿用上表的等权统计；不同评估阶段和环境种子需分别比较。', '',
             '| 策略 | 评估阶段 | 评估回合/条件 | S-L0 | S-L1 | S-L2 | D-L0 | D-L1 | D-L2 | L1/L2 均值 | 全18均值 |',
             '| --- | --- | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: |']
    for trial in trials:
        summary = json.loads(Path(trial['summary']).read_text())
        values = table_values(summary)
        assert abs(values[-2] - trial['primary_l12']) < 1e-12
        assert abs(values[-1] - trial['auxiliary_all18']) < 1e-12
        conditions = {r['env_id'].split('/')[-1]: r for r in summary['conditions']}
        episodes = summary['conditions'][0]['episodes']
        assert episodes == trial['episodes']
        cells = []
        for mode in ('S', 'R'):
            for level in range(3):
                group = [conditions[f'{scenario}-L{level}-{mode}'] for scenario in SCENARIOS]
                cells.append('/'.join(percentage(r['successes'], r['episodes']) for r in group))
        cells = [trial['name'], trial['stage'], str(episodes), *cells,
                 f'{values[-2]:.2%}', f'{values[-1]:.2%}']
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def update(root):
    destination = root / 'stage_table3.md'
    original = destination.read_text()
    trials = json.loads((root / 'status.json').read_text())['trials']
    # Follow exactly the rows already published by the registered report writer.
    published = [line.split('|')[1].strip() for line in original.split(MARKER)[0].splitlines()
                 if line.startswith('| ')][2:]
    by_name = {trial['name']: trial for trial in trials}
    ordered = [by_name[name] for name in published]
    content = original.split(MARKER)[0].rstrip() + '\n\n' + scenario_table(ordered)
    if content == original:
        return False
    # Retry next poll if the experiment published a new base table during rendering.
    if destination.read_text() != original:
        return False
    temporary = destination.with_name(destination.name + f'.scenarios.{os.getpid()}.tmp')
    temporary.write_text(content)
    temporary.replace(destination)
    print(f'Updated {destination}: {len(ordered)} scenario rows', flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=int, default=15)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root / 'scenario_report.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        update(root)
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
