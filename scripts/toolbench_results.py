"""Validated, equal-condition Table 3 reports for every search stage."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

COLUMNS = ['策略', 'S-L0', 'S-L1', 'S-L2', 'D-L0', 'D-L1', 'D-L2', 'L1/L2 均值', '全18均值']


def table_values(summary):
    rows = summary['conditions']
    assert summary['failed'] == 0 and len(rows) == 18
    expected = {f'{s}-L{l}-{m}' for s in ['G1BallMove', 'G1BallRetrieve', 'G1IceBreak']
                for l in range(3) for m in ['S', 'R']}
    assert {r['env_id'].split('/')[-1] for r in rows} == expected
    assert len({r['episodes'] for r in rows}) == 1 and rows[0]['episodes'] > 0
    for r in rows:
        assert r['status'] == 'completed' and 0 <= r['successes'] <= r['episodes']
        assert abs(r['success_rate'] - r['successes'] / r['episodes']) < 1e-12
    values = []
    for mode in ['S', 'R']:
        for level in range(3):
            chosen = [r for r in rows if r['env_id'].endswith(f'-L{level}-{mode}')]
            assert len(chosen) == 3
            values.append(sum(r['successes'] for r in chosen) / sum(r['episodes'] for r in chosen))
    primary = [r for r in rows if '-L1-' in r['env_id'] or '-L2-' in r['env_id']]
    values.extend([sum(r['successes'] for r in primary) / sum(r['episodes'] for r in primary),
                   sum(r['successes'] for r in rows) / sum(r['episodes'] for r in rows)])
    return values


def write_atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(value)
    tmp.replace(path)


def table_text(rows):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(COLUMNS)
    writer.writerows(rows)
    md = ['| ' + ' | '.join(COLUMNS) + ' |', '| --- | ' + ' | '.join(['---:'] * 8) + ' |']
    md += ['| ' + r[0] + ' | ' + ' | '.join(f'{v:.2%}' for v in r[1:]) + ' |' for r in rows]
    return output.getvalue(), '\n'.join(md) + '\n'


def write_stage_reports(root, trials):
    root = Path(root)
    rows, metadata, cohorts = [], [], {}
    for trial in trials:
        summary = json.loads(Path(trial['summary']).read_text())
        values = table_values(summary)
        assert abs(values[-2] - trial['primary_l12']) < 1e-12
        assert abs(values[-1] - trial['auxiliary_all18']) < 1e-12
        row = [trial['name'], *values]
        rows.append(row)
        csv_text, md = table_text([row])
        directory = Path(trial['summary']).parent
        write_atomic(directory / 'table3.csv', csv_text)
        write_atomic(directory / 'table3.md', md)
        config = json.loads((Path(trial['checkpoint']) / 'config.json').read_text())
        entry = dict(zip(COLUMNS, row))
        entry.update(stage=trial['stage'], training_step=trial['training_step'],
                     sampling=config['main']['sampling'], training_seed=config['main']['seed'],
                     max_steps=config['main']['max_steps'],global_batch=config['main']['batch_size'],
                     backbone_lr=config['training']['optimizer']['control_backbone_lr'],
                     adapter_lr=config['training']['optimizer']['control_adapter_lr'],
                     hand_lr=config['training']['optimizer']['hand_lr'],
                     decay_steps=config['training']['scheduler'].get('decay_steps'),
                     execution=trial['execution'], diffusion=trial['diffusion'],
                     seed_start=trial['seed_start'], episodes_per_condition=trial['episodes'],
                     primary_successes=trial['primary_successes'], primary_episodes=trial['primary_episodes'],
                     total_successes=trial['total_successes'], total_episodes=trial['total_episodes'],
                     checkpoint_sha256=trial['checkpoint_sha256'], summary=trial['summary'])
        metadata.append(entry)
        key = (trial['stage'], trial['seed_start'], trial['episodes'], trial['policy_seed'],
               trial['execution'], trial['diffusion'])
        cohorts.setdefault(key, []).append(trial)
    csv_text, md = table_text(rows)
    write_atomic(root / 'stage_table3.csv', csv_text)
    write_atomic(root / 'stage_table3.md',
                 '# 各阶段已核验成功率（Table 3）\n\n'
                 'D 对应官方 R；每个 S/D 难度列等权平均三个场景。CSV 数值为 0–1，表中为百分比。'
                 '只包含全 18 条件已核验评估；开发筛选、确认、正式测试和留出不可混为同一测试集。\n\n' + md)
    if metadata:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
        write_atomic(root / 'stage_comparison.csv', output.getvalue())
    analysis = ['# 阶段性比较分析', '',
                '主指标为 L1/L2 等权均值，辅助指标为全 18 等权均值。以下按相同评估阶段、环境种子、回合数和推理设置分组；'
                '点估计排名不等于统计显著。修订10按用户要求使用官方环境seed比较预算，'
                '该组单列为official_budget_comparison，属于预算选择成绩。', '']
    for key, candidates in cohorts.items():
        from toolbench_stage_analysis import unique
        candidates = unique(candidates)
        stage, start, episodes, policy_seed, execution, diffusion = key
        candidates = sorted(candidates, key=lambda r: (r['primary_l12'], r['auxiliary_all18']), reverse=True)
        best = candidates[0]
        analysis.extend([f'## {stage} · 环境种子 {start}–{start + episodes - 1} · E{execution}/D{diffusion}', '',
                         f"已完成 {len(candidates)} 项；当前点估计最高为 `{best['name']}`："
                         f"L1/L2 **{best['primary_l12']:.2%}**（{best['primary_successes']}/{best['primary_episodes']}），"
                         f"全18 **{best['auxiliary_all18']:.2%}**（{best['total_successes']}/{best['total_episodes']}）。", ''])
        if len(candidates) > 1:
            runner = candidates[1]
            analysis.extend([f"与次名 `{runner['name']}` 的 L1/L2 点估计差为 "
                             f"{100 * (best['primary_l12'] - runner['primary_l12']):.3f} 个百分点；"
                             '最终选择以该阶段规定的确认评估和复训规则为准。', ''])
    from toolbench_stage_analysis import detailed_analysis
    from toolbench_convergence_report import convergence_analysis
    write_atomic(root / 'stage_analysis.md', '\n'.join(analysis) + '\n\n' + convergence_analysis(root,trials)
                 + '\n\n' + detailed_analysis(root, trials))
