"""Refresh BiGym experiment tables from configs, metrics and validation results.

Run from the repository: python scripts/experiment_ledger.py
Historical entries stay intact. Live entries use stable BG5090 experiment IDs.
The Excel workbook is a dated snapshot; these server-side tables are current.
"""
from pathlib import Path
import csv
import json
import statistics
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/experiments'


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def refresh():
    path = OUT / 'bigym2_ledger.json'
    data = json.loads(path.read_text())
    state_path = ROOT / '.setup/stability_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    evaluations = []
    for row in data['runs']:
        if not row['id'].startswith('BG5090-'):
            continue
        run = ROOT / row['run_dir']
        config_path = run / 'config.json'
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        cfg = config['main']
        row.update(seed=cfg['seed'], batch=cfg['batch_size'], micro_batch=cfg['micro_batch_size'],
                   schedule_steps=cfg['max_steps'], target_steps=cfg.get('stop_after') or cfg['max_steps'])
        records = {}
        metrics = run / 'metrics.jsonl'
        if metrics.exists():
            lines = metrics.read_text().splitlines()
            for index, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if index == len(lines)-1:
                        break  # writer may not have finished the last line
                    raise
                records[record['step']] = record
        if records:
            values = [records[s] for s in sorted(records)]
            last = values[-1]
            row.update(last_step=last['step'], motion_loss=last['motion_loss'],
                       max_motion_loss=max(x['motion_loss'] for x in values),
                       skipped_steps=last.get('skipped_steps'),
                       seconds_per_step=statistics.median(x['s_per_step'] for x in values[-5:]))
        if row['id'] in ['BG5090-001','BG5090-002']:
            check = 'check_1gpu' if row['id'].endswith('001') else 'check_8gpu'
            exit_path = ROOT / f'.setup/{check}.exit'
            row['status'] = '运行中'
            if exit_path.exists():
                row['status'] = '已完成' if exit_path.read_text().strip() == '0' else '运行失败'
            elif row.get('last_step') == 200 and check == 'check_1gpu':
                row['status'] = '评测中'
            memory_path = ROOT / f'.setup/{check}.memory.csv'
            if memory_path.exists():
                samples = list(csv.reader(memory_path.read_text().splitlines()))
                memory = [int(x[2].strip().split()[0]) for x in samples if len(x)==4]
                if memory:
                    row['sampled_memory_mib'] = max(memory)
        elif row['id'] in state:
            row.update({k:v for k,v in state[row['id']].items() if k in ['status','stage','next_action','reason','started_at','finished_at']})
        for result_path in sorted((run / 'val700000').glob('checkpoint_*.json')):
            result = json.loads(result_path.read_text())
            evaluations.append(dict(id=row['id'], task=row['task'], checkpoint=int(result_path.stem.split('_')[-1]),
                                    seed_base=result['seed_base'], episodes=result['episodes'],
                                    execution_frames=50, success_rate=result['success_rate'],
                                    role='流程检查' if row['id']=='BG5090-001' else '验证调参',
                                    source=str(result_path.relative_to(ROOT))))
    data['evaluations'] = evaluations
    data['updated_at'] = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    atomic_json(path, data)
    for name, rows in [('runs',data['runs']),('evaluations',evaluations),('baseline',data['baseline']),('recipes',data['recipes'])]:
        if not rows:
            continue
        target = OUT / f'bigym2_{name}.csv'
        temp = target.with_suffix('.tmp')
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with temp.open('w',newline='',encoding='utf-8-sig') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        temp.replace(target)
    lines = ['# BiGym 实验记录', '', '更新时间：'+data['updated_at'], '',
             '服务器训练期间由实验执行脚本每 30 秒刷新。Excel 是导出时的快照。空白表示尚无记录，不表示 0。', '',
             '| 实验 ID | 任务 | 阶段 | 状态 | 当前 / 目标步数 | 方案 | 运动损失 | 秒/步 | 下一步 |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in data['runs']:
        show = lambda v: '-' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v)
        cells=[r['id'],r['task'],r['stage'],r['status'],f"{show(r['last_step'])} / {r['target_steps']}",r['recipe'],show(r['motion_loss']),show(r['seconds_per_step']),r['next_action']]
        lines.append('| '+' | '.join(c.replace('|','/') for c in cells)+' |')
    lines += ['', '## 评测结果', '', '| 实验 ID | checkpoint | 验证种子起点 | 回合数 | 成功率 | 用途 |', '|---|---|---|---|---|---|']
    for e in evaluations:
        lines.append(f"| {e['id']} | {e['checkpoint']} | {e['seed_base']} | {e['episodes']} | {e['success_rate']:.1%} | {e['role']} |")
    lines += ['', '## 记录规则', '',
              '- 原始证据是各 run 的 config.json、metrics.jsonl、train.log 和 val700000 结果文件；汇总不覆盖这些证据。',
              '- 方案参数见 bigym2_recipes.csv；历史九任务结果见 bigym2_baseline.csv，来源为 docs/benchmarks/bigym2.md §6、§6b。',
              '- 历史第一套：26k–30k 五个 checkpoint，各 100 回合。第二套：80k–100k 五个 checkpoint，各 100 回合。中途测试集最高值只能参考。',
              '- 本轮只用验证种子 700000；官方种子 620000 保留到最终评测。历史官方结果与当前验证结果不直接作同口径提升结论。',
              '- 稳定组合：学习率减半、AdamW β2=0.95、尖峰跳过因子 5。种子 1000，两个任务各一组，不扫参数。',
              '- 发散判据沿用 scripts/diag/loss_spikes.py：2000 步后连续 10 个记录点（每 50 步记录）运动损失中位数 >0.5；单个尖峰不算发散。',
              '- 先训 30k 并评 25k/30k。未发散才继续同一 run 至 101k；发散则停。',
              '- 改参重跑创建新实验 ID 和新 run 目录，写明父实验、改动与原因；恢复训练沿用同一 ID，记录阶段切换。',
              '- 历史精确日志未迁移的字段留空；方案目录中的历史参数按文档与现有配置还原，并非重新核验历史 config.json。',
              '- 单卡短检查成功率 0 不是正式结果；本轮单种子不能估计发散概率或归因到单个超参数。',
              '- 不提交或推送 GitHub，除非用户另行批准。', '']
    (OUT/'README.md').write_text('\n'.join(lines))
    return data


if __name__ == '__main__':
    d=refresh()
    print('Updated',len(d['runs']),'runs,',len(d['evaluations']),'evaluations at',d['updated_at'])
