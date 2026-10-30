"""Evidence for the user-directed revision10 budget/convergence experiment."""
import json
from pathlib import Path


def convergence_analysis(root,trials):
    from toolbench_results import table_text, table_values
    root=Path(root);plan=json.loads((root/'plan.json').read_text())
    if plan.get('registered_revision',0)<10:return ''
    cfg=plan['convergence_experiment']
    out=['## 修订10：暂停推理调参，先检查训练预算与收敛', '',
         '按用户最新指令暂停执行窗口、扩散步数、历史匹配和追加训练seed等原后续选参。'
         '已完成结果保留；两项未完成开发评估取消，其部分成功数不进入统计。'
         '取消明细及回收空间见 audit/revision10/cancellation.json。', '',
         '250k基准为已完成的H250k预算匹配模型，不是旧H200k摸底轨迹上的250k权重。'
         '300k、400k、500k分别独立从scratch重训，完整复制该250k实际config，'
         '仅max_steps与cosine decay_steps同时改为各自预算。'
         '这比较的是“训练时长与对应完整cosine曲线”的组合，不能解释为同一条固定学习率轨迹的纯延长。', '',
         '- 每候选4卡DDP，microbatch32、global batch128；task_uniform；训练seed1000。',
         '- 峰值LR：backbone=1e-5、adapter=3e-5、hand=3e-5；warmup1000、cosine最低倍率0.1。',
         '- 完整模型、冻结基础编码器、hand_init_seed3407、容量和损失权重均与250k一致；不加载419h或任务策略checkpoint。',
         '- 推理固定E50/D10；官方环境种子10000–10099，每条件100回合、最大3000控制步，共1800回合。'
         '主指标为L1/L2的1200回合，辅助指标为全部1800回合。policy基础随机种子1000保持原实现。',
         '- 训练seed1000对齐公开ACT/DP配置；环境seed是10000–10099的序列，二者不是同一概念。'
         'hand分支固定初始化seed3407属于原模型定义，不为改变训练随机性而额外改动。',
         '- 本次按用户要求用官方场景种子比较预算，因此标为official_budget_comparison；'
         '这些成绩参与预算选择，不再称为未参与选参的独立最终测试。旧开发种子成绩分组保留。', '',
         '资源安排：先并行250k官方评估与300k训练；后续400k/500k训练及各自评估按两个4卡资源组排队。'
         '评估完成仿真即释放GPU，CPU官方凭据/视频校验与后续GPU任务重叠。', '',
         '### 同口径官方100回合 Table 3', '']
    rows=sorted([r for r in trials if r['stage']=='official_budget_comparison'],key=lambda r:r['training_step'])
    if rows:
        out.append(table_text([[f"Kimodo完整 {r['training_step']//1000}k / 每条件100回合",*table_values(json.loads(Path(r['summary']).read_text()))] for r in rows])[1])
        best=max(rows,key=lambda r:(r['primary_l12'],r['auxiliary_all18']))
        out += [f"当前已完成{len(rows)}个预算；最高点估计为{best['training_step']//1000}k："
                f"L1/L2={best['primary_l12']:.2%}（{best['primary_successes']}/{best['primary_episodes']}），"
                f"全18={best['auxiliary_all18']:.2%}（{best['total_successes']}/{best['total_episodes']}）。"
                '未完成预算不得参与排序；单训练seed结果不度量训练seed方差。', '']
        for a,b in zip(rows,rows[1:]):
            out += [f"{a['training_step']//1000}k→{b['training_step']//1000}k："
                    f"主指标变化{100*(b['primary_l12']-a['primary_l12']):+.3f}个百分点，"
                    f"辅助指标变化{100*(b['auxiliary_all18']-a['auxiliary_all18']):+.3f}个百分点。"
                    '应结合相同环境seed的配对成功/失败与分场景变化解释，不能只看总loss。', '']
    else:out += ['尚无本轮完整且通过18条件官方核验的成绩；等待250k官方评估，未完成结果不填作0%。', '']
    out += ['### loss记录、平台判断与后续预算', '',
            '训练每10步记录当步全DDP平均batch loss；PNG展示原始值与约1000步的后向平滑均值，'
            '同时检查分量loss、梯度裁剪前范数、实际LR、吞吐和checkpoint。'
            '独立曲线保存于 training_supervision/runs/<运行名>/training_health.png；'
            '所有保留轨迹对比为 training_supervision/loss_comparison.png。', '',
            '预先登记的操作性平台标准：训练末尾连续3个完整20k窗口，两个相邻窗口loss均值的相对变化绝对值均小于1%。'
            '该规则识别观察到的拟合平台，不证明数学收敛，也不证明成功率达到最优。'
            '300k/400k/500k三项按计划完成；若500k仍未满足平台规则，按100k增量增加独立scratch预算及同口径评估，'
            '直到观察到平台。数值异常由监督器另报，不把loss上升当成已收敛。', '',
            '| 预算/horizon | 已记录步数 | 最后3个20k loss均值 | 相邻变化 | 平台判断 |',
            '| --- | ---: | --- | --- | --- |']
    from train_toolbench_convergence import convergence
    paths=[Path(cfg['reference_checkpoint']).parent,*sorted((root/'training').glob('convergence_task_uniform_seed1000_e50_b*'))]
    for path in paths:
        metrics=path/'metrics.jsonl'
        if not metrics.exists():continue
        records=[]
        for line in metrics.read_text().splitlines():
            try:records.append(json.loads(line))
            except json.JSONDecodeError:continue
        if not records:continue
        c=convergence(records);budget=json.loads((path/'config.json').read_text())['main']['max_steps']
        means='/'.join('—' if v is None else f'{v:.6f}' for v in c['window_means'])
        changes='/'.join(f'{v:+.2%}' for v in c['relative_changes']) or '窗口不足'
        judgment='观察到平台' if c['plateau'] else ('继续下降/波动，尚未满足平台' if c['complete_windows'] else '窗口不足')
        out.append(f"| {budget//1000}k | {c['last_step']:,} | {means} | {changes} | {judgment} |")
    out += ['', '判断顺序：先核对训练数值、配置、学习率和官方评估身份；再检查预算增加是否仍降低loss；'
            '最后依据同口径L1/L2成功率优先、全18辅助选择已测预算。'
            '若loss继续下降但成功率不升，记录为拟合改善未转化为执行收益，不把最低loss直接当作最佳策略。', '']
    return '\n'.join(out)
