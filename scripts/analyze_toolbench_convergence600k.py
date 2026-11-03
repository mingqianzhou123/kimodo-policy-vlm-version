"""Read-only600k metric analysis; no changes to the registered experiment."""
from pathlib import Path
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import statistics

os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'log/toolbench/tuning_20261027'
RUN=ROOT/'training/convergence_task_uniform_seed1000_e50_b600000'
KEYS=['loss','motion_loss','root_loss','body_loss','hand_loss','hand_state_loss','hand_transition_loss']


def main():
    os.nice(10)
    rows=[json.loads(line) for line in (RUN/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(10,600001,10))
    out=ROOT/'training_supervision/analysis_600k';out.mkdir(exist_ok=True)
    def window(lo,hi):
        chosen=[r for r in rows if lo<r['step']<=hi]
        return dict(start=lo,end=hi,records=len(chosen),**{k:statistics.mean(r[k] for r in chosen) for k in KEYS})
    windows=[window(i,i+20000) for i in range(0,600000,20000)]
    tail=windows[-1];previous=windows[-2]
    changes={k:tail[k]/previous[k]-1 for k in KEYS}
    endrows=[r for r in rows if r['step']>580000]
    blocks=[statistics.mean(r['loss'] for r in endrows[i:i+100]) for i in range(0,len(endrows),100)]
    budgets=[]
    state=json.loads((ROOT/'status.json').read_text())
    for b in [250000,300000,400000,500000,600000]:
        name=f'budget_fit_task_uniform_seed1000_e50_b{b}' if b==250000 else f'convergence_task_uniform_seed1000_e50_b{b}'
        values=[json.loads(s) for s in (ROOT/'training'/name/'metrics.jsonl').read_text().splitlines()]
        chosen=[r for r in values if b-20000<r['step']<=b]
        score=next((t for t in state['trials'] if t.get('stage')=='official_budget_comparison' and t['training_step']==b),{})
        budgets.append(dict(budget=b,loss=statistics.mean(r['loss'] for r in chosen),
                            primary_l12=score.get('primary_l12'),auxiliary_all18=score.get('auxiliary_all18')))
    weighted=dict(body=tail['body_loss'],root_x4=tail['root_loss']*4,
                  hand_state=tail['hand_state_loss'],hand_transition_x02=tail['hand_transition_loss']*.2)
    plateau=json.loads((RUN/'convergence.json').read_text())
    evidence=dict(updated_utc=datetime.now(timezone.utc).isoformat(),metrics_sha256=hashlib.sha256((RUN/'metrics.jsonl').read_bytes()).hexdigest(),
                  records=len(rows),registered_plateau=plateau,windows=windows,last20k_component_changes=changes,
                  weighted_last20k=weighted,weighted_last20k_share={k:v/tail['loss'] for k,v in weighted.items()},
                  last20k_logged_batch_sd=statistics.stdev(r['loss'] for r in endrows),
                  last20k_1kblock_sd=statistics.stdev(blocks),last20k_1kblock_means=blocks,budgets=budgets,
                  numerical_noise_interpretation='Descriptive variability; correlated logged training batches are not an independent validation set.')
    (out/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    with (out/'windows_20k.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(windows[0]));writer.writeheader();writer.writerows(windows)
    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    x=np.array([r['step'] for r in rows])/1000
    smooth=np.convolve([r['loss'] for r in rows],np.ones(100)/100,mode='valid')
    axes[0,0].plot(x,[r['loss'] for r in rows],alpha=.12,lw=.4)
    axes[0,0].plot(x[99:],smooth,lw=1,label='Trailing1k updates')
    axes[0,0].set_yscale('log');axes[0,0].set_title('600k scratch: total loss');axes[0,0].legend()
    for width,color in [(10000,'#175cd3'),(20000,'#d05b20')]:
        ws=[window(i,i+width) for i in range(480000,600000,width)]
        axes[0,1].plot([(w['start']+w['end'])/2000 for w in ws],[w['loss'] for w in ws],'-o',label=f'{width//1000}k disjoint mean',color=color)
    axes[0,1].set_title('Late loss: fixed disjoint windows');axes[0,1].legend()
    for key,weight in [('body_loss',1),('root_loss',4),('hand_state_loss',1),('hand_transition_loss',.2)]:
        axes[1,0].plot([w['end']/1000 for w in windows],[w[key]*weight for w in windows],label=f'{weight:g} x {key}')
    axes[1,0].set_yscale('log');axes[1,0].set_title('Actual weighted contributions (20k means)');axes[1,0].legend(fontsize=8)
    axes[1,1].plot([b['budget']/1000 for b in budgets],[b['loss'] for b in budgets],'-o')
    axes[1,1].set_title('Matched-budget scratch runs: final20k loss')
    for ax in axes.flat:ax.grid(alpha=.25);ax.set_xlabel('Training updates / budget (thousands)');ax.set_ylabel('Training loss')
    fig.suptitle('Near a training-loss plateau; registered strict criterion narrowly unmet. Not validation loss.')
    for ext in ['png','pdf']:fig.savefig(out/f'convergence_600k.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    lines=['# 600k训练收敛详细分析','',f"分析UTC：{evidence['updated_utc']}。源码分支kimodo-vla-lct；原始60000条记录完整覆盖step10–600000。",'',
           '**结论：600k训练已完成，训练loss在数值稳定的前提下接近平台；严格登记的停止标准尚未满足。不能据此宣布任务成功率已收敛。**','',
           '## 末期趋势与平台标准','',
           '| 区间 | 总loss均值 | 相对前20k窗口变化 |','| --- | ---: | ---: |']
    for i,w in enumerate(windows):
        if w['start']<480000:continue
        change=w['loss']/windows[i-1]['loss']-1
        lines.append(f"| {w['start']//1000}k–{w['end']//1000}k | {w['loss']:.8f} | {change:+.3%} |")
    lines += ['', '最后3个20k窗口的相邻变化为−1.104%、−0.234%；规则要求二者绝对值均<1%。第一段超出阈值约0.104个百分点，故plateau=false。'
              '最近10k均值0.010891，相对前10k下降0.219%；这是接近平台的证据，不能回头放宽登记阈值来宣称通过。', '',
              f"末20k单条batch loss标准差{evidence['last20k_logged_batch_sd']:.6f}，1k窗口均值标准差{evidence['last20k_1kblock_sd']:.6f}。"
              '最后两20k窗口均值差约0.0000256，小于短窗口波动量级；这里仅描述波动，不做独立同分布或显著性假设。'
              '最后一步loss0.012979、末1k均值0.011290高于末10k均值，不能凭这些短样本认定反弹或发散。', '',
              '## 分量与实际加权贡献','',
              '本配置total=4×root+body+hand_state+0.2×hand_transition。motion=4×root+body，hand=hand_state+0.2×hand_transition；避免把原始分量直接相加。', '',
              '| 原始分量 | 560k–580k均值 | 580k–600k均值 | 变化 |','| --- | ---: | ---: | ---: |']
    for key in KEYS[1:]:lines.append(f"| {key} | {previous[key]:.8f} | {tail[key]:.8f} | {changes[key]:+.3%} |")
    lines += ['', '| 加权项 | 末20k贡献 | 总loss占比 |','| --- | ---: | ---: |']
    for k,v in weighted.items():lines.append(f"| {k} | {v:.8f} | {v/tail['loss']:.2%} |")
    lines += ['', 'body约占70.76%，hand_state约19.45%，手部转换项约5.77%，加权root约4.03%。'
              '手部分支总loss仅下降约0.012%，转换项轻微上升约0.160%，与接近平台相符；body仍缓慢下降。'
              '不同分量的尺度、维度、掩码不同，占比不能用来断言任务失败原因或要求调整损失权重。', '',
              '## 学习率、梯度和数值健康','',
              '峰值LR为1e-5/3e-5/3e-5；warmup1000，cosine horizon600k，末步LR为1e-6/3e-6/3e-6。'
              '学习率降低会使曲线变平，因此当前结论只针对这一训练配方与学习率下限，不能证明其它学习率下也无收益。', '',
              '监督器核对60000条记录：NaN/Inf、完整JSON损坏、步数重复/逆序、LR曲线偏差均为0；当前源码指纹一致。'
              '最近1k裁剪前梯度均值control=0.203、hand=0.180，最大值0.332/0.401，最近10k超过1.0裁剪阈值比例均为0。'
              '全程hand峰值43.566在step10，control峰值6.961在step130760，均为裁剪前记录；末期没有持续爆炸或梯度归零证据。', '',
              '最近1k约0.187秒/步，数据加载占比约0.20%；每10k保存checkpoint，10k–600k共60个模型checkpoint已记录。'
              '本次源码恢复前后训练未重启，恢复后已重新通过登记源码核验。', '',
              '## 各预算终点对比','',
              '| 独立scratch预算/horizon | 末20k总loss | 官方L1/L2 | 官方全18 |','| --- | ---: | ---: | ---: |']
    for b in budgets:
        fmt=lambda v:'评估未完成' if v is None else f'{v:.2%}'
        lines.append(f"| {b['budget']//1000}k | {b['loss']:.8f} | {fmt(b['primary_l12'])} | {fmt(b['auxiliary_all18'])} |")
    improvement=budgets[-1]['loss']/budgets[-2]['loss']-1
    lines += ['', f'600k相对500k独立预算终点的末20k loss下降{abs(improvement):.2%}；模型拟合仍有改善。'
              '各预算是独立scratch、各自cosine horizon，不是500k模型继续100k；不能把跨预算差异解释为单条轨迹末期还以同样速度下降。', '',
              '## 判断与后续','',
              '当前无独立验证集loss，训练batch分布、扩散时刻与噪声持续变化；不能从训练loss判断有无过拟合，也不能保证L1/L2成功率提高。'
              '600k官方18×100评估完成后再比较500k的5.83%主指标；未完成评估不提供完整成功率结论。', '',
              '现有调度按登记规则，会在600k评估完成后继续700k从scratch训练，再做同口径官方评估；本分析没有修改规则或进程。'
              '从工程判断看600k已接近训练平台，从预登记规则看尚未通过；最终预算优先按L1/L2成功率、全18辅助选择。', '',
              '## 图和原始证据','',
              '- [收敛分析PNG](convergence_600k.png) / [PDF](convergence_600k.pdf)',
              '- [20k分窗原始统计CSV](windows_20k.csv) / [完整计算证据JSON](evidence.json)',
              '- [原监督六面板PNG](../runs/convergence_task_uniform_seed1000_e50_b600000/training_health.png)', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(out/'REPORT.md'),weighted_shares=evidence['weighted_last20k_share'],last20k_changes=changes),ensure_ascii=False))


if __name__=='__main__':main()
