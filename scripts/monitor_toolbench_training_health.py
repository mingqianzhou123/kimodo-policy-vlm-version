"""Read-only CPU monitor, loss exports and standalone plots for ToolBench tuning."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import statistics
import time

os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


REPO=Path(__file__).resolve().parents[1]
COMPONENTS=('loss','motion_loss','root_loss','body_loss','hand_loss','hand_state_loss','hand_transition_loss')


def atomic(path,text):
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(text);tmp.replace(path)


def read_metrics(path):
    raw=path.read_bytes();lines=raw.splitlines();rows=[];bad=[]
    for i,line in enumerate(lines):
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A concurrently appended partial last line is not a corrupt record.
            if i!=len(lines)-1 or raw.endswith(b'\n'):bad.append(i+1)
    return rows,bad


def rolling(values,window=100):
    values=np.asarray(values,dtype=float)
    if len(values)<window:return np.array([])
    return np.convolve(values,np.ones(window)/window,mode='valid')


def recent(rows,steps):
    last=rows[-1]['step']
    return [r for r in rows if r['step']>last-steps]


def mean(rows,key):return statistics.mean(r[key] for r in rows)


def summarize(run,rows,bad,config,active):
    latest=rows[-1];tail=recent(rows,1000);last10k=recent(rows,10000)
    previous=[r for r in rows if latest['step']-20000<r['step']<=latest['step']-10000]
    nonfinite=[r['step'] for r in rows if any(not math.isfinite(v) for v in r.values() if isinstance(v,(int,float)))]
    steps=[r['step'] for r in rows];ordering=sum(b<=a for a,b in zip(steps,steps[1:]))
    mismatches=[];scheduler=config['training']['scheduler'];horizon=scheduler.get('decay_steps',config['main']['max_steps'])
    peak=config['training']['optimizer']
    for r in rows:
        step=r['step'];warm=scheduler['warmup_steps'];floor=scheduler['min_lr_ratio']
        factor=step/max(1,warm) if step<warm else floor+(1-floor)*.5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,horizon-warm))))
        if any(not math.isclose(r[k],peak[p]*factor,rel_tol=1e-8,abs_tol=1e-13)
               for k,p in [('lr_backbone','control_backbone_lr'),('lr_adapter','control_adapter_lr')]):mismatches.append(step)
    gradient=config['main']['gradient'];clip={k:gradient[v] for k,v in [('grad_control','grad_clip_norm'),('grad_hand','hand_grad_clip_norm')]}
    active_job=next((v for v in active.values() if v.get('job','').startswith('train_'+run.name+'_')),None)
    age=max(0,time.time()-(run/'metrics.jsonl').stat().st_mtime)
    early=[r for r in rows if r['step']<=1000] or rows[:100]
    latest_mean=mean(tail,'loss');mean10k=mean(last10k,'loss')
    growth=(mean10k/mean(previous,'loss')-1) if previous else None
    issues=[];warnings=[]
    if nonfinite:issues.append(f'Nonfinite metrics at {nonfinite[:5]}')
    if bad:issues.append(f'Malformed complete JSON lines: {bad[:5]}')
    if ordering:issues.append(f'{ordering} non-increasing metric steps')
    if mismatches:issues.append(f'{len(mismatches)} learning-rate schedule mismatches')
    if active_job and age>180:warnings.append(f'Active training metrics stale {age:.0f}s')
    if growth is not None and growth>.2:warnings.append(f'Last10k loss rose {growth:.1%} against preceding10k; inspect distribution and gradients')
    if active_job:
        for key,limit in clip.items():
            if abs(mean(tail,key))<1e-12:warnings.append(f'{key}: near-zero gradient throughout recent1k logged samples')
            if sum(r[key]>limit for r in last10k)/len(last10k)>.2:
                warnings.append(f'{key}: more than20% of recent10k logged samples exceed clipping threshold')
    checkpoints=sorted(int(p.name.split('_')[-1]) for p in run.glob('checkpoint_*') if (p/'training_state.pt').exists() and (p/'config.json').exists())
    if active_job and latest['step']-(max(checkpoints) if checkpoints else 0)>config['main']['checkpoint_every']+100:
        warnings.append('Training advanced beyond checkpoint interval without a complete checkpoint')
    loss_sum_error=max(abs(r['loss']-r['motion_loss']-r['hand_loss']) for r in rows)
    if loss_sum_error>1e-5:issues.append(f'Total != motion + hand; max absolute error {loss_sum_error}')
    return dict(name=run.name,active=bool(active_job),excluded=(run/'EXCLUDED_DUPLICATE.json').exists(),
        last_step=latest['step'],configured_max_steps=config['main']['max_steps'],horizon=horizon,
        training_seed=config['main']['seed'],sampling=config['main']['sampling'],records=len(rows),
        metrics_age_seconds=age,loss_latest=latest['loss'],loss_mean_last1k=latest_mean,
        component_mean_last1k={k:mean(tail,k) for k in COMPONENTS},loss_mean_first1k=mean(early,'loss'),
        loss_mean_last10k=mean10k,loss_change_vs_previous10k=growth,
        nonfinite_records=len(nonfinite),malformed_records=len(bad),nonincreasing_steps=ordering,
        lr_schedule_mismatches=len(mismatches),loss_sum_max_error=loss_sum_error,
        gradient={k:dict(max_all=max(r[k] for r in rows),max_all_step=max(rows,key=lambda r:r[k])['step'],
            mean_last1k=mean(tail,k),max_last1k=max(r[k] for r in tail),clip_norm=v,
            above_clip_fraction_all=sum(r[k]>v for r in rows)/len(rows),
            above_clip_fraction_last10k=sum(r[k]>v for r in last10k)/len(last10k)) for k,v in clip.items()},
        seconds_per_step_median_last1k=statistics.median(r['s_per_step'] for r in tail),
        data_seconds_per_step_median_last1k=statistics.median(r['data_s_per_step'] for r in tail),
        data_time_fraction_last1k=sum(r['data_s_per_step'] for r in tail)/sum(r['s_per_step'] for r in tail),
        gpu_peak_memory_gb=max(r['gpu_peak_memory_gb'] for r in rows),
        checkpoints=checkpoints,issues=issues,warnings=warnings)


def smooth_plot(ax,rows,key,label,raw=False):
    x=np.asarray([r['step'] for r in rows])/1000;y=np.asarray([r[key] for r in rows])
    if raw:ax.plot(x,y,alpha=.14,linewidth=.5,color='#567594',label='Raw logged batch loss')
    averaged=rolling(y)
    if len(averaged):ax.plot(x[99:],averaged,linewidth=1.5,label=label)
    else:ax.plot(x,y,label=label)


def save_figure(fig,directory,name):
    for extension in ['png','pdf']:
        target=directory/f'{name}.{extension}';tmp=directory/f'.{name}.{os.getpid()}.{extension}'
        fig.savefig(tmp,dpi=160,bbox_inches='tight');tmp.replace(target)
    plt.close(fig)


def plots(directory,runs,current,updated,comparison=True):
    rows=runs[current]['rows'];summary=runs[current]['summary'];tail=recent(rows,20000)
    fig,axes=plt.subplots(3,2,figsize=(15,13),constrained_layout=True)
    ax=axes[0,0];smooth_plot(ax,rows,'loss','Total: trailing100 logged points',raw=True);ax.set_yscale('log');ax.set_title('Full training loss (log scale)');ax.legend(fontsize=8)
    ax=axes[0,1];smooth_plot(ax,tail,'loss','Total: trailing100 logged points',raw=True);ax.set_title('Recent20k updates');ax.legend(fontsize=8)
    ax=axes[1,0]
    for k in COMPONENTS[1:]:smooth_plot(ax,rows,k,k)
    ax.set_yscale('log');ax.set_title('Logged component losses (not additive weights)');ax.legend(fontsize=8,ncol=2)
    ax=axes[1,1]
    for key in ['grad_control','grad_hand']:smooth_plot(ax,rows,key,key)
    ax.axhline(summary['gradient']['grad_control']['clip_norm'],color='#a94442',ls='--',label='Control clip threshold')
    ax.axhline(summary['gradient']['grad_hand']['clip_norm'],color='#d89b24',ls=':',label='Hand clip threshold')
    ax.set_yscale('log');ax.set_title('Pre-clipping gradient norm (smoothed)');ax.legend(fontsize=8)
    ax=axes[2,0]
    for key in ['lr_backbone','lr_adapter']:ax.plot([r['step']/1000 for r in rows],[r[key] for r in rows],label=key)
    ax.set_title(f'Actual learning rate; cosine horizon={summary["horizon"]//1000}k');ax.legend(fontsize=8)
    ax=axes[2,1]
    for key in ['s_per_step','data_s_per_step']:smooth_plot(ax,rows,key,key)
    ax.set_yscale('log');ax.set_title('Update and data-loading time (seconds)');ax.legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_xlabel('Training updates (thousands)')
    fig.suptitle(f'{current}\nUpdated {updated} | smoothing=100 records (~1000 updates)',fontsize=12)
    save_figure(fig,directory,'training_health')
    if not comparison:return
    fig,axes=plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
    for name,run in runs.items():
        info=run['summary']
        if info['excluded']:continue
        label=f'{info["sampling"]}, seed{info["training_seed"]}, H{info["horizon"]//1000}k'
        for ax in axes:smooth_plot(ax,run['rows'],'loss',label)
    axes[0].set_yscale('log');axes[0].set_title('All retained training trajectories')
    end=summary['last_step']/1000;axes[1].set_xlim(max(0,end-50000/1000),end);axes[1].set_title('Recent50k range of current run')
    for ax in axes:ax.set_xlabel('Training updates (thousands)');ax.set_ylabel('Logged total training loss');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Same-seed curves are not independent-seed replicates; cancelled duplicate200k excluded',fontsize=11)
    save_figure(fig,directory,'loss_comparison')


def report(root,summary):
    lines=['# ToolBench训练过程监督与loss记录','',f"更新时间（UTC）：{summary['updated']}。监测器只读，不调整模型、数据、训练/评估参数或进程。",'',
           '## 检查结论','',f"当前阶段：`{summary['phase']}`；调度器错误：`{summary['supervisor_error']}`。",'']
    for name,info in summary['runs'].items():
        if not info['active']:continue
        lines += [f"当前训练 `{name}`：**{info['last_step']:,}/{info['configured_max_steps']:,}**，cosine horizon={info['horizon']:,}。",'',
                  f"- 数值检查：非有限记录{info['nonfinite_records']}；损坏完整JSON记录{info['malformed_records']}；"
                  f"步数逆序/重复{info['nonincreasing_steps']}；学习率曲线不匹配{info['lr_schedule_mismatches']}。",
                  f"- loss：前1k均值{info['loss_mean_first1k']:.6f}，最近1k均值{info['loss_mean_last1k']:.6f}，最近10k均值{info['loss_mean_last10k']:.6f}。",
                  f"- 最近1k中位耗时{info['seconds_per_step_median_last1k']:.3f}秒/步，数据加载{info['data_seconds_per_step_median_last1k']:.6f}秒/步；"
                  f"数据耗时占比{info['data_time_fraction_last1k']:.2%}，当前证据不支持数据加载是主要瓶颈。",
                  f"- PyTorch累计峰值分配显存{info['gpu_peak_memory_gb']:.2f}GB（训练rank0记录，不等于整张卡显存占用）。",'']
        if info['loss_change_vs_previous10k'] is not None:lines += [f"最近10k相对前10k平均loss变化：{info['loss_change_vs_previous10k']:+.2%}。短窗口随机波动不等于发散或过拟合。",'']
        lines += ['| 分量 | 最近1k均值 |','| --- | ---: |']
        lines += [f'| {k} | {v:.6f} |' for k,v in info['component_mean_last1k'].items()]
        lines += ['', '| 梯度（裁剪前） | 全程峰值/所在步 | 最近1k均值 | 最近1k峰值 | 最近10k超过裁剪阈值比例 |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for key,g in info['gradient'].items():lines += [f"| {key} | {g['max_all']:.3f} / {g['max_all_step']} | {g['mean_last1k']:.4f} | {g['max_last1k']:.4f} | {g['above_clip_fraction_last10k']:.2%} |"]
        lines += ['', '梯度范数由clip_grad_norm_返回，是裁剪前值；实际更新按各自阈值裁剪。'
                  '早期手部分支大梯度需要关注，但若近期范数有限、裁剪比例低且loss稳定下降，不能单凭早期峰值判定梯度爆炸。'
                  '梯度为rank0记录，loss是各DDP rank的当前batch均值。', '']
    lines += ['## 所有训练轨迹','', '| 运行 | 状态 | 最新步数 | seed | cosine horizon | 最近1k平均loss | 非有限记录 | LR不匹配 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,info in summary['runs'].items():
        status='已取消重复/排除' if info['excluded'] else ('训练中' if info['active'] else '本段未训练')
        lines += [f"| {name} | {status} | {info['last_step']:,} | {info['training_seed']} | {info['horizon']:,} | {info['loss_mean_last1k']:.6f} | {info['nonfinite_records']} | {info['lr_schedule_mismatches']} |"]
    lines += ['', '## 图表与原始数据','',
              '- [当前训练诊断PNG](training_health.png) / [PDF](training_health.pdf)：全程与近期loss、各分量、梯度、学习率和速度。',
              '- [各轨迹loss对比PNG](loss_comparison.png) / [PDF](loss_comparison.pdf)：取消的重复200k不进入对比图。',
              '- [原始指标CSV](metrics_all_runs.csv)：保留每条已记录指标、运行名与排除标记。原始源文件仍为各training目录下metrics.jsonl。',
              '- [监督JSON](health_summary.json) / [检查历史JSONL](health_history.jsonl) / [自动刷新页面](index.html)。', '',
              '## 解释与监测边界','',
              '训练器每10步记录一次当步的batch loss，而不是这10步的平均loss。图中平滑线为100条记录的后向滑动均值，通常约1000训练步；原始总loss以淡线保留。'
              '各分量的定义和权重不同，root/body/hand_state/hand_transition不能直接相加；只检查已定义的total=motion+hand数值关系。', '',
              '连续的训练loss下降说明拟合目标仍在改善，不能证明任务成功率提高，也不能单凭这些训练数据判断没有过拟合。'
              '当前无独立验证集loss，主要泛化证据仍是stage_table3.md中的开发场景成功率及后续独立训练seed。'
              '不同采样、horizon的训练loss反映各自batch分布和学习率曲线，不能按最低loss选策略。', '',
              '本监测器检查全部已写出的指标，并核对登记源码指纹、最近日志和活动进程；120秒更新一次。'
              '它不直接读取所有未记录更新或所有rank的瞬时状态。训练器本身每步检查总loss及梯度非有限值。'
              '监测器只记录异常，不自动结束训练；发现异常需结合检查历史和原始日志处理。', '',
              '## 本次警报与系统快照','', '```json',json.dumps({k:summary[k] for k in ['issues','warnings','fingerprints_ok','processes','gpu','cpu','log_findings']},ensure_ascii=False,indent=2),'```','']
    return '\n'.join(lines)


def generate(root):
    output=root/'training_supervision';output.mkdir(exist_ok=True)
    state=json.loads((root/'status.json').read_text());runs={};csv_output=io.StringIO();writer=None
    for run in sorted((root/'training').iterdir()):
        if not (run/'metrics.jsonl').exists():continue
        rows,bad=read_metrics(run/'metrics.jsonl')
        if not rows:continue
        config=json.loads((run/'config.json').read_text());info=summarize(run,rows,bad,config,state.get('active',{}))
        runs[run.name]=dict(rows=rows,summary=info)
        for row in rows:
            item=dict(run=run.name,excluded=info['excluded'],seed=info['training_seed'],sampling=info['sampling'],cosine_horizon=info['horizon'],**row)
            if writer is None:writer=csv.DictWriter(csv_output,fieldnames=list(item));writer.writeheader()
            writer.writerow(item)
    updated=datetime.now(timezone.utc).isoformat();issues=[];warnings=[]
    for name,run in runs.items():
        if run['summary']['excluded']:continue
        issues += [f'{name}: {s}' for s in run['summary']['issues']]
        warnings += [f'{name}: {s}' for s in run['summary']['warnings']]
    if state.get('error'):issues.append('Supervisor error: '+state['error'])
    heartbeat_age=(datetime.now(timezone.utc)-datetime.fromisoformat(state['heartbeat'])).total_seconds()
    if heartbeat_age>120:warnings.append(f'Supervisor heartbeat stale {heartbeat_age:.0f}s')
    plan=json.loads((root/'plan.json').read_text());fingerprints=all(hashlib.sha256((REPO/f).read_bytes()).hexdigest()==h
        for key in ['core_source_sha256','protocol_source_sha256'] for f,h in plan[key].items())
    if not fingerprints:issues.append('Registered source fingerprints changed')
    processes={};findings={}
    for label,pid in [('supervisor',state['pid'])]+[(v['job'],v['pid']) for v in state.get('active',{}).values()]:
        proc=Path('/proc')/str(pid)
        try:
            processes[label]=dict(pid=pid,state=proc.joinpath('stat').read_text().split(') ')[1].split()[0])
            if processes[label]['state']=='Z':issues.append(label+': zombie process still registered active')
            if processes[label]['state']=='T':warnings.append(label+': process is stopped')
        except FileNotFoundError:processes[label]=dict(pid=pid,state='unavailable_in_this_process_namespace')
        log=root/'logs'/f'{label}.log'
        if log.exists():
            with log.open('rb') as f:f.seek(max(0,log.stat().st_size-128000));text=f.read().decode(errors='replace')
            findings[label]=[line for line in text.splitlines() if any(s in line.lower() for s in ['traceback','out of memory','nccl error','nonfinite','floatingpointerror'])][-10:]
            if findings[label]:warnings.append(label+': error-like text in recent log tail; inspect log_findings')
    import subprocess
    telemetry=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
    cpu=dict(loadavg=list(os.getloadavg()),cpu_count=os.cpu_count(),memory={})
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.split(':')[0] in ['MemTotal','MemAvailable','SwapTotal','SwapFree']:cpu['memory'][line.split(':')[0]]=line.split(':')[1].strip()
    summary=dict(updated=updated,phase=state['phase'],supervisor_error=state.get('error'),runs={k:v['summary'] for k,v in runs.items()},issues=issues,warnings=warnings,
        fingerprints_ok=fingerprints,processes=processes,gpu=telemetry.stdout.strip() or telemetry.stderr.strip(),gpu_query_returncode=telemetry.returncode,cpu=cpu,log_findings=findings)
    atomic(output/'metrics_all_runs.csv',csv_output.getvalue())
    current=next((n for n,r in runs.items() if r['summary']['active']),None)
    if current is None:current=max((n for n,r in runs.items() if not r['summary']['excluded']),key=lambda n:(root/'training'/n/'metrics.jsonl').stat().st_mtime)
    # Persist diagnostic evidence even if an abnormal curve cannot be rendered.
    atomic(output/'health_summary.json',json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    try:plots(output,runs,current,updated)
    except Exception as exc:issues.append(f'Plot generation failed: {type(exc).__name__}: {exc}')
    # Preserve each new budget's own diagnostics, including simultaneous runs.
    diagnostic_names=[]
    for name,run in runs.items():
        if not (name.startswith('convergence_') or name=='budget_fit_task_uniform_seed1000_e50_b250000'):continue
        diagnostic_names.append(name)
        destination=output/'runs'/name;destination.mkdir(parents=True,exist_ok=True)
        marker=destination/'last_plotted_step.json'
        previous=json.loads(marker.read_text()) if marker.exists() else {}
        if previous.get('step')!=run['summary']['last_step']:
            try:
                plots(destination,runs,name,updated,comparison=False)
                atomic(marker,json.dumps(dict(step=run['summary']['last_step'],updated=updated)))
            except Exception as exc:issues.append(f'{name} plot failed: {exc}')
    atomic(output/'health_summary.json',json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (output/'health_history.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in summary.items() if k not in ['gpu','cpu','log_findings']},ensure_ascii=False)+'\n')
    atomic(output/'training_supervision.md',report(root,summary))
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="120"><title>ToolBench训练监督</title><style>body{font:16px system-ui;margin:24px;color:#172033}img{max-width:100%}pre{white-space:pre-wrap}a{color:#175cd3}</style><h1>ToolBench训练监督与loss曲线</h1>'
    page+=f'<p>更新UTC：{html.escape(updated)} · 每120秒刷新 · <a href="training_supervision.md">详细分析</a> · <a href="metrics_all_runs.csv">原始CSV</a> · <a href="../live_progress.html">训练与评估进度</a></p>'
    page+='<pre>'+html.escape(json.dumps(dict(issues=issues,warnings=warnings),ensure_ascii=False,indent=2))+'</pre>'
    for image in ['training_health','loss_comparison']:page+=f'<p><a href="{image}.pdf">下载PDF</a></p><img src="{image}.png?v={updated}">'
    for name in diagnostic_names:
        page+=f'<h2>{html.escape(name)}</h2><p><a href="runs/{name}/training_health.png">独立训练曲线PNG</a></p><img src="runs/{name}/training_health.png?v={updated}">'
    atomic(output/'index.html',page+'</html>')
    print(updated,'active',current,'step',runs[current]['summary']['last_step'],'issues',len(issues),'warnings',len(warnings),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=int,default=120);a=p.parse_args()
    os.nice(10);root=a.root.resolve();lock=(root/'training_health.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        generate(root)
        if not a.watch:return
        time.sleep(a.interval)


if __name__=='__main__':main()
