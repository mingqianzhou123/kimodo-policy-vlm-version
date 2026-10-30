"""Read-only tuning dashboard. Never changes training, evaluation or selection files."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import html
import json
import os
from pathlib import Path
import statistics
import time
from toolbench_results import COLUMNS, table_values


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def tail_records(path):
    try:
        with path.open('rb') as f:
            f.seek(max(0,path.stat().st_size-200_000))
            lines=f.read().splitlines()
    except FileNotFoundError:return []
    rows=[]
    for line in lines:
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    return rows[-100:]


def evaluation_progress(out):
    settings=read(out/'evaluation_settings.json',{})
    if not settings:return None
    target=settings['episodes'];start=settings['seed_start'];rows=[]
    for condition in settings['conditions']:
        directory=out/condition
        receipts=sorted(directory.glob('run-*/benchmark_result.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        result=next((r for p in receipts if (r:=read(p,{})).get('status')=='validated'
            and r.get('policy_info',{}).get('checkpoint_sha256')==settings['checkpoint_sha256']
            and r.get('seeds')==list(range(start,start+target))),None)
        if result:
            n,success=result['episodes'],result['successes'];validated=True
        else:
            runs=sorted(directory.glob('run-*'),key=lambda p:p.stat().st_mtime,reverse=True)
            episodes={}
            if runs:
                for p in runs[0].glob('summaries/episode_*.json'):
                    value=read(p)
                    try:seed=int(p.stem.removeprefix('episode_'))
                    except ValueError:continue
                    if value is not None and start<=seed<start+target:episodes[seed]=value
            n=len(episodes);success=sum(bool(v['success']) for v in episodes.values());validated=False
        rows.append(dict(condition=condition,episodes=n,successes=success,rate=success/n if n else None,validated=validated))
    primary=[r for r in rows if '-L1-' in r['condition'] or '-L2-' in r['condition']]
    covered=sum(r['episodes']>0 for r in primary)
    return dict(name=out.name,target_per_condition=target,total=sum(r['episodes'] for r in rows),
        target_total=target*len(rows),validated_conditions=sum(r['validated'] for r in rows),
        primary_coverage=covered,primary_mean=sum(r['rate'] for r in primary)/12 if covered==12 else None,
        rows=rows)


def collect(root):
    state=read(root/'status.json',{});active=state.get('active',{});training=[];evaluations=[]
    for run in sorted((root/'training').glob('*')):
        records=tail_records(run/'metrics.jsonl')
        if not records:continue
        latest=records[-1];job=next(((slot,v) for slot,v in active.items()
            if v.get('job','').startswith('train_'+run.name+'_')),None)
        target=int(job[1]['job'].rsplit('_',1)[1]) if job else None
        speed=statistics.median(v['s_per_step'] for v in records)
        training.append(dict(name=run.name,step=latest['step'],target=target,active=job is not None,
            gpus=['0,1,2,3','4,5,6,7'][int(job[0])] if job else None,seconds_per_step=speed,
            remaining_seconds=max(0,target-latest['step'])*speed if target else None,
            latest_loss=latest['loss'],recent_mean_loss=statistics.mean(v['loss'] for v in records),
            checkpoints=sorted(int(p.name.removeprefix('checkpoint_')) for p in run.glob('checkpoint_*') if (p/'training_state.pt').exists())))
    for item in [*active.values(),*state.get('validating',{}).values()]:
        if item.get('job','').startswith('eval_'):
            value=evaluation_progress(root/'evaluation'/item['job'].removeprefix('eval_'))
            if value:
                value['cpu_only']=item.get('job') in state.get('validating',{})
                value['performance']=read(root/'evaluation'/item['job'].removeprefix('eval_')/'performance_status.json',{})
                evaluations.append(value)
    scores=read(root/'leaderboard.json',[])
    table3=[]
    for score in scores:
        summary=read(score['summary'])
        if summary:table3.append([score['name'],*table_values(summary)])
    telemetry=tail_records(root/'gpu.jsonl')
    return dict(updated=datetime.now(timezone.utc).isoformat(),phase=state.get('phase','unknown'),
        supervisor_pid=state.get('pid'),heartbeat=state.get('heartbeat'),error=state.get('error'),
        training=training,evaluations=evaluations,scores=scores,table3=table3,gpu=telemetry[-1] if telemetry else None)


def render(data,root,interval):
    esc=lambda v:html.escape(str(v))
    pct=lambda v:'—' if v is None else f'{v:.2%}'
    duration=lambda s:'—' if s is None else f'{int(s//3600)}小时{int(s%3600//60)}分'
    phases={'sampling_50k':'50k 采样方式对照','sampling_confirmation':'采样方式确认',
        'sampling_provisionally_fixed':'临时固定任务均匀采样，跳过采样确认',
        'sampling_confirmation_migration':'新方案迁移：采样筛选收尾与入围者确认并行',
        'migration_complete':'迁移完成，即将进入新方案',
        'sampling_extended_confirmation':'接近的采样候选追加确认',
        'budget_exploration_100k_150k_200k':'预算摸底：100k / 150k / 200k（训练与评估流水线）',
        'budget_exploration_250k_300k':'预算摸底：250k / 300k（训练与评估流水线）',
        'budget_extension_confirmation':'150k/200k确认与延长预算判断',
        'budget_exploration_confirmation':'摸底预算确认',
        'budget_matched_scratch_retraining':'按候选预算从scratch重训',
        'budget_fit_second_training_seed':'预算接近：第二训练种子复核',
        'trained_policy_inference_execution_screen':'充分训练后筛选执行窗口',
        'trained_policy_inference_diffusion_screen':'充分训练后筛选扩散步数',
        'execution_matched_budget_retraining':'执行历史匹配重训及相邻预算复核',
        'aligned_budget_second_training_seed':'历史匹配预算接近：第二训练种子复核',
        'learning_rate_50k':'50k 学习率对照','new50k_inference_execution_screen':'新50k模型执行窗口筛选',
        'new50k_inference_diffusion_screen':'新50k模型扩散步数筛选','new50k_inference_confirm':'新50k模型推理参数确认',
        'execution_matched_50k_retraining':'执行窗口匹配后的50k重训','budget_confirmation':'训练预算确认',
        'trained_policy_inference_recheck':'训练后执行窗口复核','budget_matched_cosine_comparison':'预算匹配的学习率调度对照',
        'final_diffusion_confirmation':'最终扩散参数确认','independent_training_seeds':'独立训练种子复训',
        'canonical_test_3_training_seeds':'三个训练种子的正式18×100测试','fresh_holdout_test':'全新环境种子留出测试',
        'complete':'全部完成','failed':'已停止：请查看错误','preflight':'前置检查'}
    phase=phases.get(data['phase'],data['phase'])
    if data['phase'].startswith('budget_') and data['phase'].removeprefix('budget_').isdigit():phase=f"训练预算对照：{int(data['phase'].split('_')[-1]):,}步"
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',f'<meta http-equiv="refresh" content="{interval}">',
        '<title>Kimodo · ToolBench 实验进度</title><style>body{font:15px system-ui;color:#172033;background:#f3f6fa;margin:24px}section{background:white;padding:20px;margin:18px 0;border-radius:10px;overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid #e6ebf2;white-space:nowrap}th{background:#f7f9fc}progress{width:180px;accent-color:#2563eb}.muted{color:#667085}.error{color:#b42318}h1{font-size:25px}h2{font-size:19px}code{font-size:12px}a{color:#175cd3}</style>',
        '<h1>Kimodo 完整模型 · HumanoidToolBench</h1>',f'<p>当前阶段：<b>{esc(phase)}</b> · 每{interval}秒刷新</p>',
        f'<p class="muted">页面更新 {esc(data["updated"])} · 编排器 heartbeat {esc(data["heartbeat"])} · 时间为 UTC</p>',
        '<p>主指标：12个L1/L2条件成功率的等权均值；辅助指标：全部18条件等权均值。开发筛选50回合/条件，确认100回合/条件；正式测试每个训练种子18×100。训练loss不能替代任务成功率。</p>']
    if data['error']:parts.append(f'<section class="error"><b>编排器错误</b><pre>{esc(data["error"])}</pre></section>')
    parts+=['<section><h2>训练与当前预算的剩余时间</h2><table><tr><th>候选</th><th>GPU</th><th>更新步数</th><th>当前预算进度</th><th>秒/步</th><th>训练剩余</th><th>近期平均loss</th><th>已保存checkpoint</th></tr>']
    for r in data['training']:
        target=r['target'];ratio=r['step']/target if target else None
        bar=f'<progress value="{min(1,ratio)}" max="1"></progress> {ratio:.1%}' if ratio is not None else '本段未在训练'
        parts.append(f'<tr><td>{esc(r["name"])}</td><td>{esc(r["gpus"] or "—")}</td><td>{r["step"]:,}'+(f' / {target:,}' if target else '')+f'</td><td>{bar}</td><td>{r["seconds_per_step"]:.3f}</td><td>{duration(r["remaining_seconds"])}</td><td>{r["recent_mean_loss"]:.5f}</td><td>{esc(", ".join(map(str,r["checkpoints"][-5:])))}</td></tr>')
    parts+=['</table><p class="muted">剩余时间只包含当前候选到当前预算的训练，按最近100条记录的中位速度估计，尚不包含后续仿真、确认和其他候选。每10k保存模型；目前成功率按50k间隔评估。</p></section>',
        '<section><h2>已完成并核验的成功率</h2><p>开发集用于选参；正式测试和留出结果单独标注。分数为官方成功判据下的成功率。</p>']
    if not data['scores']:parts.append('<p><b>本轮尚无完成的18条件评估。</b>当前不是0%成功率；尚未评测。首个结果在50k训练及18×50开发评估完成后出现。</p>')
    else:
        parts.append('<table><tr><th>候选</th><th>集合</th><th>训练步数</th><th>执行帧/扩散步</th><th>回合/条件</th><th>L1/L2 主指标</th><th>全18 辅助指标</th><th>官方摘要</th></tr>')
        labels={'screen':'开发筛选','confirm':'开发确认','confirm_extended':'开发追加确认','test':'正式测试','holdout':'全新环境留出'}
        for r in data['scores']:
            link=os.path.relpath(r['summary'],root)
            parts.append(f'<tr><td>{esc(r["name"])}</td><td>{esc(labels.get(r["stage"],r["stage"]))}</td><td>{r["training_step"]:,}</td><td>{r["execution"]} / {r["diffusion"]}</td><td>{r["episodes"]}</td><td><b>{pct(r["primary_l12"])}</b></td><td>{pct(r["auxiliary_all18"])}</td><td><a href="{esc(link)}">summary.json</a></td></tr>')
        parts.append('</table>')
    parts.append('</section><section><h2>各阶段 Table 3</h2><p><a href="stage_table3.md">完整表格</a> · <a href="stage_table3.csv">下载CSV</a> · <a href="stage_analysis.md">对比分析</a> · <a href="stage_comparison.csv">配置与计数明细</a></p>')
    if data.get('table3'):
        parts.append('<table><tr>'+''.join(f'<th>{esc(c)}</th>' for c in COLUMNS)+'</tr>')
        for row in data['table3']:parts.append('<tr><td>'+esc(row[0])+'</td>'+''.join(f'<td>{pct(v)}</td>' for v in row[1:])+'</tr>')
        parts.append('</table><p class="muted">D 对应官方 R；开发、确认、测试分组请查看上表及对比分析。每个 S/D 难度列平均三个场景。</p>')
    else:parts.append('<p>尚无完整核验成绩。</p>')
    parts.append('</section><section><h2>正在评估的临时成功计数</h2><p class="muted">临时结果尚未全部核验，不用于选参。未开始的条件显示“—”，不会计作失败或0%。</p>')
    if not data['evaluations']:parts.append('<p>当前没有正在运行的评估。</p>')
    for evaluation in data['evaluations']:
        parts.append(f'<h3>{esc(evaluation["name"])}</h3><p>已完成 {evaluation["total"]} / {evaluation["target_total"]} 回合；已核验 {evaluation["validated_conditions"]} / 18 条件；L1/L2已有回合覆盖 {evaluation["primary_coverage"]} / 12。</p>')
        if evaluation.get('cpu_only'):parts.append('<p>GPU仿真已完成并释放资源；官方全视频核验在CPU后台继续，核验完成前不用于选参。</p>')
        performance=evaluation.get('performance',{})
        if performance:parts.append(f'<p class="muted">仿真完成条件 {performance.get("completed_simulations",0)}；后台核验进程 {performance.get("validators_running",0)}。</p>')
        if evaluation['primary_mean'] is not None:parts.append(f'<p>临时L1/L2等权均值：{pct(evaluation["primary_mean"])}，须等待每个条件完成规定回合和核验。</p>')
        parts.append('<table><tr><th>条件</th><th>已完成回合</th><th>成功数</th><th>临时成功率</th><th>核验</th></tr>')
        for r in evaluation['rows']:parts.append(f'<tr><td>{esc(r["condition"])}</td><td>{r["episodes"]} / {evaluation["target_per_condition"]}</td><td>{r["successes"]}</td><td>{pct(r["rate"])}</td><td>{"已核验" if r["validated"] else "待核验"}</td></tr>')
        parts.append('</table>')
    parts.append('</section>')
    if data.get('gpu'):
        parts.append(f'<section><h2>GPU 实况</h2><p class="muted">{esc(data["gpu"]["time"])} · 索引 / 显存 / 利用率 / 温度；约每分钟更新</p><pre>{esc(data["gpu"]["gpus"])}</pre></section>')
    parts+=['<p class="muted">本页面只读取实验文件，不修改训练、评估或模型选择。视频位于 evaluation/候选/条件/run-*/videos/episode_*/。最终汇总为 RESULTS.md、table3.csv、budget_comparison.csv 和 video_index.csv。</p></html>']
    return ''.join(parts)


def write(path,content):
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp');temporary.write_text(content);temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--watch',action='store_true');parser.add_argument('--interval',type=int,default=15);args=parser.parse_args()
    root=args.root.resolve()
    if args.watch:
        lock=(root/'live_monitor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        data=collect(root);write(root/'live_progress.json',json.dumps(data,indent=2,ensure_ascii=False)+'\n')
        write(root/'live_progress.html',render(data,root,args.interval))
        if not args.watch or data['phase']=='complete':return
        time.sleep(args.interval)


if __name__=='__main__':main()
