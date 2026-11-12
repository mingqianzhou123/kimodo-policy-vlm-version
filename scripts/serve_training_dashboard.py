#!/usr/bin/env python3
"""Serve a lightweight, dependency-free live training dashboard.

The generated HTML lives beside metrics.jsonl in the training run directory.
It is intentionally served by this process so browsers can poll the growing
JSONL file without file:// CORS restrictions.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
from pathlib import Path
import socketserver


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AdaLN-Zero Arena Training Dashboard</title>
<style>
:root{color-scheme:dark;--bg:#080d18;--panel:#121b2e;--grid:#2b3b5d;--text:#eef4ff;--muted:#9eafcc;--blue:#61a8ff;--cyan:#40d4c1;--orange:#ffb454;--pink:#f58cb5;--green:#78db91;--purple:#bd9cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1600px;margin:auto;padding:20px}h1{font-size:25px;margin:0 0 4px}h2{font-size:16px;margin:0 0 10px}.muted{color:var(--muted)}.top{display:flex;justify-content:space-between;gap:20px;margin-bottom:15px}.status{text-align:right;color:var(--muted)}.toolbar,.card,.panel{background:var(--panel);border:1px solid #203050;border-radius:10px}.toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:10px;margin-bottom:14px;color:var(--muted)}select,input{background:#1b2944;color:var(--text);border:1px solid #3a5078;border-radius:6px;padding:5px 8px}.cards{display:grid;grid-template-columns:repeat(7,1fr);gap:10px;margin-bottom:14px}.card{padding:11px}.label{font-size:12px;color:var(--muted)}.value{font-size:20px;font-weight:650;margin-top:4px}.progress{height:9px;background:#253451;border-radius:20px;overflow:hidden;margin-top:9px}.bar{height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--blue))}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.panel{padding:13px;min-height:315px}.chart{width:100%;height:250px;display:block}.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--muted);margin-top:5px}.legend span i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}.note{color:var(--muted);line-height:1.8}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:6px;border-bottom:1px solid #263653;text-align:right}th:first-child,td:first-child{text-align:left}@media(max-width:1000px){.cards{grid-template-columns:repeat(4,1fr)}.grid{grid-template-columns:1fr}.top{display:block}.status{text-align:left;margin-top:8px}}@media(max-width:600px){.cards{grid-template-columns:repeat(2,1fr)}main{padding:10px}}
</style>
</head>
<body><main>
<div class="top"><div><h1>AdaLN-Zero Arena 七任务训练</h1><div class="muted" id="run">读取 metrics.jsonl…</div></div><div class="status" id="updated">每 10 秒刷新</div></div>
<div class="toolbar"><label>曲线范围</label><select id="range"><option value="5000">最近 5,000 条记录</option><option value="20000">最近 20,000 条记录</option><option value="all">全程</option></select><label>平滑窗口</label><input id="smooth" type="range" min="1" max="101" step="2" value="21"><span id="smoothValue">21</span><span>半透明线=原始值，实线=移动平均</span></div>
<div class="cards"><div class="card"><div class="label">Step</div><div class="value" id="step">—</div><div class="progress"><div class="bar" id="bar"></div></div></div><div class="card"><div class="label">Progress</div><div class="value" id="progress">—</div></div><div class="card"><div class="label">Loss</div><div class="value" id="loss">—</div></div><div class="card"><div class="label">Motion loss</div><div class="value" id="motion">—</div></div><div class="card"><div class="label">Hand loss</div><div class="value" id="hand">—</div></div><div class="card"><div class="label">AdaLN LR</div><div class="value" id="lr">—</div></div><div class="card"><div class="label">ETA</div><div class="value" id="eta">—</div></div></div>
<div class="grid">
<section class="panel"><h2>总损失、Motion、Hand</h2><canvas class="chart" id="main"></canvas><div class="legend" id="mainLegend"></div></section>
<section class="panel"><h2>Root、Body、Hand 子损失</h2><canvas class="chart" id="parts"></canvas><div class="legend" id="partsLegend"></div></section>
<section class="panel"><h2>学习率</h2><canvas class="chart" id="lrChart"></canvas><div class="legend" id="lrLegend"></div></section>
<section class="panel"><h2>梯度范数</h2><canvas class="chart" id="grad"></canvas><div class="legend" id="gradLegend"></div></section>
<section class="panel"><h2>Step 时间和 Data 时间</h2><canvas class="chart" id="time"></canvas><div class="legend" id="timeLegend"></div></section>
<section class="panel"><h2>最近记录</h2><table><thead><tr><th>step</th><th>loss</th><th>motion</th><th>root</th><th>body</th><th>hand</th><th>stage</th><th>sec/step</th></tr></thead><tbody id="recent"></tbody></table></section>
<section class="panel"><h2>阅读方式</h2><div class="note">优先观察左上角总损失和 Root/Body/Hand 子损失。切换“最近 5,000 条记录”可以看清当前收敛趋势，切换“全程”可以看整体阶段变化。平滑窗口只影响实线，不会隐藏原始曲线。训练初期编译和缓存会抬高 step time，ETA 使用最近 100 条记录估计。最终是否继续训练，应以固定 checkpoint 的 Arena 成功率为准。</div></section>
</div></main>
<script>
const MAX_STEPS=500000, colors=['#61a8ff','#40d4c1','#ffb454','#f58cb5','#78db91','#bd9cff'];let rows=[];
const num=(x)=>Number.isFinite(+x)?+x:NaN, fmt=(x,n=4)=>Number.isFinite(+x)?(+x).toFixed(n):'—', sci=(x)=>Number.isFinite(+x)?(+x).toExponential(2):'—';
function moving(values,w){if(w<=1)return values.slice();const out=[],q=[];let sum=0;for(const value of values){q.push(Number.isFinite(value)?value:0);sum+=q.at(-1);if(q.length>w)sum-=q.shift();out.push(sum/q.length)}return out}
function selected(){const v=document.getElementById('range').value;if(v==='all')return rows.slice();return rows.slice(-Number(v))}
function sample(a,max=1200){if(a.length<=max)return a;return Array.from({length:max},(_,i)=>a[Math.round(i*(a.length-1)/(max-1))])}
function renderLegend(id,series){document.getElementById(id).innerHTML=series.map((s,i)=>`<span><i style="background:${colors[i%colors.length]}"></i>${s.name}</span>`).join('')}
function chart(id,legendId,defs){const data=sample(selected()),c=document.getElementById(id),dpr=devicePixelRatio||1,w=Math.max(340,c.clientWidth),h=250;c.width=w*dpr;c.height=h*dpr;const g=c.getContext('2d');g.scale(dpr,dpr);g.fillStyle='#121b2e';g.fillRect(0,0,w,h);const pad={left:54,right:15,top:25,bottom:32},cw=w-pad.left-pad.right,ch=h-pad.top-pad.bottom;const series=defs.map(d=>({name:d.name,steps:data.map(r=>num(r.step)),raw:data.map(r=>num(r[d.key]))})),all=series.flatMap(s=>s.raw.filter(Number.isFinite));if(!all.length)return;let lo=Math.min(...all),hi=Math.max(...all);if(lo===hi){lo-=1;hi+=1}const extra=(hi-lo)*.1;lo-=extra;hi+=extra;g.font='11px system-ui';g.strokeStyle='#2b3b5d';g.fillStyle='#9eafcc';for(let i=0;i<=4;i++){const y=pad.top+ch*i/4,v=hi-(hi-lo)*i/4;g.beginPath();g.moveTo(pad.left,y);g.lineTo(w-pad.right,y);g.stroke();g.fillText(fmt(v,3),5,y+4)}for(let i=0;i<=4;i++){const x=pad.left+cw*i/4,j=Math.min(data.length-1,Math.round((data.length-1)*i/4));g.fillText(data[j]?.step??'',x-17,h-10)}const xAt=i=>pad.left+(data.length<=1?0:i/(data.length-1))*cw,yAt=v=>pad.top+(hi-v)/(hi-lo)*ch;const win=Number(document.getElementById('smooth').value);series.forEach((s,si)=>{const col=colors[si%colors.length];g.strokeStyle=col;g.lineWidth=1;g.globalAlpha=.25;g.beginPath();s.raw.forEach((v,i)=>{if(!Number.isFinite(v))return;i?g.lineTo(xAt(i),yAt(v)):g.moveTo(xAt(i),yAt(v))});g.stroke();g.globalAlpha=1;g.lineWidth=2.4;g.beginPath();moving(s.raw,win).forEach((v,i)=>{i?g.lineTo(xAt(i),yAt(v)):g.moveTo(xAt(i),yAt(v))});g.stroke();const last=moving(s.raw,win).at(-1);g.fillStyle=col;g.beginPath();g.arc(xAt(data.length-1),yAt(last),3,0,Math.PI*2);g.fill()});renderLegend(legendId,series)}
function render(){if(!rows.length)return;const last=rows.at(-1),step=num(last.step),pct=100*step/MAX_STEPS;document.getElementById('step').textContent=step.toLocaleString();document.getElementById('progress').textContent=pct.toFixed(2)+'%';document.getElementById('bar').style.width=Math.min(100,pct)+'%';document.getElementById('loss').textContent=fmt(last.loss);document.getElementById('motion').textContent=fmt(last.motion_loss);document.getElementById('hand').textContent=fmt(last.hand_loss);document.getElementById('lr').textContent=sci(last.lr_adaln??last.lr_adapter);const tail=rows.slice(-100),avg=tail.reduce((a,r)=>a+(num(r.s_per_step)||0),0)/Math.max(1,tail.length),eta=(MAX_STEPS-step)*avg/3600;document.getElementById('eta').textContent=eta>48?(eta/24).toFixed(1)+' d':eta.toFixed(1)+' h';document.getElementById('updated').textContent='最新 step '+step.toLocaleString()+' · '+new Date().toLocaleTimeString();document.getElementById('run').textContent=location.pathname.replace(/\/training_dashboard\.html$/,'');chart('main','mainLegend',[{name:'loss',key:'loss'},{name:'motion',key:'motion_loss'},{name:'hand',key:'hand_loss'}]);chart('parts','partsLegend',[{name:'root',key:'root_loss'},{name:'body',key:'body_loss'},{name:'hand state',key:'hand_state_loss'},{name:'hand transition',key:'hand_transition_loss'}]);chart('lrChart','lrLegend',[{name:'AdaLN',key:'lr_adaln'},{name:'adapter',key:'lr_adapter'},{name:'backbone',key:'lr_backbone'}]);chart('grad','gradLegend',[{name:'control',key:'grad_control'},{name:'hand',key:'grad_hand'}]);chart('time','timeLegend',[{name:'step',key:'s_per_step'},{name:'data',key:'data_s_per_step'}]);document.getElementById('recent').innerHTML=rows.slice(-14).reverse().map(r=>`<tr><td>${r.step}</td><td>${fmt(r.loss)}</td><td>${fmt(r.motion_loss)}</td><td>${fmt(r.root_loss)}</td><td>${fmt(r.body_loss)}</td><td>${fmt(r.hand_loss)}</td><td>${r.stage||''}</td><td>${fmt(r.s_per_step,2)}</td></tr>`).join('')}
+async function load(){try{const text=await fetch('metrics.jsonl?ts='+Date.now(),{cache:'no-store'}).then(r=>r.text());rows=text.trim().split(/\n+/).map(x=>{try{return JSON.parse(x)}catch(e){return null}}).filter(Boolean);render()}catch(e){document.getElementById('updated').textContent='等待 metrics.jsonl…'}}
document.getElementById('range').onchange=render;document.getElementById('smooth').oninput=e=>{document.getElementById('smoothValue').textContent=e.target.value;render()};window.onresize=render;load();setInterval(load,10000);
+</script></body></html>
'''


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-steps", type=int, default=500000)
    parser.add_argument("--write-only", action="store_true", help="write HTML and exit without opening a port")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    dashboard = run_dir / "training_dashboard.html"
    dashboard.write_text(HTML.replace("__MAX_STEPS__", str(args.max_steps)), encoding="utf-8")
    if args.write_only:
        print(str(dashboard), flush=True)
        return
    os.chdir(run_dir)
    with socketserver.ThreadingTCPServer((args.host, args.port), QuietHandler) as server:
        print(json.dumps({"dashboard": str(dashboard), "url": f"http://{args.host}:{args.port}/training_dashboard.html", "metrics": str(run_dir / "metrics.jsonl")}), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
