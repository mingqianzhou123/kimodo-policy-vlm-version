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
<title>AdaLN-Zero Arena Training</title>
<style>
:root { color-scheme: dark; --bg:#0b1020; --panel:#121a2d; --grid:#293653; --text:#e7edf8; --muted:#9aa9c5; --blue:#65a8ff; --cyan:#4fd1c5; --orange:#f6ad55; --pink:#f687b3; --green:#68d391; }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--text); font:14px system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { max-width:1500px; margin:0 auto; padding:20px; } h1 { margin:0 0 4px; font-size:24px; } h2 { margin:0 0 10px; font-size:16px; } .muted { color:var(--muted); }
.header { display:flex; justify-content:space-between; gap:20px; align-items:flex-start; margin-bottom:16px; } .status { text-align:right; color:var(--muted); }
.cards { display:grid; grid-template-columns:repeat(6,minmax(120px,1fr)); gap:10px; margin-bottom:14px; } .card,.panel { background:var(--panel); border:1px solid #1f2a44; border-radius:10px; }
.card { padding:12px; } .label { color:var(--muted); font-size:12px; } .value { font-size:21px; font-weight:650; margin-top:4px; }
.progress { height:10px; background:#202b45; border-radius:99px; overflow:hidden; margin-top:9px; } #bar { height:100%; background:linear-gradient(90deg,var(--cyan),var(--blue)); width:0%; }
.grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; } .panel { padding:14px; min-height:290px; } canvas { width:100%; height:230px; display:block; }
table { width:100%; border-collapse:collapse; font-size:12px; } th,td { border-bottom:1px solid #23304c; padding:6px; text-align:right; } th:first-child,td:first-child { text-align:left; }
@media(max-width:900px) { .cards { grid-template-columns:repeat(3,1fr); } .grid { grid-template-columns:1fr; } .header { display:block; } .status { text-align:left; margin-top:8px; } }
</style></head>
<body><main>
<div class="header"><div><h1>AdaLN-Zero Arena 七任务训练</h1><div class="muted" id="run">读取中…</div></div><div class="status">自动刷新：<span id="refresh">10</span>s<br><span id="updated">等待 metrics.jsonl</span></div></div>
<div class="cards">
  <div class="card"><div class="label">Step</div><div class="value" id="step">—</div><div class="progress"><div id="bar"></div></div></div>
  <div class="card"><div class="label">Progress</div><div class="value" id="progress">—</div></div>
  <div class="card"><div class="label">Loss</div><div class="value" id="loss">—</div></div>
  <div class="card"><div class="label">LR AdaLN</div><div class="value" id="lr">—</div></div>
  <div class="card"><div class="label">sec/step</div><div class="value" id="speed">—</div></div>
  <div class="card"><div class="label">ETA</div><div class="value" id="eta">—</div></div>
</div>
<div class="grid">
  <section class="panel"><h2>总损失与分项损失</h2><canvas id="lossChart"></canvas></section>
  <section class="panel"><h2>学习率</h2><canvas id="lrChart"></canvas></section>
  <section class="panel"><h2>梯度范数</h2><canvas id="gradChart"></canvas></section>
  <section class="panel"><h2>吞吐与数据等待</h2><canvas id="timeChart"></canvas></section>
  <section class="panel"><h2>最近指标</h2><table><thead><tr><th>step</th><th>loss</th><th>motion</th><th>hand</th><th>stage</th><th>sec/step</th></tr></thead><tbody id="rows"></tbody></table></section>
  <section class="panel"><h2>说明</h2><div class="muted" style="line-height:1.7">
    曲线来自训练目录中的 <code>metrics.jsonl</code>，每次读取最新内容并自动刷新。<br>
    AdaLN adapter 使用 <code>lr_adaln</code>；兼容字段 <code>lr_adapter</code> 同时显示。<br>
    ETA 根据最近记录的平均 step time 计算；启动编译和数据缓存阶段可能暂时偏高。
  </div></section>
</div></main>
<script>
const MAX_STEPS = __MAX_STEPS__;
const COLORS = ['#65a8ff','#4fd1c5','#f6ad55','#f687b3','#68d391','#c4b5fd'];
let rows = [];
function fmt(x, n=4) { return x == null || !Number.isFinite(Number(x)) ? '—' : Number(x).toFixed(n); }
function sci(x) { if (!Number.isFinite(Number(x))) return '—'; return Number(x).toExponential(2); }
function draw(id, series, labels) {
  const c=document.getElementById(id), dpr=window.devicePixelRatio||1, rect=c.getBoundingClientRect();
  const w=Math.max(300,rect.width), h=230; c.width=w*dpr; c.height=h*dpr; const g=c.getContext('2d'); g.scale(dpr,dpr);
  g.fillStyle='#121a2d'; g.fillRect(0,0,w,h); const p={l:48,r:12,t:18,b:30}, cw=w-p.l-p.r, ch=h-p.t-p.b;
  const all=series.flatMap(s=>s.values.filter(Number.isFinite)); if(!all.length)return;
  let lo=Math.min(...all), hi=Math.max(...all); if(lo===hi){lo-=1;hi+=1;} const pad=(hi-lo)*.08; lo-=pad;hi+=pad;
  g.strokeStyle='#293653';g.lineWidth=1;g.font='11px system-ui';g.fillStyle='#9aa9c5';
  for(let i=0;i<=4;i++){const y=p.t+ch*i/4,v=hi-(hi-lo)*i/4;g.beginPath();g.moveTo(p.l,y);g.lineTo(w-p.r,y);g.stroke();g.fillText(fmt(v,3),4,y+4);}
  const n=Math.max(...series.map(s=>s.values.length),1); const xAt=i=>p.l+(n===1?0:i/(n-1))*cw; const yAt=v=>p.t+(hi-v)/(hi-lo)*ch;
  series.forEach((s,si)=>{g.strokeStyle=COLORS[si%COLORS.length];g.lineWidth=2;g.beginPath();s.values.forEach((v,i)=>{if(!Number.isFinite(v))return;const x=xAt(i),y=yAt(v);if(i===0||!Number.isFinite(s.values[i-1]))g.moveTo(x,y);else g.lineTo(x,y);});g.stroke();});
  labels.forEach((label,i)=>{g.fillStyle=COLORS[i%COLORS.length];g.fillRect(p.l+i*125,p.t-13,9,9);g.fillStyle='#c6d1e5';g.fillText(label,p.l+14+i*125,p.t-4);});
  g.fillStyle='#9aa9c5';g.fillText('step '+(rows[0]?.step??'—'),p.l,h-7);g.fillText('step '+(rows.at(-1)?.step??'—'),w-100,h-7);
}
function render() {
  if(!rows.length)return; const last=rows.at(-1), first=rows[0];
  const step=Number(last.step), pct=100*step/MAX_STEPS; document.getElementById('step').textContent=step.toLocaleString(); document.getElementById('progress').textContent=pct.toFixed(2)+'%'; document.getElementById('bar').style.width=Math.min(100,pct)+'%';
  document.getElementById('loss').textContent=fmt(last.loss); document.getElementById('lr').textContent=sci(last.lr_adaln ?? last.lr_adapter); document.getElementById('speed').textContent=fmt(last.s_per_step,2);
  const tail=rows.slice(-20), avg=tail.reduce((a,r)=>a+Number(r.s_per_step||0),0)/Math.max(1,tail.length), eta=(MAX_STEPS-step)*avg/3600; document.getElementById('eta').textContent=eta>48?(eta/24).toFixed(1)+' d':eta.toFixed(1)+' h'; document.getElementById('updated').textContent='最新 step '+step+' · '+new Date().toLocaleTimeString();
  draw('lossChart',[{values:rows.map(r=>Number(r.loss))},{values:rows.map(r=>Number(r.motion_loss))},{values:rows.map(r=>Number(r.hand_loss))},{values:rows.map(r=>Number(r.root_loss))},{values:rows.map(r=>Number(r.body_loss))}],['loss','motion','hand','root','body']);
  draw('lrChart',[{values:rows.map(r=>Number(r.lr_adaln ?? r.lr_adapter))},{values:rows.map(r=>Number(r.lr_backbone))}],['AdaLN/adapter','backbone']);
  draw('gradChart',[{values:rows.map(r=>Number(r.grad_control))},{values:rows.map(r=>Number(r.grad_hand))}],['control','hand']);
  draw('timeChart',[{values:rows.map(r=>Number(r.s_per_step))},{values:rows.map(r=>Number(r.data_s_per_step))}],['step','data']);
  document.getElementById('rows').innerHTML=rows.slice(-12).reverse().map(r=>`<tr><td>${r.step}</td><td>${fmt(r.loss)}</td><td>${fmt(r.motion_loss)}</td><td>${fmt(r.hand_loss)}</td><td>${r.stage||''}</td><td>${fmt(r.s_per_step,2)}</td></tr>`).join('');
}
async function load(){try{const text=await fetch('metrics.jsonl?ts='+Date.now(),{cache:'no-store'}).then(r=>r.text()); rows=text.trim().split(/\n+/).map(x=>{try{return JSON.parse(x)}catch(e){return null}}).filter(Boolean); document.getElementById('run').textContent=location.pathname.replace(/\/training_dashboard\.html$/,''); render();}catch(e){document.getElementById('updated').textContent='等待 metrics.jsonl…';}}
window.addEventListener('resize',render); load(); setInterval(load,10000);
</script></body></html>'''


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-steps", type=int, default=500000)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    dashboard = run_dir / "training_dashboard.html"
    dashboard.write_text(HTML.replace("__MAX_STEPS__", str(args.max_steps)), encoding="utf-8")
    os.chdir(run_dir)
    with socketserver.ThreadingTCPServer((args.host, args.port), QuietHandler) as server:
        print(json.dumps({"dashboard": str(dashboard), "url": f"http://{args.host}:{args.port}/training_dashboard.html", "metrics": str(run_dir / "metrics.jsonl")}), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
