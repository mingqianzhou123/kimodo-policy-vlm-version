"""CPU-only live curves and health receipts for an existing training pipeline.

Writes PNG/PDF/CSV, an auto-refreshing HTML dashboard, hourly snapshots and
alert transitions. Reads the run; never changes training weights or settings.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import html
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np


REPO = Path(__file__).resolve().parents[1]
TERMINAL = {"complete", "failed", "keepalive_recovery_failed", "stopped_with_checkpoint"}


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def process_alive(pid, expected=None):
    try:
        proc = Path(f"/proc/{int(pid)}")
        if proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
            return False
        return expected is None or expected in proc.joinpath("cmdline").read_bytes().decode(errors="replace")
    except (OSError, ValueError, TypeError):
        return False


def read_metrics(path):
    lines = path.read_text().splitlines()
    rows, malformed = [], []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row.get("step"), int):
                raise ValueError("missing integer step")
            rows.append(row)
        except (ValueError, AttributeError):
            # A writer can be halfway through its final append.
            if index != len(lines) - 1:
                malformed.append(index + 1)
    return rows, malformed


def health_alerts(rows, phase, pipeline_alive, age, malformed=()):
    alerts = {}
    if malformed:
        alerts["malformed_metrics"] = f"指标文件存在无效记录，行号：{list(malformed)[:5]}"
    if phase in {"failed", "keepalive_recovery_failed"}:
        alerts["pipeline_failed"] = f"训练／评估流程状态：{phase}"
    elif phase not in TERMINAL and not pipeline_alive:
        alerts["pipeline_missing"] = "训练／评估主进程已退出，但流程没有完成记录"
    if phase == "training" and age > 300:
        alerts["metrics_stalled"] = f"训练指标已 {age:.0f} 秒没有更新"
    if not rows:
        alerts["no_metrics"] = "尚无有效训练指标"
        return alerts
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                alerts["nonfinite_metrics"] = f"第 {row['step']} 步的 {key} 出现 NaN/Inf"
    # Compare within a single stage so unfreezing is not mistaken for a spike.
    same_stage = [r for r in rows if r.get("stage") == rows[-1].get("stage")]
    if len(same_stage) >= 600:
        for key, factor in [("loss", 4), ("grad_control", 5), ("grad_hand", 5)]:
            previous = [r[key] for r in same_stage[-600:-100] if math.isfinite(r.get(key, math.nan))]
            recent = [r[key] for r in same_stage[-100:] if math.isfinite(r.get(key, math.nan))]
            if previous and recent and statistics.median(recent) > factor * max(statistics.median(previous), 1e-8):
                alerts[f"sustained_{key}_spike"] = f"{key} 最近100条记录中位数超过此前500条的 {factor} 倍"
            if previous and recent:
                peak = max(recent)
                baseline = max(statistics.median(previous), 1e-8)
                if key == "loss" and peak > 20 * baseline:
                    alerts["loss_outlier"] = f"最近100条记录中出现单次损失尖峰：{peak:.4g}，此前中位数 {baseline:.4g}"
                elif key.startswith("grad_") and peak > max(100., 100 * baseline):
                    alerts[f"{key}_outlier"] = f"最近100条记录中出现裁剪前梯度尖峰：{peak:.4g}"
    return alerts


def historical_spikes(rows):
    """Retain severe transient spikes even after the rolling window recovers."""
    from collections import deque
    windows, events = {}, []
    for row in rows:
        history = windows.setdefault(row.get("stage"), deque(maxlen=100))
        reasons = []
        if len(history) == 100:
            loss = row.get("loss", math.nan)
            base = statistics.median(r.get("loss", 0) for r in history)
            if math.isfinite(loss) and loss > 20 * max(base, 1e-8):
                reasons.append("loss > 20x preceding 100-record median")
            for key in ("grad_control", "grad_hand"):
                value = row.get(key, 0)
                if value > 100 and value > 100 * max(statistics.median(r.get(key, 0) for r in history), 1e-8):
                    reasons.append(f"{key} > max(100, 100x preceding median), before clipping")
        if reasons:
            events.append({k: row.get(k) for k in ("step", "stage", "loss", "root_loss", "body_loss", "grad_control", "grad_hand")} | {"reasons": reasons})
        history.append(row)
    return events


def rolling(values, width=100):
    """Finite-only trailing mean, preserving NaNs where no values are valid."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    totals = np.r_[0., np.cumsum(np.where(valid, values, 0.))]
    counts = np.r_[0, np.cumsum(valid)]
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - width)
    denom = counts[end] - counts[start]
    return np.divide(totals[end] - totals[start], denom,
                     out=np.full(len(values), np.nan), where=denom > 0)


def plot_curves(rows, config, output, latest, save_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    x = np.array([r["step"] for r in rows])
    sample = np.unique(np.linspace(0, len(rows) - 1, min(len(rows), 2500), dtype=int))
    # Preserve rare extremes as well as uniformly sampled raw records.
    extrema = []
    for key in ("loss", "root_loss", "body_loss", "hand_loss", "grad_control", "grad_hand", "s_per_step", "data_s_per_step"):
        values = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        valid = np.flatnonzero(np.isfinite(values))
        if len(valid):
            extrema.extend(valid[np.argsort(np.abs(values[valid]))[-10:]].tolist())
    sample = np.unique(np.r_[sample, extrema]).astype(int)
    stage_start = config.get("training", {}).get("stages") or {}
    fig, axes = plt.subplots(4, 2, figsize=(17, 16), constrained_layout=True)
    def panel(axis, keys, title, log=False):
        for key, label in keys:
            values = np.array([r.get(key, np.nan) for r in rows], dtype=float)
            if not np.isfinite(values).any():
                continue
            line, = axis.plot(x[sample], rolling(values)[sample], label=label, lw=1.6)
            axis.plot(x[sample], values[sample], color=line.get_color(), alpha=.16, lw=.5)
        axis.set(title=title, xlabel="Optimizer update", ylabel="Value")
        if log:
            axis.set_yscale("log")
        axis.grid(alpha=.18)
        if axis.lines:
            axis.legend(fontsize=9)
    flat = axes.ravel()
    panel(flat[0], [("loss", "Total"), ("motion_loss", "Motion (weighted)"), ("hand_loss", "Hand")], "Training objective", True)
    panel(flat[1], [("root_loss", "Root"), ("body_loss", "Body")], "Future motion losses", True)
    panel(flat[2], [("hand_state_loss", "Hand state"), ("hand_transition_loss", "Hand transition")], "Hand supervision", True)
    panel(flat[3], [("grad_control", "Injection + unfrozen adapter"), ("grad_hand", "HandNet")], "Gradient norms before clipping", True)
    panel(flat[4], [("lr_adapter", "Injection LR (logged)"), ("lr_backbone", "Kimodo LR (logged)")], "Learning rates")
    opt = config["training"]["optimizer"]
    if stage_start.get("adapter_start") is not None:
        # The trainer logs injection LR. Derive the adapter LR using its exact
        # shared scheduler ratio; this is explicitly labelled as derived.
        warmup = config["training"].get("scheduler", {}).get("adapter_warmup_steps", 0)
        adapter_start = config["training"].get("stages", {}).get("adapter_start", 0)
        vl_lr = np.array([r.get("lr_adapter", np.nan) * opt["vl_adapter_lr"] / opt["control_adapter_lr"]
                          * (min(1., max(0., r["step"] - adapter_start) / warmup) if warmup else 1.)
                          if r.get("stage") in {"adapter", "vision"} else 0. for r in rows])
        flat[4].plot(x[sample], vl_lr[sample], label="VL adapter LR (derived)", lw=1.5)
        flat[4].legend(fontsize=9)
    panel(flat[5], [("s_per_step", "Wall seconds / update")], "Training speed (includes periodic saves)")
    panel(flat[6], [("gpu_peak_memory_gb", "Rank-0 allocated peak (GB)")], "CUDA allocation high-water mark")
    panel(flat[7], [("data_s_per_step", "Data wait seconds / update")], "Data loader wait")
    for axis in flat:
        for key in ("adapter_start", "vision_start"):
            boundary = stage_start.get(key)
            if boundary is not None and boundary <= x[-1]:
                axis.axvline(boundary, color="#666666", ls="--", alpha=.5, lw=.9)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle(f"{config['main']['task']} / GR00T conditioning / scratch / {latest.get('world_size', 4)} GPUs\n"
                 f"Update {latest['step']:,}/{config['main']['max_steps']:,} | phase: {latest['phase']} | {latest['updated_at']}\n"
                 "Faint: sampled raw records; solid: trailing mean of 100 records (~1,000 updates); dashed: unfreezing", fontsize=13)
    for extension in (["png", "pdf"] if save_pdf else ["png"]):
        path = output / f"training_curves.{extension}"
        temporary = path.with_name(path.name + ".tmp")
        fig.savefig(temporary, format=extension, dpi=135)
        temporary.replace(path)
    plt.close(fig)


def dashboard(summary, rows, output, interval):
    esc = html.escape
    alerts = summary["alerts"]
    alert_html = "".join(f"<li>{esc(v)}</li>" for v in alerts.values()) or "<li>未触发告警</li>"
    cards = {"训练进度": f"{summary['step']:,} / {summary['max_steps']:,}（{summary['progress_percent']:.1f}%）",
             "当前阶段": summary["stage"], "流程状态": summary["phase"],
             "最近1000步平均损失": f"{summary['recent_loss_mean']:.5f}",
             "最近1000步平均耗时": f"{summary['recent_s_per_step']:.3f} 秒/步",
             "训练预计剩余": f"{summary['remaining_training_hours']:.1f} 小时（不含评估）",
             "最新完整保存点": str(summary["latest_checkpoint"]),
             "历史明显尖峰记录": str(summary.get("historical_spike_count", 0)),
             "保活监控": "运行中" if summary["xiancun_alive"] else "需要检查"}
    card_html = "".join(f"<div class='card'><small>{esc(k)}</small><strong>{esc(v)}</strong></div>" for k,v in cards.items())
    recent_html = "".join("<tr>" + "".join(f"<td>{esc(str(r.get(k,'')))}</td>" for k in
                    ["step", "loss", "root_loss", "body_loss", "hand_loss", "grad_control", "grad_hand"]) + "</tr>" for r in rows[-10:][::-1])
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta http-equiv="refresh" content="{interval}"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(summary.get('task', 'Kimodo'))} 实时训练监督</title><style>
body{{font:16px system-ui,sans-serif;max-width:1450px;margin:28px auto;padding:0 20px;color:#17212d;background:#f5f7fa}}
h1{{font-size:25px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}
.card,section{{background:white;border:1px solid #dde2e8;border-radius:10px;padding:16px;margin:10px 0}}
small{{display:block;color:#596574}}strong{{display:block;margin-top:8px}}img{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #eee;text-align:left}}
.scroll{{overflow:auto}}a{{color:#1656a8}}.alert{{border-color:{'#c24535' if alerts else '#6b9a75'}}}
</style><h1>{esc(summary.get('task', 'Kimodo'))} 实时训练监督</h1>
<p>更新时间：{esc(summary['updated_at'])} · 每 {interval} 秒刷新 · 监控进程 {os.getpid()}</p>
<div class="cards">{card_html}</div><section class="alert"><strong>监控检查</strong><ul>{alert_html}</ul>
<p>检查非有限数值、5分钟无指标更新、主进程异常退出、同阶段持续损失／梯度突增及 xiancun 状态。
告警写入 alerts.jsonl 和监控日志，历史尖峰保存在 <a href="historical_spikes.json">historical_spikes.json</a>。
训练损失不能替代闭环任务成功率，最终成功率以仿真评估为准。</p></section>
<section><p><a href="training_curves.png">PNG 曲线</a> · <a href="training_curves.pdf">PDF 曲线</a> ·
<a href="metrics.csv">完整指标 CSV</a> · <a href="health.json">监控状态 JSON</a> · <a href="../train.log">训练日志</a></p>
<img src="training_curves.png?t={time.time_ns()}" alt="训练过程曲线"></section>
<section><strong>最近10条训练记录</strong><div class="scroll"><table><tr><th>Step</th><th>Total loss</th><th>Root</th><th>Body</th><th>Hand</th><th>Grad control</th><th>Grad hand</th></tr>{recent_html}</table></div></section>
<p>完整原始记录持续保存在 ../metrics.jsonl；本面板导出全部记录到 CSV，图中原始点均匀采样，滑动平均在完整记录上计算。
VL adapter 学习率由训练配置和已记录注入层学习率推导。图表不包含验证集损失。每小时及流程结束保存快照。</p></html>"""
    atomic_text(output / "dashboard.html", page)


def refresh(args):
    output, run = args.out, args.run_dir
    status = read_json(args.status)
    config = read_json(run / "config.json")
    metrics = run / "metrics.jsonl"
    rows, malformed = read_metrics(metrics) if metrics.exists() else ([], [])
    if not rows:
        if status.get('phase') == 'training' and not args.once:
            print('Waiting for the first complete training metrics record', flush=True)
            return 'training'
        raise RuntimeError("No complete metrics records available")
    phase = status.get("phase", "unknown")
    fallback = 'run_arena_stable.py' if 'resume_validation' in status else 'run_groot_experiment.py'
    pipeline_alive = process_alive(status.get("pid"), status.get('pipeline_script', fallback))
    age = time.time() - metrics.stat().st_mtime
    alerts = health_alerts(rows, phase, pipeline_alive, age, malformed)
    xiancun = read_json(Path.home() / ".cache/xiancun/status.json")
    xiancun_alive = process_alive(xiancun.get("pid"), "xiancun")
    if not xiancun_alive:
        alerts["xiancun_missing"] = "xiancun 监控进程不在运行"
    elif time.time() - xiancun.get("updated", 0) > 180:
        alerts["xiancun_stale"] = "xiancun 状态超过3分钟未更新"
    max_steps, step = config["main"]["max_steps"], rows[-1]["step"]
    recent = rows[-100:]
    finite = lambda key: [r[key] for r in recent if math.isfinite(r.get(key, math.nan))]
    speed = statistics.mean(finite("s_per_step")) if finite("s_per_step") else 0.
    checkpoints = [int(p.name.split("_")[-1]) for p in run.glob("checkpoint_*")
                   if p.name.split("_")[-1].isdigit() and (p / "training_state.pt").is_file() and (p / "config.json").is_file()]
    summary = {"updated_at": datetime.now(timezone.utc).isoformat(), "monitor_pid": os.getpid(),
               "run_dir": str(run), "task": config['main']['task'], "world_size": status.get('world_size', 4),
               "phase": phase, "stage": rows[-1].get("stage", "unknown"),
               "pipeline_alive": pipeline_alive, "step": step, "max_steps": max_steps,
               "progress_percent": step / max_steps * 100, "metrics_age_seconds": age,
               "recent_loss_mean": statistics.mean(finite("loss")) if finite("loss") else math.nan,
               "recent_s_per_step": speed, "remaining_training_hours": max(0, max_steps-step)*speed/3600,
               "latest_checkpoint": max(checkpoints, default=0), "xiancun_alive": xiancun_alive,
               "xiancun_mode": xiancun.get("mode"), "alerts": alerts, "last_record": rows[-1]}
    summary["evaluation"] = read_json(run / "experiment_summary.json")
    spikes = historical_spikes(rows)
    summary["historical_spike_count"] = len(spikes)
    summary["last_historical_spike"] = spikes[-1] if spikes else None
    atomic_text(output / "historical_spikes.json", json.dumps(spikes, ensure_ascii=False, indent=2))
    summary["gpu_snapshot_from_xiancun"] = [g for g in xiancun.get("gpus", []) if g.get("index") in status.get("gpus", [])]
    previous = read_json(output / "health.json")
    with (output / "alerts.jsonl").open("a") as handle:
        for key in set(previous.get("alerts", {})) | set(alerts):
            was, current = previous.get("alerts", {}).get(key), alerts.get(key)
            if was == current or was and current:
                continue
            event = {"time": summary["updated_at"], "step": step, "kind": key,
                     "state": "active" if current else "resolved", "message": current or was}
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(json.dumps(event, ensure_ascii=False), flush=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    temporary = output / "metrics.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "metrics.csv")
    plot_curves(rows, config, output, summary, save_pdf=True)
    dashboard(summary, rows, output, args.interval)
    atomic_text(output / "health.json", json.dumps(summary, ensure_ascii=False, indent=2))
    snapshots = output / "snapshots"
    snapshots.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H")
    if phase in TERMINAL:
        stamp += "_" + phase
    snapshot = snapshots / f"{stamp}.png"
    if not snapshot.exists():
        import shutil
        shutil.copyfile(output / "training_curves.png", snapshot)
        atomic_text(snapshot.with_suffix(".json"), json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"{summary['updated_at']} step={step}/{max_steps} phase={phase} "
          f"recent_loss={summary['recent_loss_mean']:.6f} remaining={summary['remaining_training_hours']:.2f}h alerts={len(alerts)}", flush=True)
    return phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.interval < 10:
        parser.error("interval must be at least 10 seconds")
    args.run_dir, args.status = args.run_dir.resolve(), args.status.resolve()
    args.out = (args.out or args.run_dir / "monitor").resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.detach:
        existing = read_json(args.out / "health.json").get("monitor_pid")
        if process_alive(existing, "monitor_groot_training.py"):
            print(json.dumps({"pid": existing, "already_running": True,
                              "dashboard": str(args.out / "dashboard.html")}, indent=2))
            return
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--run-dir", str(args.run_dir),
                   "--status", str(args.status), "--out", str(args.out), "--interval", str(args.interval)]
        with (args.out / "monitor.log").open("a") as handle:
            process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1"})
        print(json.dumps({"pid": process.pid, "dashboard": str(args.out / "dashboard.html"), "log": str(args.out / "monitor.log")}, indent=2))
        return
    with (args.out / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_text(args.out / "monitor.pid", str(os.getpid()))
        while True:
            started = time.monotonic()
            try:
                phase = refresh(args)
                if args.once or phase in TERMINAL:
                    break
            except Exception as exc:
                atomic_text(args.out / "monitor_error.json", json.dumps({"time": time.time(), "error": repr(exc)}))
                import traceback
                traceback.print_exc()
                if args.once:
                    raise
            time.sleep(max(1, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
