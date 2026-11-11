"""Motion-loss and gradient-norm curves of the nine main-table tasks on one page: one panel per task, one line per run set.

    python scripts/diag/plot_loss_curves.py --set 第二套=log/bigym2/main9_recipe --set 第一套=log/bigym2/main9 \
        --set EMA重跑=log/bigym2/main9_recipe_ema --out outputs/loss_curves.html

Each ``--set label=folder`` names a folder of run directories; the run of a task is the one called
``<task>_<variant>`` (``--variant``, default 3cam). Standard library only, so it runs anywhere the
``metrics.jsonl`` files are. A switch above the panels picks the measure: motion loss, or the gradient norm
of the ControlNet or the hand head (before clipping, as ``clip_grad_norm_`` returns it). Lines are the median
of each 500-step stretch (10 logged values), the faint line of the same colour its maximum; log scale, shared
across panels. Shaded: 80k-100k, the checkpoints the ACT/DP-conditions headline pools. Dashed: 0.5, the
divergence threshold of ``loss_spikes.py``, or the clipping threshold for a gradient norm.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loss_spikes import scan  # noqa: E402  scripts/diag/ is on sys.path when run as a script
from summarize_bigym2 import PI05  # noqa: E402  the paper's nine main tasks, in its order

BIN = 500
METRICS = ("motion_loss", "grad_control", "grad_hand")  # the switch's order
COLORS = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70")]  # light, dark


def binned(run: Path) -> dict[str, list[list[float]]]:
    """Per measure, ``[step, median, max]`` per BIN steps of a run's ``metrics.jsonl``."""
    rows = {}
    for line in (run / "metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        rows[record["step"]] = record  # a resumed run repeats steps; keep the last
    curves = {}
    for key in METRICS:
        bins: dict[int, list[float]] = {}
        for step, record in rows.items():
            if record.get(key) is not None:
                bins.setdefault((step - 1) // BIN, []).append(record[key])
        curves[key] = [[(b + 1) * BIN, statistics.median(v), max(v)] for b, v in sorted(bins.items())]
    return curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="sets", action="append", required=True, help="label=folder; up to three")
    parser.add_argument("--variant", default="3cam")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.sets) > len(COLORS):
        raise SystemExit(f"at most {len(COLORS)} sets")

    sets, panels, table = [], [], []
    for index, entry in enumerate(args.sets):
        label, folder = entry.split("=", 1)
        sets.append({"label": label, "light": COLORS[index][0], "dark": COLORS[index][1]})
    for task in PI05:
        curves = []
        for index, entry in enumerate(args.sets):
            label, folder = entry.split("=", 1)
            run = Path(folder) / f"{task}_{args.variant}"
            s = scan(run)
            if s is None:
                continue
            curves.append({"set": index, "points": binned(run)})
            table.append([task, label, s["last_step"], s["first_spike"], s["diverged"], s["max_loss"], s["max_loss_step"], s["mean_last_10pct"], s["max_grad"]])
        panels.append({"task": task, "curves": curves})
    if not any(p["curves"] for p in panels):
        raise SystemExit("no metrics.jsonl found under " + ", ".join(args.sets))

    data = json.dumps({"sets": sets, "panels": panels, "table": table, "sources": ", ".join(args.sets)}, ensure_ascii=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(PAGE.replace("__DATA__", data.replace("</", "<\\/")))
    print(f"wrote {args.out}")


PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>BiGym 训练曲线</title>
<style>
:root { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --band:#f0efea; --chip:#ecebe6; }
@media (prefers-color-scheme: dark) { :root { --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --band:#242422; --chip:#2c2c2a; } }
body { margin:0; background:var(--surface); color:var(--ink); font:14px/1.45 -apple-system, "PingFang SC", "Helvetica Neue", sans-serif; }
main { max-width:1180px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:20px; margin:0 0 4px; } p.note { color:var(--ink2); margin:0 0 16px; max-width:860px; }
.bar { display:flex; gap:16px 24px; flex-wrap:wrap; align-items:center; margin:0 0 16px; }
.switch { display:inline-flex; background:var(--chip); border-radius:8px; padding:3px; }
.switch button { border:0; background:transparent; color:var(--ink2); font:inherit; padding:5px 12px; border-radius:6px; cursor:pointer; }
.switch button[aria-pressed="true"] { background:var(--surface); color:var(--ink); font-weight:600; box-shadow:0 1px 2px rgba(0,0,0,.12); }
.legend { display:flex; gap:20px; flex-wrap:wrap; color:var(--ink2); }
.legend span { display:inline-flex; align-items:center; gap:6px; } .legend i { display:inline-block; width:18px; height:2px; }
.grid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:16px 20px; }
@media (max-width: 760px) { .grid { grid-template-columns:1fr; } }
.panel h2 { font-size:14px; margin:0 0 2px; font-weight:600; } .panel .sub { color:var(--muted); font-size:12px; min-height:16px; }
svg { width:100%; height:auto; display:block; overflow:visible; } svg text { fill:var(--muted); font-size:10px; }
.tip { position:fixed; pointer-events:none; background:var(--surface); border:1px solid var(--grid); border-radius:6px;
       padding:6px 8px; font-size:12px; box-shadow:0 2px 8px rgba(0,0,0,.12); display:none; z-index:2; }
.tip div { display:flex; align-items:center; gap:6px; } .tip b { color:var(--ink); } .tip span { color:var(--ink2); }
.tip i { display:inline-block; width:12px; height:2px; }
table { border-collapse:collapse; margin-top:28px; font-size:12px; width:100%; } th, td { padding:4px 8px; text-align:right; border-bottom:1px solid var(--grid); }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align:left; } th { color:var(--ink2); font-weight:600; }
.wrap { overflow-x:auto; }
</style></head><body><main>
<h1>BiGym 主表 9 个任务：训练曲线（三路相机）</h1>
<p class="note" id="note"></p>
<div class="bar"><div class="switch" id="switch" role="group" aria-label="看哪一项"></div><div class="legend" id="legend"></div></div>
<div class="grid" id="grid"></div>
<div class="wrap"><table id="table"><thead><tr><th>任务</th><th>哪一套</th><th>训到</th><th>损失第一次超过 0.5</th><th>持续发散起点</th><th>损失最高（步）</th><th>最后 10% 平均损失</th><th>梯度最大（ControlNet）</th></tr></thead><tbody></tbody></table></div>
<div class="tip" id="tip"></div>
</main>
<script>
const DATA = __DATA__;
const METRICS = {
  motion_loss: {name: "运动损失", line: 0.5, lineText: "虚线：0.5，超过并持续 500 步算发散。正常水平约 0.001。"},
  grad_control: {name: "梯度大小（ControlNet）", line: 1, lineText: "虚线：1.0，梯度裁剪的阈值；数值是裁剪前的大小，线一直在虚线以上就说明每一步都在被裁剪。"},
  grad_hand: {name: "梯度大小（手部头）", line: 1, lineText: "虚线：1.0，梯度裁剪的阈值；数值是裁剪前的大小。"},
};
const dark = matchMedia("(prefers-color-scheme: dark)").matches;
const color = s => DATA.sets[s][dark ? "dark" : "light"];
const W = 300, H = 180, L = 34, R = 6, T = 6, B = 20, XMAX = 101000;
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent && parent.appendChild(e); return e; };
const k = v => v >= 1000 ? String(Math.round(v / 100) / 10) + "k" : String(v);
const fmt = v => v >= 0.1 ? v.toFixed(2) : v.toPrecision(2);
const x = v => L + (W - L - R) * v / XMAX;

const legend = document.getElementById("legend");
DATA.sets.forEach((s, i) => { const span = document.createElement("span"); const key = document.createElement("i");
  key.style.background = color(i); span.append(key, document.createTextNode(s.label)); legend.appendChild(span); });
const tip = document.getElementById("tip"), grid = document.getElementById("grid"), sw = document.getElementById("switch");
let current = "motion_loss";
try { current = localStorage.getItem("bigym-curve-metric") || current; } catch (e) {}
if (!(current in METRICS)) current = "motion_loss";
for (const key in METRICS) { const b = document.createElement("button"); b.type = "button"; b.textContent = METRICS[key].name; b.dataset.key = key;
  b.addEventListener("click", () => { current = key; try { localStorage.setItem("bigym-curve-metric", key); } catch (e) {} render(); }); sw.appendChild(b); }

function render() {
  const m = METRICS[current];
  for (const b of sw.children) b.setAttribute("aria-pressed", b.dataset.key === current);
  document.getElementById("note").textContent = `每条线是每 500 步${m.name}的中位数，同色淡线是这 500 步里的最大值；纵轴是对数刻度，9 张图刻度相同。` +
    `灰色底：80k–100k（ACT/DP 口径算分的 5 个 checkpoint）。${m.lineText}数据来源：${DATA.sources}`;
  const values = DATA.panels.flatMap(p => p.curves.flatMap(c => (c.points[current] || []).flatMap(q => [q[1], q[2]]))).filter(v => v > 0);
  const YMIN = Math.floor(Math.log10(Math.min(...values, m.line))), YMAX = Math.ceil(Math.log10(Math.max(...values, m.line)));
  const y = v => T + (H - T - B) * (YMAX - Math.log10(Math.min(Math.max(v, 10 ** YMIN), 10 ** YMAX))) / (YMAX - YMIN || 1);
  const step = Math.max(1, Math.ceil((YMAX - YMIN) / 6));
  grid.replaceChildren();
  for (const panel of DATA.panels) {
    const box = document.createElement("div"); box.className = "panel";
    const h = document.createElement("h2"); h.textContent = panel.task;
    const sub = document.createElement("div"); sub.className = "sub";
    sub.textContent = DATA.table.filter(r => r[0] === panel.task && r[4] !== null).map(r => `${r[1]}从 ${k(r[4])} 起发散`).join("；");
    box.append(h, sub); grid.appendChild(box);
    const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${panel.task} ${m.name}`}, box);
    el("rect", {x: x(80000), y: T, width: x(100000) - x(80000), height: H - T - B, fill: "var(--band)"}, svg);
    for (let e = YMIN; e <= YMAX; e += step) {
      el("line", {x1: L, x2: W - R, y1: y(10 ** e), y2: y(10 ** e), stroke: "var(--grid)", "stroke-width": 1}, svg);
      el("text", {x: L - 4, y: y(10 ** e) + 3, "text-anchor": "end"}, svg).textContent = e >= 0 && e <= 3 ? String(10 ** e) : "1e" + e;
    }
    for (const v of [0, 25000, 50000, 75000, 100000]) el("text", {x: x(v), y: H - 6, "text-anchor": "middle"}, svg).textContent = k(v);
    el("line", {x1: L, x2: W - R, y1: y(m.line), y2: y(m.line), stroke: "var(--muted)", "stroke-dasharray": "3 3", "stroke-width": 1}, svg);
    for (const c of panel.curves) {
      const pts = c.points[current] || [];
      if (!pts.length) continue;
      const path = i => pts.map((p, j) => `${j ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[i]).toFixed(1)}`).join("");
      el("path", {d: path(2), fill: "none", stroke: color(c.set), "stroke-width": 1, opacity: 0.3}, svg);
      el("path", {d: path(1), fill: "none", stroke: color(c.set), "stroke-width": 2, "stroke-linejoin": "round"}, svg);
    }
    const cross = el("line", {y1: T, y2: H - B, stroke: "var(--ink2)", "stroke-width": 1, visibility: "hidden"}, svg);
    const hit = el("rect", {x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent"}, svg);
    hit.addEventListener("pointermove", ev => {
      const r = svg.getBoundingClientRect(), s = Math.max(0, (ev.clientX - r.left) / r.width * W - L) / (W - L - R) * XMAX;
      const at = Math.round(s / 500) * 500 || 500;
      cross.setAttribute("x1", x(at)); cross.setAttribute("x2", x(at)); cross.setAttribute("visibility", "visible");
      tip.replaceChildren(); const head = document.createElement("div"); head.textContent = `${panel.task} · 第 ${k(at)} 步 · ${m.name}`; tip.appendChild(head);
      for (const c of panel.curves) { const p = (c.points[current] || []).find(q => q[0] === at); if (!p) continue;
        const row = document.createElement("div"); const key = document.createElement("i"); key.style.background = color(c.set);
        const b = document.createElement("b"); b.textContent = fmt(p[1]); const t = document.createElement("span"); t.textContent = `${DATA.sets[c.set].label}（最高 ${fmt(p[2])}）`;
        row.append(key, b, t); tip.appendChild(row); }
      tip.style.display = "block"; tip.style.left = Math.min(ev.clientX + 12, innerWidth - tip.offsetWidth - 8) + "px"; tip.style.top = ev.clientY + 12 + "px";
    });
    hit.addEventListener("pointerleave", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
  }
}
render();
const body = document.querySelector("#table tbody");
for (const r of DATA.table) { const tr = document.createElement("tr");
  [r[0], r[1], k(r[2]), r[3] === null ? "从未" : k(r[3]), r[4] === null ? "没有" : k(r[4]), `${fmt(r[5])}（${k(r[6])}）`, r[7] === null ? "-" : fmt(r[7]), r[8] === null ? "-" : fmt(r[8])]
    .forEach(v => { const td = document.createElement("td"); td.textContent = v; tr.appendChild(td); }); body.appendChild(tr); }
</script></body></html>
"""

if __name__ == "__main__":
    main()
