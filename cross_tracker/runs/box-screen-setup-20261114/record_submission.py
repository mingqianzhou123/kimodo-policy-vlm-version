"""Record the observed ACP receipt without overwriting live trainer state."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SETUP = Path(__file__).resolve().parent
job = 'kvla-cross-box-s42-20261114'
url = 'https://console.sensecore.cn/cn-sh-01/acp/list'
receipt = json.loads((SETUP / 'submission.json').read_text())
receipt.update(status='submitted', submitted=True, job_id=job, scheduler_state_observed='启动中',
               submitted_at='2026-11-14T15:19:15+08:00', console_url=url,
               recorded_utc=datetime.now(timezone.utc).isoformat())
(SETUP / 'submission.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
plan = json.loads((SETUP / 'plan.json').read_text())
plan.update(status='submitted', job_id=job)
(SETUP / 'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
for order, run_id in enumerate(plan['run_order']):
    target = ROOT / 'runs' / run_id / 'scheduler.json'
    target.write_text(json.dumps({'job_id':job,'execution_order':order+1,'submission':receipt},ensure_ascii=False,indent=2)+'\n')
# This receipt does not assert that DDP has started or modify a concurrently written run.json.
status = ROOT / 'docs/status.md'
s = status.read_text()
line = '\n已于2026-11-14 15:19:15（UTC+8）提交 SenseCore 作业 `kvla-cross-box-s42-20261114`：单节点8×A800，SONIC→TWIST2→双源顺序执行。控制台提交后显示“启动中”；GPU模型运行以实际训练日志为准。108条资产/图像检查、三组真实加载器和正式入口预检均通过。回执见 `runs/box-screen-setup-20261114/submission.json`。\n'
s = s.replace('# 当前进度\n', '# 当前进度\n'+line, 1)
status.write_text(s)
partial = ROOT / 'data/cache/setup/box-inputs-20261114.zip'
if partial.exists():
    partial.rename(partial.with_suffix('.zip.cancelled-partial'))
files = [p for p in SETUP.iterdir() if p.is_file() and p.suffix in ('.json','.log','.py','.sh')]
files += [status, ROOT / 'README.md', ROOT / 'docs/box-screen.md']
for run_id in plan['run_order']:
    files += [ROOT/'runs'/run_id/'run.json',ROOT/'runs'/run_id/'scheduler.json']
archive = ROOT/'data/cache/setup/box-submission-evidence-20261114.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    hashes={}
    for p in files:
        name=str(p.relative_to(ROOT));data=p.read_bytes();z.writestr(name,data)
        hashes[name]=hashlib.sha256(data).hexdigest()
    z.writestr('SHA256.json',json.dumps(hashes,indent=2))
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(files),'job_log_present':(SETUP/'job.log').exists()}),flush=True)
