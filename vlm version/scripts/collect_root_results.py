"""Finish the three authorized cells, validate them, and package immutable evidence."""
import hashlib,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
deadline=time.monotonic()+3600
while True:
    logs=[ROOT/f'root-{c}-r1.log' for c in ('load','support','both')]
    if all('CELL COMPLETE' in p.read_text() for p in logs):break
    if any('Traceback (most recent call last)' in p.read_text() for p in logs):raise RuntimeError('A cell failed; preserve logs and investigate')
    if time.monotonic()>deadline:raise TimeoutError('Cells did not complete within one hour')
    time.sleep(10)
with (ROOT/'root-report-check.log').open('w') as log:
    subprocess.run([sys.executable,'scripts/report_root_dynamics.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
files=[f for p in (ROOT/'runs').glob('root-*') if p.is_dir() for f in p.rglob('*') if f.is_file() and not any(x in f.parts for x in ('runtime','__pycache__'))]
files+=[ROOT/'docs/root-dynamics-20261120.md']
files+=list(ROOT.glob('root-*.log'))
p=ROOT/'root-evidence-20261120.zip'
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for f in files:z.write(f,str(f.relative_to(ROOT)))
print('COMPLETE',len(files),p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
