"""Read-only experiment snapshot and localhost dashboard, behind the existing WebIDE proxy."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, json, os, statistics, threading, time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
ROOT=Path(__file__).resolve().parents[1]
TZ=timezone(timedelta(hours=8))
def read(path,default):
    try: return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError): return default

def snapshot():
    data=read(ROOT/'docs/experiments/bigym2_ledger.json',{})
    state=read(ROOT/'.setup/stability_state.json',{})
    data['curves']={}
    for row in data.get('runs',[]):
        if not row.get('run_dir'): continue
        run=ROOT/row['run_dir']; metrics=run/'metrics.jsonl'; records={}
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                try:
                    r=json.loads(line); records[r['step']]=r
                except json.JSONDecodeError: continue
        values=[records[k] for k in sorted(records)]
        if values:
            last=values[-1]
            row.update(last_step=last['step'],motion_loss=last['motion_loss'],max_motion_loss=max(x['motion_loss'] for x in values),skipped_steps=last.get('skipped_steps'),seconds_per_step=statistics.median(x['s_per_step'] for x in values[-5:]),metrics_updated_at=datetime.fromtimestamp(metrics.stat().st_mtime,TZ).isoformat())
            data['curves'][row['id']]=[{k:r.get(k) for k in ['step','loss','motion_loss','hand_loss','grad_control','skipped_steps','s_per_step']} for r in values]
        config=read(run/'config.json',{}).get('main',{})
        if config: row.update(target_steps=config.get('stop_after') or config.get('max_steps'),seed=config.get('seed'),batch=config.get('batch_size'),micro_batch=config.get('micro_batch_size'))
        if row['id'] in state:
            st=state[row['id']]
            row.update({k:v for k,v in st.items() if k in ['status','stage','reason','next_action','started_at','finished_at']})
            pid=st.get('pid')
            alive=bool(pid and Path(f'/proc/{pid}').exists())
            row['process_alive']=alive
            if row['status'] in ['运行中','评测中'] and not alive:
                row['status']='待核查'; row['next_action']='状态文件显示运行，但对应进程已退出；检查阶段切换或失败日志'
    evaluations={ (e['id'],e['checkpoint'],e.get('seed_base'),e.get('weight','raw')):e for e in data.get('evaluations',[]) }
    for row in data.get('runs',[]):
        if not row.get('run_dir'): continue
        for path in (ROOT/row['run_dir']/'val700000').glob('checkpoint_*.json'):
            r=read(path,{})
            if not all(k in r for k in ['episodes','seed_base','success_rate']): continue
            e=dict(id=row['id'],task=row['task'],checkpoint=int(path.stem.split('_')[-1]),seed_base=r['seed_base'],episodes=r['episodes'],execution_frames=50,success_rate=r['success_rate'],role='流程检查' if row['id']=='BG5090-001' else '验证调参',source=str(path.relative_to(ROOT)))
            evaluations[(e['id'],e['checkpoint'],e['seed_base'],'raw')]=e
    data['evaluations']=list(evaluations.values())
    data['events']=read(ROOT/'docs/experiments/events.json',[])
    data['notes']=read(ROOT/'docs/experiments/research_notes.json',[])
    data['updated_at']=datetime.now(TZ).isoformat(timespec='seconds')
    data['meta']={'mode':'live','host':'T2M-PLUS · 8 × RTX 5090','refresh_seconds':30,'source':'训练目录 config.json / metrics.jsonl / val700000 + 项目历史文档','official_test_reserved':True}
    return data

def write_snapshot(target):
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_suffix('.tmp'); tmp.write_text(json.dumps(snapshot(),ensure_ascii=False)); tmp.replace(target)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--export',type=Path); p.add_argument('--serve',type=Path); p.add_argument('--port',type=int,default=8765); a=p.parse_args()
    if a.export: write_snapshot(a.export)
    if a.serve:
        directory=a.serve.resolve(); write_snapshot(directory/'data.json')
        def update():
            while True:
                time.sleep(30)
                try: write_snapshot(directory/'data.json')
                except Exception as e: print('Snapshot failed:',e,flush=True)
        threading.Thread(target=update,daemon=True).start()
        class Handler(SimpleHTTPRequestHandler):
            notes_lock=threading.Lock()
            def __init__(self,*args,**kwargs): super().__init__(*args,directory=str(directory),**kwargs)
            def do_POST(self):
                if self.path.rstrip('/') != '/notes':
                    self.send_error(404); return
                try:
                    if self.headers.get('Content-Type','').split(';')[0] != 'application/json': raise ValueError('Expected JSON')
                    size=int(self.headers.get('Content-Length','0'))
                    if not 0 < size <= 32000: raise ValueError('Invalid body size')
                    note=json.loads(self.rfile.read(size))
                    if not isinstance(note,dict): raise ValueError('Expected object')
                    if note.get('kind') not in ['观察事实','待验证假设','下一实验计划']: raise ValueError('Invalid kind')
                    limits={'id':100,'kind':30,'run':180,'title':180,'body':6000,'source':600,'time':100}
                    clean={k:note.get(k,'') for k in limits}
                    if any(not isinstance(v,str) or len(v)>limits[k] for k,v in clean.items()): raise ValueError('Invalid field')
                    if not all(clean[k].strip() for k in ['id','title','body']): raise ValueError('Missing field')
                    target=ROOT/'docs/experiments/research_notes.json'
                    with self.notes_lock:
                        notes=read(target,[])
                        if not any(n['id']==clean['id'] for n in notes): notes.append(clean)
                        tmp=target.with_suffix('.tmp'); tmp.write_text(json.dumps(notes,ensure_ascii=False)); tmp.replace(target)
                    payload=json.dumps(notes,ensure_ascii=False).encode()
                    self.send_response(200); self.send_header('Content-Type','application/json; charset=utf-8'); self.end_headers(); self.wfile.write(payload)
                except (ValueError,KeyError,json.JSONDecodeError): self.send_error(400,'Invalid note')
            def end_headers(self):
                self.send_header('Cache-Control','no-store'); super().end_headers()
        print(f'Experiment dashboard http://127.0.0.1:{a.port}',flush=True)
        ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
