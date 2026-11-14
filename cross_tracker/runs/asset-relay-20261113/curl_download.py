from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, subprocess, sys
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'scripts'))
from fetch_assets import matches
records=json.loads((root/'data/manifests/weights-v1.json').read_text())['files']
def fetch(item):
    name,record=item; dest=root/'data/cache/weights'/name
    if dest.exists():
        assert matches(dest,record), name
        return name,'verified'
    dest.parent.mkdir(parents=True,exist_ok=True)
    partial=dest.with_name(dest.name+'.partial')
    command=['curl','--location','--fail','--silent','--show-error','--retry','8','--retry-all-errors','--retry-delay','2','--connect-timeout','20','--max-time','1800','--continue-at','-','--output',str(partial),record['url']]
    subprocess.run(command,check=True)
    assert matches(partial,record), name
    partial.rename(dest)
    return name,'downloaded'
with ThreadPoolExecutor(max_workers=4) as pool:
    jobs=[pool.submit(fetch,item) for item in records.items()]
    for future in as_completed(jobs): print(json.dumps(future.result()),flush=True)
