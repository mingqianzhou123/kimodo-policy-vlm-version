"""Download official, revision-pinned Qwen files and verify upstream content hashes."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import requests
from huggingface_hub import model_info


def digest(path,kind='sha256',git_blob=False):
    h=hashlib.new(kind)
    if git_blob:h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('checkpoints/Qwen3-VL-4B-Instruct'))
    p.add_argument('--revision',required=True)
    a=p.parse_args();repo='Qwen/Qwen3-VL-4B-Instruct'
    for attempt in range(5):
        try:info=model_info(repo,revision=a.revision,files_metadata=True);break
        except Exception:
            if attempt==4:raise
            time.sleep(2)
    revision=info.sha;assert revision==a.revision
    print('PINNED',repo,revision,flush=True);a.out.mkdir(parents=True,exist_ok=True)
    upstream={};files={}
    for item in info.siblings:
        name=item.rfilename
        if '/' in name or not name.endswith(('.json','.safetensors','.txt','.jinja','.model')):continue
        dest=a.out/name
        expected=item.lfs.sha256 if item.lfs else item.blob_id
        kind='sha256' if item.lfs else 'sha1'
        valid=lambda p:p.is_file() and p.stat().st_size==item.size and digest(p,kind,not item.lfs)==expected
        if not valid(dest):
            part=dest.with_suffix(dest.suffix+'.part')
            for attempt in range(5):
                try:
                    offset=part.stat().st_size if part.exists() else 0
                    headers={'Range':f'bytes={offset}-'} if offset else {}
                    url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
                    with requests.get(url,headers=headers,stream=True,timeout=(20,120)) as response:
                        response.raise_for_status()
                        append=offset>0 and response.status_code==206
                        if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                            raise ValueError('Unexpected partial response')
                        with part.open('ab' if append else 'wb') as f:
                            for block in response.iter_content(4<<20):f.write(block)
                    if not valid(part):raise ValueError(f'Upstream hash/size mismatch: {name}')
                    part.replace(dest);break
                except (requests.RequestException,ValueError) as exc:
                    print('RETRY',name,attempt,type(exc).__name__,flush=True)
                    if attempt==4:raise
                    time.sleep(2)
        files[name]=digest(dest);upstream[name]={'size':item.size,'hash':expected,'kind':kind}
        print('VERIFIED',name,flush=True)
    manifest={'repo_id':repo,'revision':revision,'files_sha256':files,'upstream':upstream}
    (a.out/'download-manifest.json').write_text(json.dumps(manifest,indent=2))
    print('READY',str(a.out),len(files),flush=True)

if __name__=='__main__':main()
