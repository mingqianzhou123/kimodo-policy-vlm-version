"""Watch the requested CCT job; validate and relay its final checkpoint when ready."""
from pathlib import Path
import hashlib
import json
import subprocess
import time
import zipfile

HERE = Path(__file__).resolve().parent
DEST = HERE.parents[1] / 'data/cache/scratch-finals-20261118/cct'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', '1028', 'root@103.237.29.246']
REMOTE_PY = '/data/NTU_slab/mingqianzhou/humanoid_vla/envs/kimodo/bin/python'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def main():
    previous = None
    while True:
        try:
            check = subprocess.run(SSH + [REMOTE_PY + ' -'], input=(HERE / 'verify_cct_final.py').read_text(),
                                   text=True, capture_output=True, timeout=300)
            if check.returncode:
                raise RuntimeError(check.stderr[-2000:])
            receipt = json.loads(check.stdout)
            atomic_json(HERE / 'cct-latest.json', receipt)
            step = receipt['latest_metric']['step']
            status = (receipt['status'], step // 5000)
            if status != previous:
                print(time.strftime('%F %T'), receipt['status'], step, flush=True)
                previous = status
            if receipt['status'] == 'verified_final':
                break
        except Exception as error:
            atomic_json(HERE / 'cct-watch-error.json', dict(checked_unix=time.time(), error=repr(error)))
            print('CCT_CHECK_ERROR', repr(error), flush=True)
        time.sleep(300)
    DEST.mkdir(parents=True, exist_ok=True)
    for name, info in receipt['files'].items():
        target = DEST / name
        if target.exists():
            assert sha(target) == info['sha256'], f'Existing final differs: {target}'
            continue
        part = target.with_suffix(target.suffix + '.partial')
        while True:
            try:
                copy = subprocess.run(['scp', '-P', '1028', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                    'root@103.237.29.246:' + receipt['checkpoint'] + '/' + name, str(part)],
                    text=True, capture_output=True, timeout=7200)
            except subprocess.TimeoutExpired:
                print('CCT_RELAY_TIMEOUT', name, flush=True)
                time.sleep(30)
                continue
            if copy.returncode == 0 and part.stat().st_size == info['bytes'] and sha(part) == info['sha256']:
                part.replace(target)
                print('CCT_RELAY_VERIFIED', name, info['bytes'], flush=True)
                break
            print('CCT_RELAY_RETRY', name, copy.stderr[-500:], flush=True)
            time.sleep(30)
    archive = DEST.parent / 'scratch-cct-200k-20261118.zip'
    if not archive.exists():
        partial_archive = archive.with_suffix('.zip.partial')
        with zipfile.ZipFile(partial_archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as package:
            for name in receipt['files']:
                package.write(DEST / name, name)
        partial_archive.replace(archive)
    with zipfile.ZipFile(archive) as package:
        for name, info in receipt['files'].items():
            assert hashlib.sha256(package.read(name)).hexdigest() == info['sha256']
    atomic_json(HERE / 'cct-relay-ready.json', dict(verified_unix=time.time(), checkpoint=receipt,
        archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive)))
    print('CCT_FINAL_READY_FOR_BAIDU_UPLOAD', archive, flush=True)


if __name__ == '__main__':
    main()
