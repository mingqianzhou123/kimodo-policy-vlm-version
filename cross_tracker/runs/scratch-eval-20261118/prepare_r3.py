"""Resume pending scenes in a new snapshot; wait for busy GPUs instead of failing."""
from pathlib import Path
import hashlib
import json
import shutil
import time

ROOT = Path('/root/workspace/humanoidVLA')
OLD = ROOT / 'cross_tracker/runs/scratch-eval-20261118-r2'
RUN = ROOT / 'cross_tracker/runs/scratch-eval-20261118-r3'

GPU_WAIT = '''    while True:
        gpu_uuids = {}
        busy = []
        for device in (args.sim_gpu, args.policy_gpu):
            values = subprocess.check_output(["nvidia-smi", f"--id={device}",
                "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
            uuid, memory, utilization = map(str.strip, values.split(","))
            if int(memory) >= 100 or int(utilization) != 0:
                busy.append((device, values.strip()))
            gpu_uuids[device] = uuid
        if not busy:
            break
        print("WAITING_FOR_IDLE_GPUS", busy, flush=True)
        time.sleep(15)
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous = json.loads((OLD / 'batch.json').read_text())
    assert previous['status'] == 'partial_failure'
    assert not Path('/proc/' + str(previous['pid'])).exists()
    completed = previous['completed']
    keys = {(x['method'], x['controller'], x['group'], x['repeat']) for x in completed}
    assert len(keys) == len(completed) == 5
    for item in completed:
        path = OLD / 'code/cross_tracker/runs' / item['run_id'] / 'run.json'
        receipt = json.loads(path.read_text())
        assert receipt['status'] == 'completed' and receipt['valid_for_success_rate']
        for key, value in receipt['episodes'][0].items():
            assert item[key] == value, (item['run_id'], key)
    RUN.mkdir(exist_ok=False)
    code = RUN / 'code'
    sources = json.loads((OLD / 'source-files.json').read_text())
    for name, digest in sources.items():
        src, dst = OLD / 'code' / name, code / name
        assert sha(src) == digest, name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    launcher = code / 'cross_tracker/scripts/evaluate_box.py'
    text = launcher.read_text()
    start = text.index('    gpu_uuids = {}\n')
    end = text.index('    run.mkdir(exist_ok=False)\n', start)
    assert 'assert memory < 100 and utilization == 0' in text[start:end]
    text = text[:start] + GPU_WAIT + text[end:]
    compile(text, str(launcher), 'exec')
    launcher.write_text(text)
    sources[str(launcher.relative_to(code))] = sha(launcher)
    (code / 'cross_tracker/data').mkdir()
    (code / 'cross_tracker/data/cache').symlink_to(ROOT / 'cross_tracker/data/cache')
    (code / 'checkpoints').symlink_to((ROOT / 'checkpoints').resolve())
    for name in ['sonic-manifest.json', 'twist2-manifest.json', 'mix-manifest.json',
                 'local-verification.json', 'shared-eval-assets.json']:
        shutil.copy2(OLD / name, RUN / name)
    (RUN / 'previous-batch.json').write_text(json.dumps(previous, indent=2))
    (RUN / 'carried-completed.json').write_text(json.dumps(completed, indent=2))
    batch = (OLD / 'run_batch.py').read_text().replace('-20261118-r2', '-20261118-r3')
    assert batch.count('completed=[],failed=[]') == 1
    batch = batch.replace('completed=[],failed=[]',
        "completed=json.loads((RUN/'carried-completed.json').read_text()),failed=[]")
    anchor = '                for method,controller in COMBINATIONS:\n'
    assert batch.count(anchor) == 1
    batch = batch.replace(anchor, anchor +
        "                    if any(x['method']==method and x['controller']==controller "
        "and x['group']==group and x['repeat']==repeat for x in state['completed']):continue\n")
    compile(batch, str(RUN / 'run_batch.py'), 'exec')
    (RUN / 'run_batch.py').write_text(batch)
    (RUN / 'source-files.json').write_text(json.dumps(sources, indent=2))
    receipt = json.loads((OLD / 'prepare.json').read_text())
    receipt.update(code=str(code), source_files=len(sources), finished_unix=time.time(),
                   previous_failed_batch=str(OLD), carried_valid_episodes=len(completed),
                   fix='Wait for both GPUs to be idle; skip previously completed scenes; no model/protocol changes')
    (RUN / 'prepare.json').write_text(json.dumps(receipt, indent=2))
    print('PREPARED_R3', len(sources), 'CARRIED', len(completed), 'PENDING', 360-len(completed))


if __name__ == '__main__':
    main()
