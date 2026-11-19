"""Copy the verified R4 source snapshot for parallel scheduling; do not start jobs."""
from pathlib import Path
import hashlib
import json
import shutil
import time

ROOT = Path('/root/workspace/humanoidVLA/cross_tracker')
OLD = ROOT / 'runs/scratch-eval-20261118-r4'
RUN = ROOT / 'runs/scratch-eval-20261119-r5'
WATCH = ROOT / 'runs/scratch-watch-20261118'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    sources = json.loads((OLD / 'source-files.json').read_text())
    assets = json.loads((OLD / 'shared-eval-assets.json').read_text())
    context = json.loads((OLD / 'cct-context-files.json').read_text())
    for name, digest in sources.items():
        assert sha(OLD / 'code' / name) == digest, name
    for name, digest in assets.items():
        assert sha(ROOT / 'data/cache' / name) == digest, name
    for name, digest in context.items():
        assert sha(OLD / 'code/cross_tracker/data/cache/cct-box-v1' / name) == digest, name
    RUN.mkdir(exist_ok=False)
    code = RUN / 'code'
    for name in sources:
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OLD / 'code' / name, target)
    cache = code / 'cross_tracker/data/cache'
    cache.mkdir(parents=True)
    for path in (OLD / 'code/cross_tracker/data/cache').iterdir():
        (cache / path.name).symlink_to(path.resolve())
    (code / 'checkpoints').symlink_to((OLD / 'code/checkpoints').resolve())
    for name in ('source-files.json', 'shared-eval-assets.json', 'cct-context-files.json',
                 'sonic-manifest.json', 'twist2-manifest.json', 'mix-manifest.json', 'protocol.json'):
        shutil.copy2(OLD / name, RUN / name)
    # The existing final importer publishes atomically to R4; both queues use that exact manifest.
    (RUN / 'cct-manifest.json').symlink_to(OLD / 'cct-manifest.json')
    files = {'run_batch.py': WATCH / 'run_parallel.py', 'parallel_state.py': WATCH / 'parallel_state.py',
             'episode_protocol.py': OLD / 'run_batch.py'}
    for name, source in files.items():
        compile(source.read_text(), str(source), 'exec')
        shutil.copy2(source, RUN / name)
    runners = {name: sha(RUN / name) for name in files}
    (RUN / 'runner-sources.json').write_text(json.dumps(runners, indent=2))
    (RUN / 'enabled-pairs.json').write_text(json.dumps([[6, 7]]) + '\n')
    (RUN / 'prepare.json').write_text(json.dumps(dict(status='complete', finished_unix=time.time(),
        previous=str(OLD), code=str(code), source_files=len(sources), source_hashes_unchanged=True,
        protocol_unchanged=True, runner_sources=runners,
        activation='Not started. Requires audited idle-boundary handoff; extra GPU pairs remain disabled.'), indent=2))
    print('PREPARED_R5', len(sources), runners, flush=True)


if __name__ == '__main__':
    main()
