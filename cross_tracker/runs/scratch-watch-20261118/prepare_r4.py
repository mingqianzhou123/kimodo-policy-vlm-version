"""Build immutable evaluation snapshot; preserve R3 rollout protocol and add explicit CCT serving."""
from pathlib import Path
import hashlib
import json
import shutil
import time

ROOT = Path('/root/workspace/humanoidVLA')
OLD = ROOT / 'cross_tracker/runs/scratch-eval-20261118-r3'
WATCH = ROOT / 'cross_tracker/runs/scratch-watch-20261118'
RUN = ROOT / 'cross_tracker/runs/scratch-eval-20261118-r4'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence = json.loads((WATCH / 'cct-preparation-source.json').read_text())
    context = WATCH / 'cct-context'
    selected = ['metadata.pt'] + [evidence['cache_manifest']['episodes'][i]['file']
        for ids in evidence['selection']['episodes'].values() for i in ids]
    for name in selected:
        assert sha(context / name) == evidence['cache_manifest']['files'][name], name
    assert json.loads((context / 'manifest.json').read_text()) == evidence['cache_manifest']
    RUN.mkdir(exist_ok=False)
    code = RUN / 'code'
    sources = json.loads((OLD / 'source-files.json').read_text())
    for name, digest in sources.items():
        src, dst = OLD / 'code' / name, code / name
        assert sha(src) == digest, name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for src in (WATCH / 'sensecore-source').rglob('*'):
        if src.is_file():
            name = str(src.relative_to(WATCH / 'sensecore-source'))
            assert sha(src) == evidence['source_files'][name], name
            dst = code / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            sources[name] = sha(dst)
    for name in sources:
        if name.startswith(('kimodo_vla/', 'cross_tracker/src/cct_')) and name in evidence['source_files']:
            assert sha(code / name) == evidence['source_files'][name], name
    launcher = code / 'cross_tracker/scripts/evaluate_box.py'
    text = launcher.read_text()
    anchor = '    args = parser.parse_args()\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '    parser.add_argument("--cct", action="store_true")\n' + anchor)
    anchor = '    env = {k: v for k, v in os.environ.items()\n'
    assert text.count(anchor) == 1
    branch = '''    if args.cct:
        settings = json.loads((ROOT / "configs/box-screen.json").read_text())["evaluation_policy"]
        assert settings["dtype"] == "fp32" and settings["execution_frames_30hz"] == 15
        assert settings["diffusion_steps"] == 10 and settings["rtc_overlap_frames"] == 0
        server_command = [sys.executable, "-u", str(ROOT / "scripts/serve_cct.py"),
            "--checkpoint", str(checkpoint), "--text-cache", str(CACHE / "text-eval"),
            "--controller", args.controller, "--port", str(port), "--timing", args.timing]
        if args.deterministic:
            server_command += ["--deterministic"]
'''
    text = text.replace(anchor, branch + anchor)
    compile(text, str(launcher), 'exec')
    launcher.write_text(text)
    sources[str(launcher.relative_to(code))] = sha(launcher)
    cache = code / 'cross_tracker/data/cache'
    cache.mkdir(parents=True)
    for src in (ROOT / 'cross_tracker/data/cache').iterdir():
        if src.name != 'cct-box-v1':
            (cache / src.name).symlink_to(src.resolve())
    shutil.copytree(context, cache / 'cct-box-v1')
    (code / 'checkpoints').symlink_to((ROOT / 'checkpoints').resolve())
    for name in ['sonic-manifest.json', 'twist2-manifest.json', 'mix-manifest.json',
                 'local-verification.json', 'shared-eval-assets.json']:
        shutil.copy2(OLD / name, RUN / name)
    shutil.copy2(WATCH / 'run_batch.py', RUN / 'run_batch.py')
    protocol = json.loads((OLD / 'batch.json').read_text())['protocol']
    (RUN / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    (RUN / 'source-files.json').write_text(json.dumps(sources, indent=2))
    (RUN / 'cct-context-files.json').write_text(json.dumps({n: sha(context / n) for n in selected}, indent=2))
    (RUN / 'prepare.json').write_text(json.dumps(dict(status='complete', finished_unix=time.time(),
        previous=str(OLD), code=str(code), source_files=len(sources),
        context_selection=evidence['selection'], pending='CCT final 200000 manifest and GPU validation'), indent=2))
    print('PREPARED_R4', len(sources), 'CCT_CONTEXT', len(selected), flush=True)


if __name__ == '__main__':
    main()
