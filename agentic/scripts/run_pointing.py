"""Run one pointing backend over the image set, then score it (experiment 3).

  run_pointing.py run --dataset <set> --backend qwen --localizer single --out <run dir> --gpu 5
  run_pointing.py evaluate --dataset <set> --run <run dir>

``run`` reads only <set>/inputs and writes one record per frame to <run dir>/calls.jsonl; it resumes
if interrupted. ``evaluate`` is the only step that opens <set>/truth.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess

from paths import AGENTIC as ROOT, CHECKPOINTS, REPO


def run(a):
    if a.backend == 'qwen':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    from PIL import Image
    from kimodo_vla.vlm.backends import API_BACKENDS, QwenBackend
    from kimodo_vla.vlm.pointing import LOCALIZERS, PROMPT
    frames = [json.loads(line) for line in (a.dataset/'inputs/frames.jsonl').read_text().splitlines()][:a.limit]
    config = json.loads(a.config.read_text())
    backend = QwenBackend(a.qwen_model or CHECKPOINTS/'Qwen3-VL-4B-Instruct') if a.backend == 'qwen' else API_BACKENDS[a.backend](config[a.backend])
    a.out.mkdir(parents=True, exist_ok=True)
    calls = a.out/'calls.jsonl'
    done = {json.loads(line)['frame_id'] for line in calls.read_text().splitlines()} if calls.exists() else set()
    record = {'backend': a.backend, 'localizer': a.localizer, 'origin': a.origin, 'dataset': a.dataset.name,
              'frames_sha256': subprocess.check_output(['git', 'hash-object', str(a.dataset/'inputs/frames.jsonl')], text=True).strip(),
              'prompt_template': PROMPT, 'backend_info': backend.info, 'resumed_frames': len(done),
              'git_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
              'git_dirty': bool(subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()),
              'started': datetime.now(timezone.utc).isoformat()}
    with calls.open('a') as log:
        for frame in frames:
            if frame['id'] in done:
                continue
            image = Image.open(a.dataset/'inputs'/frame['image']).convert('RGB')
            result = LOCALIZERS[a.localizer](backend, image, frame['instruction'])
            if any(g.get('http_status', 200) != 200 for g in result['generations']):
                # A paid interface that fails once is not asked again: keep the failed call apart and stop.
                # The frame stays undone, so the next run resumes with it.
                with (a.out/'errors.jsonl').open('a') as errors:
                    errors.write(json.dumps({'frame_id': frame['id'], 'called_at': datetime.now(timezone.utc).isoformat(), **result})+'\n')
                raise SystemExit(f"{frame['id']}: the interface returned an error; see {a.out/'errors.jsonl'}")
            log.write(json.dumps({'frame_id': frame['id'], 'instruction': frame['instruction'], 'backend': a.backend,
                'model': backend.info['model'], 'localizer': a.localizer, 'origin': a.origin,
                'called_at': datetime.now(timezone.utc).isoformat(), **result})+'\n')
            log.flush()
            print(frame['id'], f"{result['latency_s']:.2f}s", json.dumps(result['answer']), flush=True)
    if hasattr(backend, 'memory'):
        record['gpu_memory'] = backend.memory()
    record['finished'] = datetime.now(timezone.utc).isoformat()
    (a.out/'run.json').write_text(json.dumps(record, indent=1))


def evaluate(a):
    from kimodo_vla.vlm import pointing_eval as ev
    frames, meshes = ev.load_dataset(a.dataset)
    frames = {f['id']: f for f in frames}
    rows = []
    for line in (a.run/'calls.jsonl').read_text().splitlines():
        call = json.loads(line)
        frame = frames[call['frame_id']]
        masks, geometry = ev.load_masks(a.dataset, frame['id']), ev.load_geometry(a.dataset, frame['id'])
        generations = call['generations']
        total = lambda key: sum(g[key] for g in generations) if all(g.get(key) is not None for g in generations) else None
        rows.append({'frame_id': frame['id'], 'category': frame['category'], 'scene': frame['scene'],
            'contact': ev.score_point(call['answer']['contact_point'], frame, masks, geometry, meshes),
            'pregrasp': ev.score_point(call['answer']['pregrasp_point'], frame, masks, geometry, meshes),
            'format_ok': call['answer']['format_ok'], 'action': call['answer']['action'], 'latency_s': call['latency_s'],
            'generations': len(generations), 'http_errors': sum(g.get('http_status', 200) != 200 for g in generations),
            'input_tokens': total('input_tokens'), 'output_tokens': total('output_tokens'), 'cost_usd': total('cost_usd')})
    summary = ev.summarize(rows)
    first = [r['latency_s'] for r in rows[:50]]
    summary.update(format_ok_rate=sum(r['format_ok'] for r in rows)/len(rows), http_errors=sum(r['http_errors'] for r in rows),
        latency_first_50_s={'calls': len(first), 'median': ev.percentile(first, 50), 'p90': ev.percentile(first, 90), 'max': max(first)},
        run=json.loads((a.run/'run.json').read_text()) if (a.run/'run.json').exists() else None,
        evaluator_git_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip())
    (a.run/'scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (a.run/'summary.json').write_text(json.dumps(summary, indent=1))
    brief = {k: summary[k] for k in ('frames', 'both_hit_3d_rate', 'latency_s', 'latency_first_50_s', 'format_ok_rate', 'http_errors')}
    for key in ('contact', 'pregrasp'):
        brief[key] = summary[key]
    brief['contact_hit_3d_by_category'] = {c: v['contact']['hit_3d_rate'] for c, v in summary['by_category'].items()}
    print(json.dumps(brief, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest='command', required=True)
    r = commands.add_parser('run')
    r.add_argument('--dataset', type=Path, required=True)
    r.add_argument('--backend', choices=['qwen', 'openai', 'gemini'], required=True)
    r.add_argument('--localizer', choices=['single', 'two-scale'], default='single')
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--limit', type=int, help='Only the first N frames')
    r.add_argument('--gpu', type=int, default=0, help='GPU for the local model')
    r.add_argument('--qwen-model', type=Path, help='Default: <KVLA_CHECKPOINTS>/Qwen3-VL-4B-Instruct')
    r.add_argument('--config', type=Path, default=ROOT/'configs/pointing-backends.json')
    r.add_argument('--origin', default=platform.node(), help='Where the calls are sent from; recorded with every call')
    r.set_defaults(func=run)
    e = commands.add_parser('evaluate')
    e.add_argument('--dataset', type=Path, required=True)
    e.add_argument('--run', type=Path, required=True)
    e.set_defaults(func=evaluate)
    a = p.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
