"""Independently audit completed diagnostics, real state times and shared scene seeds."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'data/cache/python'))
import av
from PIL import Image
folders = [root / 'runs/sim-closed-loop-20261113-r9/sonic']
folders += sorted((root / 'runs/sim-closed-loop-20261113-r10').iterdir())
assert len(folders) == 6
records = []
seeds = {}
for folder in folders:
    run = json.loads((folder / 'run.json').read_text())
    episode = json.loads((folder / 'episode.json').read_text())
    assert run['status'] == 'completed' and run['exit_code'] == 0
    assert not episode.get('error')
    assert (folder / 'arena_history.py').read_bytes() == (root / 'src/arena_history.py').read_bytes()
    history = [json.loads(line) for line in (folder / 'state-history.jsonl').read_text().splitlines()]
    assert len(history) == run['successful_inference_requests'] > 0
    times = [t for packet in history for t in packet['times']]
    np.testing.assert_allclose(np.diff(times), 0.02, rtol=0, atol=1e-6)
    assert all(np.asarray(packet['states']).shape == (len(packet['times']),64) for packet in history)
    video = Path(episode['video_path'])
    assert video.is_file() and video.stat().st_size > 0
    frame_count = 0
    with av.open(str(video)) as movie:
        for frame in movie.decode(video=0):
            if frame_count == 0:
                rgb = frame.to_ndarray(format='rgb24')
                assert rgb.std() > 1
                Image.fromarray(rgb).save(folder / 'first-frame.png')
            frame_count += 1
    assert frame_count >= episode['episode_steps'] + 1
    seed_key = episode['task']
    object_seed = episode['episode_object_seed']
    assert seeds.setdefault(seed_key, object_seed) == object_seed
    assert episode['episode_object_seed_source'] == 'env_seed'
    records.append({'run':str(folder.relative_to(root)), 'controller':run['controller'],
        'task':run['task'], 'steps':episode['episode_steps'], 'inference_requests':len(history),
        'history_packet_sizes':[len(p['times']) for p in history],
        'object_seed':object_seed, 'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
        'video_frames':frame_count, 'task_success':episode['success'], 'termination':episode['failure_reason']})
result = {'passed':True, 'meaning':'short closed-loop diagnostics only; not a baseline success-rate table',
          'bridge_sha256':hashlib.sha256((root / 'src/arena_history.py').read_bytes()).hexdigest(),
          'runs':records}
(Path(__file__).resolve().parent / 'closed-loop-audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
