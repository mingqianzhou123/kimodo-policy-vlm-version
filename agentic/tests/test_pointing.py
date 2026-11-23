"""Pointing test (experiment 3): answer format, crop mapping, truth geometry, scoring and backend records."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from kimodo_vla.vlm import backends, pointing, pointing_eval as ev

W, H, F = 640, 480, 240.
CUBE = np.array([[x, y, z] for x in (-.1, .1) for y in (-.1, .1) for z in (-.1, .1)])
FACES = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])


def cube_mesh(graspable=None):
    return {'vertices': CUBE.astype(np.float32), 'faces': FACES,
            'graspable': np.ones(len(FACES), bool) if graspable is None else graspable}


def render(objects):
    """Depth of axis-aligned 20 cm cubes in front of a wall 2 m away; the camera looks along world +Z."""
    K = np.array([[F, 0, (W-1)/2], [0, F, (H-1)/2], [0, 0, 1]])
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(u-K[0, 2])/F, (v-K[1, 2])/F], -1)
    depth = np.full((H, W), 2.)
    for obj in objects.values():
        cx, cy, cz = obj['position']
        front = cz-.1
        hit = (np.abs(rays[..., 0]*front-cx) <= .1) & (np.abs(rays[..., 1]*front-cy) <= .1) & (front < depth)
        depth[hit] = front
    return {'depth': depth, 'K': K, 'camera_position': np.zeros(3), 'camera_rotation': np.eye(3)}


def scene(**positions):
    objects = {name: {'asset': 'cube', 'position': list(p), 'quat_wxyz': [1, 0, 0, 0]} for name, p in positions.items()}
    return {'id': 'f', 'scene': 's', 'category': 'distance', 'target': 'a', 'instruction': 'Pick up the cube.', 'objects': objects}


def normalized(x, y):
    return [x/(W-1)*1000, y/(H-1)*1000]


def test_answer_parsing_keeps_usable_fields_and_reports_format():
    exact = pointing.parse_answer('{"contact_point": [500, 400], "pregrasp_point": [480.5, 400], "action": "Reach forward."}')
    assert exact == {'contact_point': [500, 400], 'pregrasp_point': [480.5, 400], 'action': 'Reach forward.', 'format_ok': True}
    wrapped = pointing.parse_answer('Sure:\n```json\n{"contact_point": [1, 2], "pregrasp_point": [3, 4], "action": "Go."}\n```')
    assert wrapped['contact_point'] == [1, 2] and not wrapped['format_ok']
    partial = pointing.parse_answer('{"contact_point": [500, 1200], "pregrasp_point": [3, 4]}')
    assert partial['contact_point'] is None and partial['pregrasp_point'] == [3, 4] and partial['action'] is None
    assert pointing.parse_answer('no json here')['contact_point'] is None
    assert pointing.parse_answer('{"contact_point": [true, 2], "pregrasp_point": "x", "action": ""}')['contact_point'] is None


class CentreBackend:
    """Answers the full image with fixed points and every crop with a fixed crop coordinate."""
    def __init__(self, crop_point):
        self.crop_point, self.images = crop_point, []

    def generate(self, image, prompt, max_tokens):
        self.images.append(image.size)
        text = ('{"contact_point": [500, 500], "pregrasp_point": [400, 500], "action": "Reach."}' if 'contact_point' in prompt
                else json.dumps({'point': self.crop_point}))
        return {'text': text, 'latency_s': .1}


def test_two_scale_runs_five_generations_and_maps_crops_back():
    image = Image.new('RGB', (W, H))
    single = pointing.locate_single(CentreBackend([500, 500]), image, 'Pick up the cube.')
    assert len(single['generations']) == 1 and single['answer']['contact_point'] == [500, 500]
    backend = CentreBackend([500, 500])
    result = pointing.locate_two_scale(backend, image, 'Pick up the cube.')
    assert [g['stage'] for g in result['generations']] == ['full', 'contact_point-192', 'contact_point-256', 'pregrasp_point-192', 'pregrasp_point-256']
    assert backend.images == [(W, H)]+[(640, 640)]*4
    # The crop centre is the coarse point to within the rounding of the crop window.
    for key in pointing.POINTS:
        refined = np.array(result['answer'][key])/1000*[W-1, H-1]
        coarse = np.array(result['coarse_answer'][key])/1000*[W-1, H-1]
        assert np.abs(refined-coarse).max() <= 1 and result['refinement'][key]['disagreement_px'] <= 1
    shifted = pointing.locate_two_scale(CentreBackend([1000, 500]), image, 'Pick up the cube.')
    moved = np.array(shifted['answer']['contact_point'])[0]/1000*(W-1)-.5*(W-1)
    assert 100 < moved < 130  # right edge of the 192 and 256 px crops, averaged
    missing = pointing.locate_two_scale(CentreBackend(None), image, 'Pick up the cube.')
    assert missing['answer']['contact_point'] == [500, 500] and missing['refinement']['contact_point']['estimates_px'] == []


def test_triangle_distance_is_exact():
    rng = np.random.default_rng(0)
    triangles = CUBE[FACES]
    samples, _ = ev.sample_surface(CUBE, FACES, .004)
    for point in rng.uniform(-.4, .4, (20, 3)):
        brute = np.linalg.norm(samples-point, axis=1).min()
        assert abs(ev.triangle_distance(point, triangles)-brute) < .004
    assert ev.triangle_distance(np.array([.3, 0, 0]), triangles) == pytest.approx(.2)
    assert ev.triangle_distance(np.array([.2, .2, .2]), triangles) == pytest.approx(.1*np.sqrt(3))


def test_masks_follow_depth_and_measure_occlusion():
    frame = scene(a=(0, 0, 1.))
    masks = ev.object_masks(frame, {'cube': cube_mesh()}, render(frame['objects']))
    side = .2/.9*F
    assert masks['stats']['a']['visible_px'] == pytest.approx(side**2, rel=.06)
    assert masks['stats']['a']['visible_fraction'] > .97 and masks['graspable'].sum() == masks['stats']['a']['visible_px']
    hidden = scene(a=(0, 0, 1.), b=(.1, 0, .6))  # b stands in front of the right half of a
    masks = ev.object_masks(hidden, {'cube': cube_mesh()}, render(hidden['objects']))
    assert .35 < masks['stats']['a']['visible_fraction'] < .6
    assert ev.check_frame('occlusion', {'graspable_visible_px': 500, 'visible_fraction': masks['stats']['a']['visible_fraction']}) == []
    assert ev.check_frame('distance', {'graspable_visible_px': 500, 'visible_fraction': .5}) == ['target is hidden']


def test_scoring_separates_the_error_classes():
    frame = scene(a=(0, 0, 1.), b=(.5, 0, 1.))
    meshes, geometry = {'cube': cube_mesh()}, render(frame['objects'])
    masks = ev.object_masks(frame, meshes, geometry)
    cx, cy, half = (W-1)/2, (H-1)/2, .1/.9*F
    score = lambda x, y: ev.score_point(normalized(x, y), frame, masks, geometry, meshes)
    centre = score(cx, cy)
    assert centre['outcome'] == 'correct' and centre['on_target'] and centre['surface_m'] < .002
    edge = score(cx-half-4, cy)  # four pixels off the object: depth lands on the wall a metre behind
    assert edge['outcome'] == 'near_miss' and not edge['on_target'] and edge['surface_m'] > .5 and edge['px_to_target'] <= ev.NEAR_PX
    assert score(40, 40)['outcome'] == 'background'
    other = score(cx+.5/.9*F, cy)
    assert other['outcome'] == 'wrong_object' and other['on_object'] == 'b'
    assert ev.score_point(None, frame, masks, geometry, meshes)['outcome'] == 'invalid'
    # Only the left half of the front face is graspable: the right half is the right object but the wrong part.
    left = cube_mesh(CUBE[FACES].mean(1)[:, 0] < -.03)
    masks = ev.object_masks(frame, {'cube': left}, geometry)
    part = ev.score_point(normalized(cx+half-3, cy), frame, masks, geometry, {'cube': left})
    assert part['outcome'] == 'wrong_part' and part['on_target'] and part['object_m'] < .002 < ev.SURFACE_M < part['surface_m']


def test_manifest_and_summary(tmp_path):
    frame = scene(a=(0, 0, 1.))
    geometry = render(frame['objects'])
    for folder in ('inputs', 'truth/meshes'):
        (tmp_path/folder).mkdir(parents=True)
    Image.new('RGB', (W, H)).save(tmp_path/'inputs/f.png')
    np.savez_compressed(tmp_path/'inputs/f.npz', depth=geometry['depth'], K=geometry['K'], camera_position=np.zeros(3),
                        camera_rotation=np.eye(3), depth_kind='distance_to_image_plane', camera_convention='ros_optical')
    np.savez_compressed(tmp_path/'truth/meshes/cube.npz', **cube_mesh())
    (tmp_path/'truth/frames-s.json').write_text(json.dumps([frame]))
    manifest = ev.build_manifest(tmp_path, {'git_commit': 'x'})
    assert manifest['violations'] == [] and manifest['coverage'] == {'distance': {'s': 1}}
    assert manifest['frames'][0]['distance_bin'] == 'mid' and len(manifest['frames'][0]['sha256']) == 3
    assert json.loads((tmp_path/'inputs/frames.jsonl').read_text()) == {'id': 'f', 'image': 'f.png', 'instruction': 'Pick up the cube.'}
    frames, meshes = ev.load_dataset(tmp_path)
    masks, loaded = ev.load_masks(tmp_path, 'f'), ev.load_geometry(tmp_path, 'f')
    hit = ev.score_point([500, 500], frames[0], masks, loaded, meshes)
    miss = ev.score_point([10, 10], frames[0], masks, loaded, meshes)
    rows = [{'category': 'distance', 'contact': hit, 'pregrasp': miss, 'latency_s': t, 'generations': 1,
             'input_tokens': 100, 'output_tokens': 20, 'cost_usd': None} for t in (1., 2., 3., 10.)]
    summary = ev.summarize(rows)
    assert summary['contact']['hit_3d_rate'] == 1 and summary['pregrasp']['outcomes']['background'] == 4
    assert summary['both_hit_3d_rate'] == 0 and summary['pregrasp']['edge_or_background'] == 4
    edge = ev.score_point([(319.5-.1/.9*F-4)/(W-1)*1000, 500], frames[0], masks, loaded, meshes)
    jumped = ev.summarize([dict(rows[0], contact=edge)])['contact']
    assert jumped['outcomes']['near_miss'] == 1 and jumped['near_miss_depth_jump'] == 1  # landed on the wall a metre behind
    assert summary['latency_s'] == {'median': 2.5, 'p90': pytest.approx(7.9), 'max': 10.}
    assert summary['input_tokens'] == {'mean': 100., 'total': 400.} and summary['cost_usd'] is None


def test_hosted_backends_build_requests_and_keep_no_key(monkeypatch):
    image, sent = Image.new('RGB', (8, 8)), {}

    def post(url, headers, body, timeout):
        sent.update(url=url, headers=headers, body=body)
        if 'chat/completions' in url:
            return 200, {'model': 'm-2026-01-01', 'choices': [{'message': {'content': '{"point": [1, 2]}'}}],
                         'usage': {'prompt_tokens': 1000, 'completion_tokens': 500}}
        return 200, {'modelVersion': 'g-001', 'candidates': [{'content': {'parts': [{'text': 'hidden', 'thought': True}, {'text': 'ok'}]}}],
                     'usageMetadata': {'promptTokenCount': 300, 'candidatesTokenCount': 10, 'thoughtsTokenCount': 90}}

    config = {'model': 'm', 'base_url': 'https://example.invalid/v1', 'request': {'reasoning_effort': 'low'},
              'price_per_million_tokens_usd': {'input': 2., 'output': 8.}}
    monkeypatch.setenv('OPENAI_API_KEY', 'secret-o')
    monkeypatch.setenv('GEMINI_API_KEY', 'secret-g')
    record = backends.OpenAIBackend(config, post).generate(image, 'Where?', 64)
    assert sent['url'] == 'https://example.invalid/v1/chat/completions' and sent['headers'] == {'Authorization': 'Bearer secret-o'}
    assert sent['body']['model'] == 'm' and sent['body']['reasoning_effort'] == 'low' and 'max_tokens' not in sent['body']
    assert sent['body']['messages'][0]['content'][0]['image_url']['url'].startswith('data:image/png;base64,')
    assert record['text'] == '{"point": [1, 2]}' and record['model_version'] == 'm-2026-01-01'
    assert record['cost_usd'] == pytest.approx((1000*2+500*8)/1e6) and 'secret' not in json.dumps(record)
    record = backends.GeminiBackend({**config, 'price_per_million_tokens_usd': None}, post).generate(image, 'Where?', 64)
    assert sent['url'] == 'https://example.invalid/v1/models/m:generateContent' and sent['headers'] == {'x-goog-api-key': 'secret-g'}
    assert record['text'] == 'ok' and record['output_tokens'] == 100 and record['cost_usd'] is None and 'secret' not in json.dumps(record)
    failed = backends.OpenAIBackend(config, lambda *a: (429, {'error_body': 'slow down'})).generate(image, 'Where?', 64)
    assert failed['text'] == '' and failed['http_status'] == 429 and pointing.parse_answer(failed['text'])['contact_point'] is None


def test_hosted_backends_refuse_to_guess_model_or_run_without_key(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    released = json.loads((Path(__file__).resolve().parents[1]/'configs/pointing-backends.json').read_text())
    for name in ('openai', 'gemini'):
        assert released[name]['request'] == {}  # provider defaults unless the user asks otherwise
        with pytest.raises(SystemExit, match='exact model version'):
            backends.API_BACKENDS[name]({**released[name], 'model': None})
    with pytest.raises(SystemExit, match='OPENAI_API_KEY is not set'):
        backends.OpenAIBackend({'model': 'm', 'base_url': 'https://example.invalid'})


def test_frame_specification_covers_the_plan():
    spec = json.loads((Path(__file__).resolve().parents[1]/'configs/pointing-frames.json').read_text())
    frames = spec['frames']
    assert len(frames) == 80 and len({f['id'] for f in frames}) == 80
    for category in ('small', 'occlusion', 'distance', 'distractor'):
        scenes = [f['scene'] for f in frames if f['category'] == category]
        assert len(scenes) == 20 and {'pp_box', 'doubledesk'} <= set(scenes)
    for frame in frames:
        instances = spec['scenes'][frame['scene']]['instances']
        assert frame['target'] in frame['objects'] and set(frame['objects']) <= set(instances)
        assert all(len(pose) == 4 for pose in frame['objects'].values()) and len(frame['robot']) == 3
        assert instances[frame['target']] in spec['assets'] and frame['instruction'].strip()
