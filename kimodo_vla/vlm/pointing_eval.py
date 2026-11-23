"""Truth geometry and scoring for the pointing test (experiment 3).

Evaluator only: it reads ``truth/`` and nothing here builds a prompt. Object masks are derived from
the recorded meshes and poses and checked against the depth image, so a wrong pose or calibration
shows up as a low visible fraction instead of silently shifting the scores.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SURFACE_M = .03     # a point counts in 3D when it back-projects this close to the graspable surface
ON_OBJECT_M = .008  # a depth pixel belongs to an object when its 3D point is this close to the mesh
NEAR_PX = 10        # "only slightly off": this close to the target in the image
DEPTH_JUMP_M = .10  # a near miss whose depth landed this far from the object hit the background, not the support beside it
MAX_SAMPLES = 1_500_000
MIN_GRASPABLE_PX = 12
RULES = {'all': 'at least 12 px of graspable surface visible', 'occlusion': 'visible_fraction in [0.2, 0.85]',
         'distractor': 'at least 2 other objects in view; visible_fraction >= 0.8',
         'small': 'target covers at most 2000 px; visible_fraction >= 0.8',
         'distance': 'visible_fraction >= 0.8; near (<0.8 m), mid, far (>=1.6 m) recorded per frame'}


def rotation(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def world_vertices(mesh, obj):
    return mesh['vertices'].astype(float) @ rotation(obj['quat_wxyz']).T + np.asarray(obj['position'])


def sample_surface(vertices, faces, spacing, seed=0):
    """Area-weighted surface samples about ``spacing`` apart, with the triangle each one came from."""
    tri = vertices[faces]
    area = .5*np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    n = int(np.clip(area.sum()/spacing**2, 2000, MAX_SAMPLES))
    rng = np.random.default_rng(seed)
    index = rng.choice(len(faces), n, p=area/area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u+v > 1
    u[flip], v[flip] = 1-u[flip], 1-v[flip]
    t = tri[index]
    return t[:, 0]+u[:, None]*(t[:, 1]-t[:, 0])+v[:, None]*(t[:, 2]-t[:, 0]), index


def to_pixels(points, geometry):
    """World points -> (pixel coordinates, optical depth)."""
    optical = (points-geometry['camera_position']) @ geometry['camera_rotation']
    z = optical[:, 2]
    uv = optical @ geometry['K'].T
    with np.errstate(divide='ignore', invalid='ignore'):
        return uv[:, :2]/z[:, None], z


def pixel_points(geometry):
    """World point of every depth pixel (NaN where depth is missing)."""
    depth = geometry['depth']
    h, w = depth.shape
    grid = np.stack([*np.meshgrid(np.arange(w), np.arange(h)), np.ones((h, w))], -1).reshape(-1, 3)
    z = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan).reshape(-1, 1)
    return (z*np.linalg.solve(geometry['K'], grid.T).T) @ geometry['camera_rotation'].T + geometry['camera_position']


def backproject(pixel, geometry):
    x, y = pixel
    z = geometry['depth'][y, x]
    if not np.isfinite(z) or z <= 0:
        return None
    return geometry['camera_rotation'] @ (z*np.linalg.solve(geometry['K'], [x, y, 1.])) + geometry['camera_position']


def _shift_any(mask):
    out = mask.copy()
    out[1:] |= mask[:-1]; out[:-1] |= mask[1:]; out[:, 1:] |= mask[:, :-1]; out[:, :-1] |= mask[:, 1:]
    return out


def _close(mask):
    return ~_shift_any(~_shift_any(mask)) | mask


def object_masks(frame, meshes, geometry):
    """Per-pixel object identity and each object's visibility, from meshes, poses and the depth image."""
    h, w = geometry['depth'].shape
    points = pixel_points(geometry)
    best = np.full(h*w, np.inf)
    ids = np.full(h*w, -1)
    names, stats, graspable = list(frame['objects']), {}, np.zeros(h*w, bool)
    for i, name in enumerate(names):
        obj = frame['objects'][name]
        mesh = meshes[obj['asset']]
        vertices = world_vertices(mesh, obj)
        _, depth = to_pixels(vertices, geometry)
        near = max(.15, float(depth[depth > .05].min())) if (depth > .05).any() else .15
        samples, face = sample_surface(vertices, mesh['faces'], min(.002, .3*near/geometry['K'][0, 0]))
        uv, z = to_pixels(samples, geometry)
        pixel = np.rint(uv)
        inside = (z > .05) & (pixel[:, 0] >= 0) & (pixel[:, 0] < w) & (pixel[:, 1] >= 0) & (pixel[:, 1] < h)
        silhouette = np.zeros(h*w, bool)
        silhouette[(pixel[inside, 1]*w+pixel[inside, 0]).astype(int)] = True
        silhouette = _close(silhouette.reshape(h, w)).reshape(-1)
        candidates = np.flatnonzero(silhouette & np.isfinite(points[:, 0]))
        distance = np.full(h*w, np.inf)
        if len(candidates):
            distance[candidates] = cKDTree(samples).query(points[candidates], distance_upper_bound=ON_OBJECT_M)[0]
        stats[name] = {'silhouette_px': int(silhouette.sum()), 'in_frame_fraction': float(inside.mean()),
                       'distance_m': float(np.linalg.norm(vertices.mean(0)-geometry['camera_position']))}
        closer = distance < best
        ids[closer], best[closer] = i, distance[closer]
        if name == frame['target']:
            grasp = np.flatnonzero(np.isfinite(distance))
            keep = mesh['graspable'][face]
            if len(grasp) and keep.any():
                graspable[grasp] = np.isfinite(cKDTree(samples[keep]).query(points[grasp], distance_upper_bound=ON_OBJECT_M)[0])
    for i, name in enumerate(names):
        visible = int((ids == i).sum())
        stats[name].update(visible_px=visible, visible_fraction=visible/max(stats[name]['silhouette_px'], 1))
    graspable &= ids == names.index(frame['target'])
    return {'ids': ids.reshape(h, w), 'names': names, 'stats': stats, 'graspable': graspable.reshape(h, w)}


def triangle_distance(point, triangles):
    """Exact distance from one point to a triangle soup (F,3,3)."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b-a, c-a
    ap, bp, cp = point-a, point-b, point-c
    d1, d2 = (ab*ap).sum(1), (ac*ap).sum(1)
    d3, d4 = (ab*bp).sum(1), (ac*bp).sum(1)
    d5, d6 = (ab*cp).sum(1), (ac*cp).sum(1)
    va, vb, vc = d3*d6-d5*d4, d5*d2-d1*d6, d1*d4-d3*d2
    with np.errstate(divide='ignore', invalid='ignore'):
        inner = a+ab*(vb/(va+vb+vc))[:, None]+ac*(vc/(va+vb+vc))[:, None]
        closest = np.select(
            [((d1 <= 0) & (d2 <= 0))[:, None], ((d3 >= 0) & (d4 <= d3))[:, None],
             ((vc <= 0) & (d1 >= 0) & (d3 <= 0))[:, None], ((d6 >= 0) & (d5 <= d6))[:, None],
             ((vb <= 0) & (d2 >= 0) & (d6 <= 0))[:, None], ((va <= 0) & (d4-d3 >= 0) & (d5-d6 >= 0))[:, None]],
            [a, b, a+ab*(d1/(d1-d3))[:, None], c, a+ac*(d2/(d2-d6))[:, None],
             b+(c-b)*((d4-d3)/((d4-d3)+(d5-d6)))[:, None]], inner)
    distance = np.linalg.norm(point-closest, axis=1)
    # Degenerate triangles have no interior; their corners are still part of the surface.
    distance = np.where(np.isfinite(distance), distance, np.linalg.norm(ap, axis=1))
    return float(distance.min())


def to_pixel(point, shape):
    """Normalized [0,1000] over the full image -> nearest pixel, the convention of kimodo_vla.vlm.geometry."""
    h, w = shape
    return tuple(int(v) for v in np.rint(np.asarray(point, float)/1000*np.array([w-1, h-1])))


def score_point(point, frame, masks, geometry, meshes):
    """Score one predicted point; ``point`` is normalized [0,1000] or None when the model gave no usable answer."""
    if point is None:
        return {'outcome': 'invalid', 'on_target': False, 'hit_3d': False}
    x, y = pixel = to_pixel(point, masks['ids'].shape)
    names, target = masks['names'], frame['target']
    here = masks['ids'][y, x]
    ys, xs = np.nonzero(masks['ids'] >= 0)
    gap = {}
    for i, name in enumerate(names):
        chosen = masks['ids'][ys, xs] == i
        if chosen.any():
            gap[name] = float(np.hypot(xs[chosen]-x, ys[chosen]-y).min())
    result = {'pixel': list(pixel), 'on_object': names[here] if here >= 0 else None, 'on_target': bool(here >= 0 and names[here] == target),
              'px_to_target': gap.get(target), 'depth_m': None, 'surface_m': None, 'object_m': None, 'hit_3d': False}
    world = backproject(pixel, geometry)
    if world is not None:
        mesh, obj = meshes[frame['objects'][target]['asset']], frame['objects'][target]
        triangles = world_vertices(mesh, obj)[mesh['faces']]
        result.update(depth_m=float(geometry['depth'][y, x]), object_m=triangle_distance(world, triangles),
                      surface_m=triangle_distance(world, triangles[mesh['graspable']]))
        result['hit_3d'] = result['surface_m'] <= SURFACE_M
    others = {name: d for name, d in gap.items() if name != target and d <= NEAR_PX}
    if result['hit_3d']:
        outcome = 'correct'
    elif others and min(others.values()) < gap.get(target, np.inf):
        outcome = 'wrong_object'     # on or next to a different object: recognition or selection error
    elif result['on_target']:
        outcome = 'wrong_part'       # on the right object but not on its graspable part
    elif gap.get(target, np.inf) <= NEAR_PX:
        outcome = 'near_miss'        # pixel only slightly off the object, depth landed on what is behind it
    else:
        outcome = 'background'
    return {**result, 'outcome': outcome}


def load_dataset(dataset):
    dataset = Path(dataset)
    frames = json.loads((dataset/'truth/frames.json').read_text())
    meshes = {}
    for path in sorted((dataset/'truth/meshes').glob('*.npz')):
        with np.load(path) as f:
            meshes[path.stem] = {k: f[k] for k in f.files}
    return frames, meshes


def load_geometry(dataset, frame_id):
    with np.load(Path(dataset)/'inputs'/f'{frame_id}.npz', allow_pickle=False) as f:
        if str(f['depth_kind']) != 'distance_to_image_plane' or str(f['camera_convention']) != 'ros_optical':
            raise ValueError('Expected metric optical Z and ROS optical coordinates')
        return {k: f[k].astype(float) for k in ('depth', 'K', 'camera_position', 'camera_rotation')}


def load_masks(dataset, frame_id):
    with np.load(Path(dataset)/'truth/masks'/f'{frame_id}.npz', allow_pickle=False) as f:
        return {'ids': f['ids'], 'graspable': f['graspable'], 'names': [str(n) for n in f['names']]}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def check_frame(category, entry):
    """Does the measured frame match the coverage class it was specified for?"""
    problems = []
    if entry['graspable_visible_px'] < MIN_GRASPABLE_PX:
        problems.append('graspable surface not visible')
    if category == 'occlusion' and not .2 <= entry['visible_fraction'] <= .85:
        problems.append('not partly hidden')
    if category == 'distractor' and entry['other_objects_in_view'] < 2:
        problems.append('fewer than 2 other objects in view')
    if category == 'small' and entry['visible_px'] > 2000:
        problems.append('target is not small in the image')
    if category in ('distance', 'small', 'distractor') and entry['visible_fraction'] < .8:
        problems.append('target is hidden')
    return problems


def build_manifest(dataset, meta):
    """Merge per-scene truth, derive masks and coverage measurements, hash every file."""
    dataset = Path(dataset)
    frames = [f for path in sorted((dataset/'truth').glob('frames-*.json')) for f in json.loads(path.read_text())]
    (dataset/'truth/frames.json').write_text(json.dumps(frames, indent=1))
    (dataset/'truth/masks').mkdir(exist_ok=True)
    with (dataset/'inputs/frames.jsonl').open('w') as f:
        for frame in frames:
            f.write(json.dumps({'id': frame['id'], 'image': frame['id']+'.png', 'instruction': frame['instruction']})+'\n')
    _, meshes = load_dataset(dataset)
    entries, coverage, violations = [], {}, []
    for frame in frames:
        masks = object_masks(frame, meshes, load_geometry(dataset, frame['id']))
        np.savez_compressed(dataset/'truth/masks'/f"{frame['id']}.npz", ids=masks['ids'].astype(np.int8),
                            graspable=masks['graspable'], names=np.array(masks['names']))
        target = masks['stats'][frame['target']]
        entry = {'id': frame['id'], 'scene': frame['scene'], 'category': frame['category'],
                 'target_asset': frame['objects'][frame['target']]['asset'], 'distance_m': round(target['distance_m'], 3),
                 'distance_bin': 'near' if target['distance_m'] < .8 else 'mid' if target['distance_m'] < 1.6 else 'far',
                 'visible_px': target['visible_px'], 'visible_fraction': round(target['visible_fraction'], 3),
                 'in_frame_fraction': round(target['in_frame_fraction'], 3), 'graspable_visible_px': int(masks['graspable'].sum()),
                 'other_objects_in_view': sum(s['visible_px'] >= 30 for n, s in masks['stats'].items() if n != frame['target']),
                 'sha256': {name: sha(dataset/name) for name in (f"inputs/{frame['id']}.png", f"inputs/{frame['id']}.npz",
                                                                 f"truth/masks/{frame['id']}.npz")}}
        entries.append(entry)
        coverage.setdefault(frame['category'], {}).setdefault(frame['scene'], 0)
        coverage[frame['category']][frame['scene']] += 1
        violations += [f"{frame['id']}: {problem}" for problem in check_frame(frame['category'], entry)]
    manifest = {**meta, 'frames': entries, 'coverage': coverage, 'coverage_rules': RULES, 'violations': violations,
                'thresholds': {'surface_m': SURFACE_M, 'on_object_m': ON_OBJECT_M, 'near_px': NEAR_PX, 'depth_jump_m': DEPTH_JUMP_M},
                'sha256': {str(p.relative_to(dataset)): sha(p) for p in [dataset/'truth/frames.json', dataset/'inputs/frames.jsonl',
                                                                         *sorted((dataset/'truth/meshes').glob('*.npz'))]}}
    (dataset/'manifest.json').write_text(json.dumps(manifest, indent=1))
    return manifest


def percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def summarize(rows):
    """Aggregate scored frames: rows are dicts with category, contact, pregrasp, latency_s and usage fields."""
    def block(subset):
        out = {'frames': len(subset)}
        for key in ('contact', 'pregrasp'):
            scores = [r[key] for r in subset]
            outcomes = {name: sum(s['outcome'] == name for s in scores)
                        for name in ('correct', 'near_miss', 'background', 'wrong_part', 'wrong_object', 'invalid')}
            out[key] = {'on_target_rate': float(np.mean([s['on_target'] for s in scores])) if scores else None,
                        'hit_3d_rate': float(np.mean([s['hit_3d'] for s in scores])) if scores else None, 'outcomes': outcomes,
                        # Near misses whose back-projection jumped to the background behind the object.
                        'near_miss_depth_jump': sum(s['outcome'] == 'near_miss' and (s['object_m'] is None or s['object_m'] > DEPTH_JUMP_M)
                                                    for s in scores),
                        # The two error groups of the plan.
                        'edge_or_background': outcomes['near_miss']+outcomes['background']+outcomes['wrong_part'],
                        'wrong_object_or_invalid': outcomes['wrong_object']+outcomes['invalid']}
        out['both_hit_3d_rate'] = float(np.mean([r['contact']['hit_3d'] and r['pregrasp']['hit_3d'] for r in subset])) if subset else None
        return out
    latency = [r['latency_s'] for r in rows]
    summary = {**block(rows), 'by_category': {c: block([r for r in rows if r['category'] == c]) for c in sorted({r['category'] for r in rows})},
               'latency_s': {'median': percentile(latency, 50), 'p90': percentile(latency, 90), 'max': max(latency, default=None)},
               'generations_per_frame': float(np.mean([r['generations'] for r in rows])) if rows else None}
    for key in ('input_tokens', 'output_tokens', 'cost_usd'):
        values = [r[key] for r in rows if r.get(key) is not None]
        summary[key] = {'mean': float(np.mean(values)), 'total': float(np.sum(values))} if values else None
    return summary
