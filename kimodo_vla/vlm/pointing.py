"""One prompt and one answer format for pointing at natural objects, shared by every backend.

The model sees an RGB image and a task sentence. It never sees depth, poses or anything from ``truth/``.
"""
import json
import time

import numpy as np
from PIL import Image

from kimodo_vla.vlm.refinement import CROP_SIZES, crop_box, crop_to_full

PROMPT = (
    'You control a humanoid robot and see the scene through its head camera. Task: {instruction}\n'
    'Point to where the robot hand should grasp the object named in the task.\n'
    '- contact_point: the point on the object where the palm and fingers should close.\n'
    '- pregrasp_point: a second point on the same object, on the side the hand arrives from. '
    'The line from pregrasp_point to contact_point shows how the hand is oriented.\n'
    'Both points must lie on the visible surface of that object, not on the table, the background or another object.\n'
    'Return ONLY JSON: {{"contact_point": [x, y], "pregrasp_point": [x, y], "action": "<one sentence describing the motion>"}}\n'
    'x and y are numbers from 0 to 1000 over the full image: x from the left edge, y from the top edge.')
CROP_PROMPT = (
    'This is a zoomed crop from the head camera of a humanoid robot. Task: {instruction}\n'
    'Point to {what}\n'
    'The point must lie on the visible surface of the object named in the task.\n'
    'Return ONLY JSON: {{"point": [x, y]}}, numbers from 0 to 1000 over this crop: x from the left edge, y from the top edge. '
    'If that object is not in the crop, return {{"point": null}}.')
WHAT = {'contact_point': 'the point on the object where the palm and fingers should close.',
        'pregrasp_point': 'a second grasp reference point on the same object, on the side the hand arrives from.'}
POINTS = ('contact_point', 'pregrasp_point')
MAX_TOKENS, CROP_MAX_TOKENS, CROP_PIXELS = 160, 64, 640


def first_object(text):
    """The first JSON object in a reply, and whether the reply was nothing but that object."""
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value, True
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for start in (i for i, c in enumerate(text) if c == '{'):
        try:
            value, _ = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value, False
    return None, False


def valid_point(point):
    return (isinstance(point, list) and len(point) == 2 and all(type(v) in (int, float) for v in point)
            and bool(np.isfinite(point).all()) and all(0 <= v <= 1000 for v in point))


def parse_answer(text):
    """Every usable field of a reply. Missing or malformed fields become None and are scored as invalid."""
    value, strict = first_object(text)
    value = value or {}
    answer = {key: value[key] if valid_point(value.get(key)) else None for key in POINTS}
    action = value.get('action')
    answer['action'] = action.strip() if isinstance(action, str) and action.strip() else None
    answer['format_ok'] = strict and set(value) == {*POINTS, 'action'} and None not in answer.values()
    return answer


def parse_crop(text):
    value, _ = first_object(text)
    point = (value or {}).get('point')
    return point if valid_point(point) else None


def locate_single(backend, image, instruction):
    """One generation on the full image."""
    start = time.perf_counter()
    call = backend.generate(image, PROMPT.format(instruction=instruction), MAX_TOKENS)
    return {'answer': parse_answer(call['text']), 'generations': [dict(call, stage='full')],
            'latency_s': time.perf_counter()-start}


def locate_two_scale(backend, image, instruction):
    """The coarse-to-fine procedure of the marker experiment: one full-image generation, then two crops per point."""
    start = time.perf_counter()
    call = backend.generate(image, PROMPT.format(instruction=instruction), MAX_TOKENS)
    coarse = parse_answer(call['text'])
    answer, generations, refinement = dict(coarse), [dict(call, stage='full')], {}
    scale = np.array([image.width-1, image.height-1])
    for key in POINTS:
        if coarse[key] is None:
            continue
        estimates = []
        for size in CROP_SIZES:
            box = crop_box(coarse[key], image.size, size)
            crop = image.crop(box).resize((CROP_PIXELS, CROP_PIXELS), Image.Resampling.BICUBIC)
            call = backend.generate(crop, CROP_PROMPT.format(instruction=instruction, what=WHAT[key]), CROP_MAX_TOKENS)
            generations.append(dict(call, stage=f'{key}-{size}', box=list(box)))
            point = parse_crop(call['text'])
            if point is not None:
                estimates.append(crop_to_full(point, box))
        refinement[key] = {'estimates_px': [e.tolist() for e in estimates],
                           'disagreement_px': float(np.linalg.norm(estimates[0]-estimates[1])) if len(estimates) == 2 else None}
        # A crop that returns nothing leaves the other estimate, or the coarse point, in place; the evaluator scores whatever results.
        if estimates:
            answer[key] = (np.mean(estimates, axis=0)/scale*1000).tolist()
    return {'answer': answer, 'coarse_answer': coarse, 'refinement': refinement, 'generations': generations,
            'latency_s': time.perf_counter()-start}


LOCALIZERS = {'single': locate_single, 'two-scale': locate_two_scale}
