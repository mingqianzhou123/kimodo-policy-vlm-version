"""The body command set the VLM must emit, and the validation of one command.

Frames. ``walk`` and ``turn`` are relative to the robot as it stands when the command starts:
``forward_m`` along its facing direction, ``left_m`` to its left, ``left_deg`` counter-clockwise
seen from above. Every other position is in the world frame of the observation: x, y on the
floor, z up, metres, floor at z = 0. Pelvis height is the height of the pelvis origin above the
floor; torso pitch is the forward tilt of the torso from upright, in degrees.

One command per reply, as one JSON object:

  {"type": "walk", "forward_m": 0.4, "left_m": -0.1}
  {"type": "turn", "left_deg": 90}
  {"type": "pelvis_height", "height_m": 0.62}
  {"type": "pelvis_height", "height_m": 0.55, "behind_feet_m": 0.15}
  {"type": "torso_pitch", "pitch_deg": 20}
  {"type": "hand_pose", "hands": {"left": {"position": [x, y, z], "palm": [px, py, pz], "fingers": [fx, fy, fz]},
                                  "right": {...}}, "duration_s": 1.5}
  {"type": "hand", "left": "close", "right": "close"}
  {"type": "done"}

``pelvis_height`` keeps the feet where they are and the body balanced over them; with
``behind_feet_m`` the pelvis instead ends that far behind the midpoint of the ankles, which is how
the robot sits down (the seat then carries it).
``hand_pose`` moves one or both wrists (a hand not named keeps its pose): ``position`` is the
wrist, ``palm`` the direction the palm faces and ``fingers`` the direction the straight fingers
point, both in the world frame; they need not be exactly perpendicular (the palm direction is
kept, the finger direction is made perpendicular to it). ``hand`` opens or closes either hand;
the hands of this robot only open and close. ``done`` ends the episode.

Validation does not look at the scene: it checks the shape, the types, finiteness and the ranges
in the limits (one config, ``DEFAULT_LIMITS`` as defaults). A command that fails it is not
executed; the loop records why and asks again (see ``loop``).
"""
import json
import math
import re

import numpy as np

TYPES = ('walk', 'turn', 'pelvis_height', 'torso_pitch', 'hand_pose', 'hand', 'done')
SIDES = ('left', 'right')

# Ranges a command may ask for. The experiment's config may narrow or widen them; these values are
# what the planner was built for, not tuned on outcomes.
DEFAULT_LIMITS = {
    'walk_max_m': 2.0,            # longest single walk command (either component and the norm)
    'walk_min_m': 0.02,           # shorter walks are refused (the walker cannot place a foot that finely)
    'turn_max_deg': 180.,
    'turn_min_deg': 2.,
    'pelvis_height_m': [0.40, 0.80],
    'behind_feet_m': [0., 0.30],
    'torso_pitch_deg': [0., 45.],
    'hand_duration_s': [0.5, 4.0],
    'hand_default_duration_s': 1.5,
    'workspace_radius_m': 1.2,    # a wrist target farther than this from the pelvis is refused as malformed
    'min_palm_finger_angle_deg': 45.,
}


class CommandError(ValueError):
    """The reply is not one valid command; ``code`` names the rule it broke."""
    def __init__(self, code, message):
        super().__init__(f'{code}: {message}')
        self.code = code


def extract_json(text):
    """The first JSON object in a model reply (a fenced block or bare braces)."""
    if not isinstance(text, str) or not text.strip():
        raise CommandError('empty', 'the reply is empty')
    fenced = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, flags=re.S)
    candidates = fenced or [text[i:] for i, c in enumerate(text) if c == '{']
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value, _ = decoder.raw_decode(candidate.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise CommandError('no_json', 'no JSON object found in the reply')


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CommandError('bad_number', f'{name} must be a finite number')
    return float(value)


def _vector(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise CommandError('bad_vector', f'{name} must be a list of three numbers')
    return [_number(v, f'{name}[{i}]') for i, v in enumerate(value)]


def _in(value, bounds, name, unit):
    low, high = bounds
    if not low <= value <= high:
        raise CommandError('out_of_range', f'{name} {value:g} {unit} is outside [{low:g}, {high:g}]')


def _keys(command, allowed):
    unknown = set(command) - set(allowed) - {'type', 'note'}
    if unknown:
        raise CommandError('unknown_field', f'unexpected fields {sorted(unknown)} for {command["type"]}')


def wrist_rotation(side, palm, fingers):
    """World rotation [3, 3] of a wrist link (``*_wrist_yaw_link``) from palm and finger directions.

    The link's x axis runs along the forearm to the fingers. The left palm faces the link's -y,
    the right palm its +y (``grasp_policy.waypoints``), so two level wrists with palms facing
    each other act as parallel jaws.
    """
    palm = np.asarray(palm, dtype=float)
    fingers = np.asarray(fingers, dtype=float)
    palm = palm/np.linalg.norm(palm)
    x = fingers-(fingers @ palm)*palm
    x /= np.linalg.norm(x)
    y = -palm if side == 'left' else palm
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def palm_and_fingers(side, rotation):
    """Inverse of ``wrist_rotation``: (palm, fingers) directions of a wrist rotation [3, 3]."""
    rotation = np.asarray(rotation, dtype=float)
    y = rotation[:, 1]
    return (-y if side == 'left' else y), rotation[:, 0]


def validate(command, limits=None, pelvis=None):
    """Normalized copy of one command, or ``CommandError``.

    ``pelvis`` (world [3], optional) lets a wrist target be checked against the workspace radius.
    """
    lim = {**DEFAULT_LIMITS, **(limits or {})}
    if not isinstance(command, dict):
        raise CommandError('not_object', 'a command is one JSON object')
    kind = command.get('type')
    if kind not in TYPES:
        raise CommandError('unknown_type', f'type must be one of {list(TYPES)}, got {kind!r}')
    out = {'type': kind}
    if kind == 'walk':
        _keys(command, ('forward_m', 'left_m'))
        forward = _number(command.get('forward_m', 0.), 'forward_m')
        left = _number(command.get('left_m', 0.), 'left_m')
        norm = math.hypot(forward, left)
        if norm > lim['walk_max_m']:
            raise CommandError('out_of_range', f'walk of {norm:.2f} m exceeds {lim["walk_max_m"]:g} m')
        if norm < lim['walk_min_m']:
            raise CommandError('out_of_range', f'walk of {norm:.3f} m is shorter than {lim["walk_min_m"]:g} m')
        out.update(forward_m=forward, left_m=left)
    elif kind == 'turn':
        _keys(command, ('left_deg',))
        angle = _number(command.get('left_deg'), 'left_deg')
        if not lim['turn_min_deg'] <= abs(angle) <= lim['turn_max_deg']:
            raise CommandError('out_of_range', f'turn of {angle:g} deg outside [{lim["turn_min_deg"]:g}, {lim["turn_max_deg"]:g}] in magnitude')
        out['left_deg'] = angle
    elif kind == 'pelvis_height':
        _keys(command, ('height_m', 'behind_feet_m'))
        height = _number(command.get('height_m'), 'height_m')
        _in(height, lim['pelvis_height_m'], 'height_m', 'm')
        out['height_m'] = height
        if 'behind_feet_m' in command:
            behind = _number(command['behind_feet_m'], 'behind_feet_m')
            _in(behind, lim['behind_feet_m'], 'behind_feet_m', 'm')
            out['behind_feet_m'] = behind
    elif kind == 'torso_pitch':
        _keys(command, ('pitch_deg',))
        pitch = _number(command.get('pitch_deg'), 'pitch_deg')
        _in(pitch, lim['torso_pitch_deg'], 'pitch_deg', 'deg')
        out['pitch_deg'] = pitch
    elif kind == 'hand_pose':
        _keys(command, ('hands', 'duration_s'))
        hands = command.get('hands')
        if not isinstance(hands, dict) or not hands or set(hands) - set(SIDES):
            raise CommandError('bad_hands', 'hands must name "left" and/or "right"')
        duration = _number(command.get('duration_s', lim['hand_default_duration_s']), 'duration_s')
        _in(duration, lim['hand_duration_s'], 'duration_s', 's')
        out['hands'], out['duration_s'] = {}, duration
        for side in SIDES:
            if side not in hands:
                continue
            target = hands[side]
            if not isinstance(target, dict) or set(target) - {'position', 'palm', 'fingers'} or 'position' not in target:
                raise CommandError('bad_hands', f'{side} needs position and may have palm and fingers')
            position = _vector(target['position'], f'{side}.position')
            if pelvis is not None and np.linalg.norm(np.subtract(position, pelvis)) > lim['workspace_radius_m']:
                raise CommandError('out_of_range', f'{side} wrist target is more than {lim["workspace_radius_m"]:g} m from the pelvis')
            entry = {'position': position}
            if ('palm' in target) != ('fingers' in target):
                raise CommandError('bad_hands', f'{side}: give palm and fingers together, or neither (keep the orientation)')
            if 'palm' in target:
                palm, fingers = _vector(target['palm'], f'{side}.palm'), _vector(target['fingers'], f'{side}.fingers')
                if min(np.linalg.norm(palm), np.linalg.norm(fingers)) < 1e-6:
                    raise CommandError('bad_vector', f'{side}: palm and fingers must be non-zero directions')
                cos = np.dot(palm, fingers)/np.linalg.norm(palm)/np.linalg.norm(fingers)
                if np.degrees(np.arccos(np.clip(abs(cos), 0., 1.))) < lim['min_palm_finger_angle_deg']:
                    raise CommandError('bad_vector', f'{side}: palm and fingers are nearly parallel')
                n = lambda v: (np.asarray(v)/np.linalg.norm(v)).tolist()
                entry.update(palm=n(palm), fingers=n(fingers))
            out['hands'][side] = entry
    elif kind == 'hand':
        _keys(command, SIDES)
        named = {s: command[s] for s in SIDES if s in command}
        if not named or any(v not in ('open', 'close') for v in named.values()):
            raise CommandError('bad_hands', 'hand needs "left" and/or "right" set to "open" or "close"')
        out.update(named)
    elif kind == 'done':
        _keys(command, ('reason',))
        if 'reason' in command:
            out['reason'] = str(command['reason'])[:200]
    return out


def parse(text, limits=None, pelvis=None):
    """One validated command from a model reply."""
    return validate(extract_json(text), limits, pelvis)


SCHEMA = {
    'oneOf': [
        {'type': 'object', 'properties': {'type': {'const': 'walk'}, 'forward_m': {'type': 'number'}, 'left_m': {'type': 'number'}},
         'required': ['type']},
        {'type': 'object', 'properties': {'type': {'const': 'turn'}, 'left_deg': {'type': 'number'}}, 'required': ['type', 'left_deg']},
        {'type': 'object', 'properties': {'type': {'const': 'pelvis_height'}, 'height_m': {'type': 'number'},
                                          'behind_feet_m': {'type': 'number'}},
         'required': ['type', 'height_m']},
        {'type': 'object', 'properties': {'type': {'const': 'torso_pitch'}, 'pitch_deg': {'type': 'number'}},
         'required': ['type', 'pitch_deg']},
        {'type': 'object', 'properties': {'type': {'const': 'hand_pose'}, 'duration_s': {'type': 'number'},
            'hands': {'type': 'object', 'additionalProperties': False, 'properties': {s: {
                'type': 'object', 'required': ['position'], 'properties': {
                    'position': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3},
                    'palm': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3},
                    'fingers': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}}} for s in SIDES}}},
         'required': ['type', 'hands']},
        {'type': 'object', 'properties': {'type': {'const': 'hand'}, 'left': {'enum': ['open', 'close']},
                                          'right': {'enum': ['open', 'close']}}, 'required': ['type']},
        {'type': 'object', 'properties': {'type': {'const': 'done'}, 'reason': {'type': 'string'}}, 'required': ['type']},
    ]}
