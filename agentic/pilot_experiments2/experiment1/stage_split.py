"""Experiment 1: split the error of the historical target -> Kimodo -> SONIC grasp episodes into stages.

Reads the run folders of the box (``exp1-*``, ``exp4-*``) and hammer (``exp1b-*``) grasp families
and, for the first attempt (round 0) of every episode, measures each body quantity at four layers:

  T  task target: the round's grasp pose (box: the open-hand ``pre`` pose the gate checks;
     hammer: the right hand's ``grip`` pose), the anchored root, the torso at the anchor
  K  raw Kimodo output: the 29 joints Kimodo generated for this control step, by XML FK
  S  sent reference: the same plan's command after IK projection and root pinning
  R  consumed reference: what SONIC was tracking at this step (``execution.jsonl``)
  A  executed: the measured robot

Stage 1a = T->K, stage 1b = K->S (plus the hand-off S->R, reported apart), stage 1 = T->R,
stage 2 = R->A. Quantities: wrist position (m), wrist orientation (deg), root position
(horizontal, m), torso orientation (deg, root orientation plus the waist joints by FK).

The instants (fixed before any outcome was looked at; see README):
  first contact  first round-0 step where a task hand presses the object with >= CONTACT_N
  P  pre-contact  the step before the earlier of first contact and hand close; if neither
                  happens, the last raise/advance step of round 0
  C  close        the first round-0 step with the hands closed
  W  window       mean over the WINDOW steps that end at P
and the object: horizontal box/hammer displacement from first contact to close (zero when the
hands closed before touching), and the target's staleness (latched vs actual object position).

Usage (dev box, policy Python):
  python stage_split.py episodes --out DIR [--runs RUNS] [--workers N] [--threshold N]
  python stage_split.py summary --out DIR
"""
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CONTACT_N = .05          # = grasp_judge.Criteria.hand_force_n, the code's own definition of a touch
SENSITIVITY_N = 1.       # second threshold, reported as a sensitivity check
WINDOW = 25              # 0.5 s at 50 Hz
PREFIXES = ('exp1-', 'exp1b-', 'exp4-')
BOX_PITCH_DEG = 20.      # agentic/configs/grasp-loop.json policy.wrist_pitch_down_deg (never changed)
HAMMER_PITCH_DEG = 40.   # agentic/configs/hammer-grasp.json policy.wrist_pitch_down_deg (never changed)
STANDING = ('raise', 'advance', 'grip', 'squeeze', 'lift', 'hold', 'release')
BODIES = ('left_wrist_yaw_link', 'right_wrist_yaw_link', 'torso_link')
QUANTITIES = ('wrist_pos', 'wrist_rot', 'root_pos', 'torso_rot')
STAGES = ('lag', 'correction', '1a', '1b', 'handoff', '1', '2', 'total')
FEEDBACK_GAIN, FEEDBACK_MAX_M = .7, .12   # policy.reach_feedback_gain / _max_m, same in both configs, never changed
PRIMARY = ('P.wrist_pos.1', 'P.wrist_pos.2', 'push_contact_to_close_m')


# ---- small geometry ---------------------------------------------------------------------------
def quat_matrix(q):
    """3x3 rotation from a (w, x, y, z) quaternion."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                     [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                     [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])


def angle_deg(a, b):
    """Angle of the relative rotation between two 3x3 rotations, degrees."""
    c = (np.trace(np.asarray(a).T @ np.asarray(b)) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1., 1.))))


def pitch_down(deg):
    """Rotation about world y that tips a +x-facing wrist ``deg`` downwards (the box grasp pose)."""
    p = np.radians(deg)
    return np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])


def hammer_rotation(axis, deg=HAMMER_PITCH_DEG):
    """Right-wrist grasp rotation over a handle (same formula as hammer_policy.grasp_rotation)."""
    axis = np.asarray(axis, dtype=float)
    across = np.array([axis[1], -axis[0], 0.])
    p = np.radians(deg)
    up = np.array([0., 0., 1.])
    return np.stack([np.cos(p)*across - np.sin(p)*up, -np.sin(p)*across - np.cos(p)*up, axis], axis=1)


def smooth(u):
    u = np.clip(u, 0., 1.)
    return u**3 * (10 - 15*u + 6*u*u)


def rotvec_matrix(v):
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)
    k = np.asarray(v) / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def slerp(a, b, w):
    """Rotation a fraction ``w`` of the way from ``a`` to ``b``."""
    rel = np.asarray(a).T @ np.asarray(b)
    theta = np.arccos(np.clip((np.trace(rel) - 1) / 2, -1., 1.))
    if theta < 1e-9:
        return np.asarray(a).copy()
    if np.pi - theta < 1e-6:   # half turn: the axis is the eigenvector of +1
        values, vectors = np.linalg.eig(rel)
        axis = np.real(vectors[:, np.argmin(abs(values - 1))])
    else:
        axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]) / (2 * np.sin(theta))
    return np.asarray(a) @ rotvec_matrix(axis * theta * w)


def path_pieces(kind, row, rest_pos, rest_rot):
    """The planned wrist path of one attempt as [(end time, positions [2,3], rotations [2,3,3])], starting at rest.

    Box (grasp_policy.anchor): rest -> via by raise_end -> pre by advance_end, both hands, wrists
    pitched down. Hammer (hammer_policy.anchor): the right hand rest -> via -> pre (over_end) ->
    grip (advance_end); the left hand stays at rest.
    """
    wrists = row['wrist_targets']
    rest_pos, rest_rot = np.asarray(rest_pos, float), np.asarray(rest_rot, float)
    if kind == 'box':
        rot = np.stack([pitch_down(BOX_PITCH_DEG)] * 2)
        return [(row['raise_end'], np.asarray(wrists['via'], float), rot),
                (row['advance_end'], np.asarray(wrists['pre'], float), rot)]
    rot = np.stack([rest_rot[0], hammer_rotation(row['handle_axis'])])
    both = lambda right: np.stack([rest_pos[0], np.asarray(right, float)])
    return [(row['raise_end'], both(wrists['via']), rot), (row['over_end'], both(wrists['pre']), rot),
            (row['advance_end'], both(wrists['grip']), rot)]


def planned_wrists(pieces, start, rest_pos, rest_rot, t):
    """Positions and rotations of the planned path at time ``t`` (WristPath semantics)."""
    t0, p0, r0 = start, np.asarray(rest_pos, float), np.asarray(rest_rot, float)
    for t1, p1, r1 in pieces:
        if t < t1:
            w = float(smooth((t - t0) / (t1 - t0))) if t1 > t0 else 1.
            return p0 + w * (p1 - p0), np.stack([slerp(a, b, w) for a, b in zip(r0, r1)])
        t0, p0, r0 = t1, p1, r1
    return p0, r0


def replay_bias(plan_times, wanted, measured, hands, gain=FEEDBACK_GAIN, limit=FEEDBACK_MAX_M):
    """The policy's wrist correction (grasp_policy.correct) after each plan: [(time, bias [2,3])].

    ``wanted(t)`` and ``measured(t)`` give the planned and the measured wrists [2,3] at a plan time;
    only plans made in the advance phase after its planned end belong here.
    """
    bias, out = np.zeros((2, 3)), []
    for t in plan_times:
        step = bias + gain * (np.asarray(wanted(t)) - np.asarray(measured(t)))
        for h in hands:
            size = np.linalg.norm(step[h])
            bias[h] = step[h] * min(1., limit / max(size, 1e-9))
        out.append((t, bias.copy()))
    return out


def bias_at(history, t):
    """The correction in force for the command executed at step time ``t`` (made by an earlier plan)."""
    chosen = np.zeros((2, 3))
    for when, bias in history:
        if when < t - 1e-3:
            chosen = bias
    return chosen


# ---- per-episode timing -------------------------------------------------------------------------
def key(t):
    return int(round(float(t) * 50))


def task_hands(kind):
    return (0, 1) if kind == 'box' else (1,)


def first_contact(steps, hands, threshold, round_index=0):
    """Index into ``steps`` of the first step of the round where a task hand presses >= threshold."""
    names = [('left_n', 'right_n')[h] for h in hands]
    for i, s in enumerate(steps):
        if s.get('round') == round_index and max(s.get(n, 0.) for n in names) >= threshold:
            return i
    return None


def first_close(steps, round_index=0):
    return next((i for i, s in enumerate(steps) if s.get('round') == round_index and s.get('hands_closed')), None)


def pre_contact(steps, contact, close, round_index=0):
    """Index of the step before the earlier of contact and close, else the last reach step of the round."""
    events = [i for i in (contact, close) if i is not None]
    if events:
        return max(min(events) - 1, 0)
    reach = [i for i, s in enumerate(steps) if s.get('round') == round_index and s.get('phase') in ('raise', 'advance')]
    return reach[-1] if reach else None


def failure_mode(contact, close):
    """How the first attempt met the object: the hands never closed (the body did not arrive), the
    hands touched it before closing (pushed while approaching), or they closed first as designed."""
    if close is None:
        return 'never_closed'
    return 'touched_before_close' if contact is not None and contact < close else 'closed_first'


def latest_commands(rows):
    """Per control step, the command row of the newest plan that covers it."""
    out = {}
    for row in rows:
        k = key(row['time'])
        if k not in out or row['plan'] >= out[k]['plan']:
            out[k] = row
    return out


def round_targets(rounds, round_index=0):
    """[(from time, wrist targets, notes row)] of one round, in log order."""
    out = []
    for row in rounds:
        if row.get('round') == round_index and 'wrist_targets' in row:
            out.append((float(row.get('now', row['time'])), row['wrist_targets'], row))
    return out


def target_at(targets, t):
    chosen = None
    for start, wrists, row in targets:
        if start <= t + 1e-6:
            chosen = (wrists, row)
    return chosen if chosen is not None else ((targets[0][1], targets[0][2]) if targets else None)


# ---- statistics ------------------------------------------------------------------------------------
def auc(fail, succ):
    """P(a failed episode's value > a successful one's), ties count half (Mann-Whitney AUC)."""
    fail, succ = np.asarray(fail, float), np.asarray(succ, float)
    if not len(fail) or not len(succ):
        return float('nan')
    greater = (fail[:, None] > succ[None, :]).sum()
    ties = (fail[:, None] == succ[None, :]).sum()
    return float((greater + .5 * ties) / (len(fail) * len(succ)))


def bootstrap_auc(fail, succ, n=2000, seed=0):
    """95% percentile interval of the AUC, resampling failures and successes separately."""
    fail, succ = np.asarray(fail, float), np.asarray(succ, float)
    if len(fail) < 2 or len(succ) < 2:
        return [float('nan'), float('nan')]
    rng = np.random.default_rng(seed)
    values = [auc(rng.choice(fail, len(fail)), rng.choice(succ, len(succ))) for _ in range(n)]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def mannwhitney_p(fail, succ):
    """Two-sided p by the normal approximation with tie correction (no scipy needed)."""
    fail, succ = np.asarray(fail, float), np.asarray(succ, float)
    n1, n2 = len(fail), len(succ)
    if n1 < 1 or n2 < 1:
        return float('nan')
    values = np.concatenate([fail, succ])
    order = values.argsort(kind='mergesort')
    ranks = np.empty(len(values))
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    _, counts = np.unique(values, return_counts=True)
    n = n1 + n2
    var = n1 * n2 / 12 * ((n + 1) - (counts**3 - counts).sum() / (n * (n - 1)))
    if var <= 0:
        return 1.
    z = (abs(u - n1 * n2 / 2) - .5) / math.sqrt(var)
    return float(min(1., math.erfc(max(z, 0.) / math.sqrt(2))))


def wilson(k, n, z=1.96):
    if n == 0:
        return [float('nan'), float('nan')]
    p = k / n
    centre, half = (p + z*z/(2*n)) / (1 + z*z/n), z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [centre - half, centre + half]


def separates(entry):
    """The pre-registered rule: enough of both outcomes, AUC away from 0.5 and p < 0.01."""
    if entry['n_fail'] < 5 or entry['n_succ'] < 5 or not np.isfinite(entry['auc']):
        return 'too_few'
    lo, hi = entry['auc_ci']
    if entry['p'] < .01 and ((entry['auc'] >= .7 and lo > .5) or (entry['auc'] <= .3 and hi < .5)):
        return 'yes'
    return 'no'


# ---- one episode -------------------------------------------------------------------------------------
def classify(name, result, rounds):
    """Family, cell and condition of a run folder."""
    family = 'hammer' if name.startswith('exp1b-') else 'box-latency' if name.startswith('exp4-') else 'box'
    cell = re.sub(r'-2026\d{4}.*$', '', name)
    layout = result.get('layout') or {}
    moving = float(layout.get('speed_mps') or 0.) > 0 or '-move' in cell
    walking = any(r.get('round') == 0 and r.get('phase') == 'walk' for r in rounds)
    condition = ('moving' if moving else 'static') + ('-walk' if walking else '-stand')
    if '-inj-' in cell:
        condition += '-injected'
    return family, cell, condition


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Layers:
    """Body poses of the four layers at one control step (None where a layer is not logged)."""
    def __init__(self, fk):
        self.fk = fk

    def pose(self, joints, root, rotation):
        positions, rotations = self.fk(np.asarray(joints, float), np.asarray(root, float), np.asarray(rotation, float),
                                       body_names=BODIES)
        return {'wrist_pos': positions[:2], 'wrist_rot': rotations[:2], 'root_pos': np.asarray(root, float),
                'torso_rot': rotations[2]}

    def at(self, execution, command):
        after = execution['after']
        out = {'A': {**self.pose(after['joints'], after['root_pos'], quat_matrix(after['root_quat'])),
                     'wrist_pos': np.asarray(after['wrist_pos']), 'wrist_rot': np.asarray(after['wrist_rot'])}}
        if 'reference_joints' in execution and 'reference_root_world' in execution:
            out['R'] = self.pose(execution['reference_joints'], execution['reference_root_world'],
                                 quat_matrix(execution['reference_root_quat']))
        if command is not None:
            out['K'] = self.pose(command['raw_joints'], command['raw_root_world'], command['raw_root_rotation_world'])
            out['S'] = self.pose(command['sent_joints'], command['sent_root_world'], command['sent_root_rotation_world'])
        return out


def distance(quantity, a, b, hands):
    if a is None or b is None:
        return float('nan')
    if quantity == 'wrist_pos':
        return float(np.mean([np.linalg.norm(a['wrist_pos'][h] - b['wrist_pos'][h]) for h in hands]))
    if quantity == 'wrist_rot':
        return float(np.mean([angle_deg(a['wrist_rot'][h], b['wrist_rot'][h]) for h in hands]))
    if quantity == 'root_pos':
        return float(np.linalg.norm(a['root_pos'][:2] - b['root_pos'][:2]))
    return angle_deg(a['torso_rot'], b['torso_rot'])


def stage_errors(layers, targets, hands):
    """Every stage of every quantity at one step: {'wrist_pos.1a': ..., ...}.

    ``targets`` holds the task target (``task``), the planned path at this step (``plan``) and the
    command the planner asked of Kimodo and IK (``command`` = plan + the closed-loop correction).
    """
    T, Tp, Tc = targets['task'], targets['plan'], targets['command']
    K, S, R, A = (layers.get(k) for k in 'KSRA')
    pairs = {'lag': (T, Tp), 'correction': (Tp, Tc), '1a': (Tc, K), '1b': (K, S), 'handoff': (S, R),
             '1': (Tc, R), '2': (R, A), 'total': (T, A)}
    return {f'{q}.{stage}': distance(q, *pairs[stage], hands) for q in QUANTITIES for stage in STAGES}


def target_poses(kind, row, rest, anchor_torso, t, bias):
    """Task target, planned path and commanded target at time ``t`` as pose dicts."""
    pieces = path_pieces(kind, row, rest[0], rest[1])
    task_pos, task_rot = pieces[-1][1], pieces[-1][2]
    plan_pos, plan_rot = planned_wrists(pieces, rest[2], rest[0], rest[1], t)
    common = {'root_pos': np.asarray(row['anchor_root'], float), 'torso_rot': anchor_torso}
    return {'task': {'wrist_pos': task_pos, 'wrist_rot': task_rot, **common},
            'plan': {'wrist_pos': plan_pos, 'wrist_rot': plan_rot, **common},
            'command': {'wrist_pos': plan_pos + bias, 'wrist_rot': plan_rot, **common}}


def episode(run, fk, threshold=CONTACT_N, sensitivity=SENSITIVITY_N):
    """One record per run folder, or {'run', 'skip': reason}."""
    run = Path(run)
    try:
        result = json.loads((run / 'grasp-result.json').read_text())
        rounds = read_jsonl(run / 'rounds.jsonl')
        steps = read_jsonl(run / 'grasp-steps.jsonl')
    except (OSError, ValueError) as exc:
        return {'run': run.name, 'skip': f'unreadable: {exc}'}
    family, cell, condition = classify(run.name, result, rounds)
    kind = 'hammer' if family == 'hammer' else 'box'
    hands = task_hands(kind)
    base = {'run': run.name, 'family': family, 'cell': cell, 'condition': condition, 'seed': result.get('seed'),
            'success': bool(result.get('success')), 'first_success': bool(result.get('first_attempt_success'))}
    targets = round_targets(rounds)
    if not targets:
        return {**base, 'skip': 'round 0 never anchored (no wrist targets)'}
    execution = {key(r['after']['time']): r for r in read_jsonl(run / 'execution.jsonl')}
    commands = latest_commands(read_jsonl(run / 'commands.jsonl')) if (run / 'commands.jsonl').exists() else {}
    layers = Layers(fk)

    contact = first_contact(steps, hands, threshold)
    contact_hi = first_contact(steps, hands, sensitivity)
    close = first_close(steps)
    P = pre_contact(steps, contact, close)
    if P is None:
        return {**base, 'skip': 'no reach step in round 0'}
    anchor_time = targets[0][0]
    anchor_row = execution.get(key(anchor_time)) or execution[min(execution, key=lambda k: abs(k - key(anchor_time)))]
    anchor_torso = layers.at(anchor_row, None)['A']['torso_rot']
    rest = (np.asarray(anchor_row['after']['wrist_pos'], float), np.asarray(anchor_row['after']['wrist_rot'], float),
            anchor_time)
    measured = lambda t: np.asarray((execution.get(key(t)) or {'after': {'wrist_pos': np.full((2, 3), np.nan)}})
                                    ['after']['wrist_pos'], float)

    def wanted(t):
        _, row = target_at(targets, t)
        return planned_wrists(path_pieces(kind, row, rest[0], rest[1]), anchor_time, rest[0], rest[1], t)[0]

    # The closed-loop wrist correction, replayed from the plans made after the advance should have ended.
    reach_end = max((s['time'] for s in steps if s.get('round') == 0 and s.get('phase') == 'advance'), default=anchor_time)
    stand_plans = [p['time'] for p in read_jsonl(run / 'plans.jsonl') if 'kimodo_fk_target_m' in p]
    corrections = [t for t in stand_plans if anchor_time - 1e-6 <= t <= reach_end + 1e-6
                   and t >= target_at(targets, t)[1]['advance_end'] - 1e-6]
    history = replay_bias(corrections, wanted, measured, hands)

    def at(index):
        t = steps[index]['time']
        row = execution.get(key(t))
        if row is None:
            return None
        _, notes = target_at(targets, t)
        return stage_errors(layers.at(row, commands.get(key(t))),
                            target_poses(kind, notes, rest, anchor_torso, t, bias_at(history, t)), hands)

    out = {**base, 'kind': kind, 'contact': contact is not None, 'closed': close is not None,
           'contact_before_close': contact is not None and (close is None or contact < close),
           'contact_hi': contact_hi is not None,
           'robot_other_before_P': any(s.get('round') == 0 and s.get('robot_other_n', 0.) >= threshold
                                       for s in steps[:P + 1]),
           'mode': failure_mode(contact, close),
           'first_label': ((result.get('attempts') or [{}])[0] or {}).get('label'),
           'time_P': steps[P]['time'], 'phase_P': steps[P].get('phase'),
           'commands_at_P': key(steps[P]['time']) in commands}
    for name, value in at(P).items():
        out[f'P.{name}'] = value
    window = [at(i) for i in range(max(P - WINDOW + 1, 0), P + 1) if steps[i].get('round') == 0]
    window = [w for w in window if w is not None]
    for name in window[0] if window else ():
        values = [w[name] for w in window if np.isfinite(w[name])]
        out[f'W.{name}'] = float(np.mean(values)) if values else float('nan')
    if close is not None:
        for name, value in at(close).items():
            out[f'C.{name}'] = value
    if contact_hi is not None:   # the sensitivity threshold moves P only when it moves the earlier event
        P_hi = pre_contact(steps, contact_hi, close)
        out['P_hi.wrist_pos.1'], out['P_hi.wrist_pos.2'] = (at(P_hi) or {}).get('wrist_pos.1'), (at(P_hi) or {}).get('wrist_pos.2')

    # The object between first contact and close, and how stale the target was.
    obj = lambda i: np.asarray(steps[i]['box_com'], float)
    if contact is not None and (close is None or contact < close):
        end = close if close is not None else P
        out['push_contact_to_close_m'] = float(np.linalg.norm(obj(end)[:2] - obj(contact)[:2]))
        before = [i for i in range(max(contact - 10, 0), contact) if steps[i].get('round') == 0]
        out['object_speed_at_contact_mps'] = (float(np.linalg.norm(obj(contact)[:2] - obj(before[0])[:2]) /
                                              (steps[contact]['time'] - steps[before[0]]['time'])) if before else 0.)
    else:
        out['push_contact_to_close_m'] = 0.
        out['object_speed_at_contact_mps'] = float('nan')
    latched = np.asarray(target_at(targets, steps[P]['time'])[1].get('target_box_pos') or
                         target_at(targets, steps[P]['time'])[1]['box_pos'], float)
    out['staleness_P_m'] = float(np.linalg.norm(obj(P)[:2] - latched[:2]))
    if close is not None:
        latched_c = target_at(targets, steps[close]['time'])[1]
        out['staleness_C_m'] = float(np.linalg.norm(
            obj(close)[:2] - np.asarray(latched_c.get('target_box_pos') or latched_c['box_pos'], float)[:2]))

    grip_row = next((r for r in rounds if r.get('round') == 0 and r.get('phase') == 'grip' and 'wrist_bias' in r), None)
    if grip_row is not None and history:
        logged = np.asarray(grip_row['wrist_bias'], float).reshape(-1, 3)
        replayed = bias_at(history, grip_row['time'] + 1.)[[1] if kind == 'hammer' else [0, 1]]
        out['bias_replay_error_m'] = float(np.abs(logged - replayed).max())
    out['correction_plans'] = len(corrections)

    # Kimodo against what it was asked for: the newest stand plan before P, at its keyframe.
    plans = [p for p in read_jsonl(run / 'plans.jsonl') if 'kimodo_fk_target_m' in p and p['time'] <= steps[P]['time'] + 1e-6]
    if plans:
        last = plans[-1]
        out['P.asked.kimodo_fk_target_m'] = float(np.mean([last['kimodo_fk_target_m'][h][0] for h in hands]))
        out['P.asked.kimodo_angle_deg'] = float(np.mean([last['projected_wrist_angle_deg'][h] for h in hands]))

    out.update(walk_split(run, rounds, execution))
    return out


def walk_split(run, rounds, execution):
    """Round 0's walk to the stance, plan by plan, in relative terms.

    SONIC follows the reference's root motion, not its absolute position (the reference root drifts
    away from the measured one during a walk), so each plan's span is compared as displacements:
    stage 1 = Kimodo's own logged root error against the requested path (clip walks: ``end_error_m``;
    official-way walks: ``root_path_error_m``), stage 2 = |measured displacement - consumed
    reference displacement| over the span. Total = measured stance against the walk goal.
    """
    walk = next((r for r in rounds if r.get('round') == 0 and r.get('phase') == 'raise' and 'walk_goal' in r), None)
    settle = next((r for r in rounds if r.get('round') == 0 and r.get('phase') == 'settle'), None)
    if walk is None or settle is None:
        return {}
    native = walk.get('walk_generator') == 'native'
    if native:
        path = run / 'walk-r0' / 'plans.jsonl'
        plans = read_jsonl(path) if path.exists() else []
        errors = [p.get('root_path_error_m') for p in plans]
    else:
        plans = [p for p in read_jsonl(run / 'plans.jsonl') if p.get('kind') == 'walk' and p.get('round') == 0]
        errors = [p.get('end_error_m') for p in plans]
    times = [p['time'] for p in plans] + [settle['time']]
    stage2, travel, reference_travel = [], 0., 0.
    for a, b in zip(times[:-1], times[1:]):
        ra, rb = execution.get(key(a)), execution.get(key(b))
        if ra is None or rb is None or 'reference_root_world' not in ra or 'reference_root_world' not in rb:
            continue
        dR = np.subtract(rb['reference_root_world'][:2], ra['reference_root_world'][:2])
        dA = np.subtract(rb['after']['root_pos'][:2], ra['after']['root_pos'][:2])
        stage2.append(float(np.linalg.norm(dA - dR)))
        travel, reference_travel = travel + np.linalg.norm(dA), reference_travel + np.linalg.norm(dR)
    end = execution.get(key(settle['time']))
    errors = [e for e in errors if isinstance(e, (int, float))]
    out = {'walk.native': native, 'walk.plans': len(plans),
           'walk.root_pos.1': float(np.mean(errors)) if errors else float('nan'),
           'walk.root_pos.2': float(np.mean(stage2)) if stage2 else float('nan'),
           'walk.travel_ratio': float(travel / reference_travel) if reference_travel > 0 else float('nan')}
    if end is not None:
        out['walk.root_pos.total'] = float(np.linalg.norm(np.subtract(end['after']['root_pos'][:2], walk['walk_goal'])))
    return out


# ---- the summary -----------------------------------------------------------------------------------------
def metrics(records):
    names = set()
    for r in records:
        names.update(k for k, v in r.items() if k[:2] in ('P.', 'W.', 'C.')
                     or k.startswith(('walk.root', 'walk.travel', 'push', 'stale', 'P_hi.', 'object_')))
    return sorted(n for n in names if not n.endswith('.native'))


def compare(records, name):
    succ = [r[name] for r in records if r['first_success'] and isinstance(r.get(name), (int, float)) and np.isfinite(r[name])]
    fail = [r[name] for r in records if not r['first_success'] and isinstance(r.get(name), (int, float)) and np.isfinite(r[name])]
    entry = {'n_succ': len(succ), 'n_fail': len(fail),
             'median_succ': float(np.median(succ)) if succ else None, 'median_fail': float(np.median(fail)) if fail else None,
             'q_succ': [float(np.percentile(succ, q)) for q in (25, 75)] if succ else None,
             'q_fail': [float(np.percentile(fail, q)) for q in (25, 75)] if fail else None,
             'auc': auc(fail, succ), 'auc_ci': bootstrap_auc(fail, succ), 'p': mannwhitney_p(fail, succ)}
    entry['cliffs_delta'] = 2 * entry['auc'] - 1 if np.isfinite(entry['auc']) else float('nan')
    entry['separates'] = separates(entry)
    return entry


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        if 'skip' not in r:
            groups[(r['family'], r['condition'])].append(r)
    out = []
    for (family, condition), rows in sorted(groups.items()):
        k = sum(r['first_success'] for r in rows)
        out.append({'family': family, 'condition': condition, 'n': len(rows), 'first_success': k,
                    'first_success_ci': wilson(k, len(rows)), 'cells': sorted({r['cell'] for r in rows}),
                    'contact_before_close': sum(r['contact_before_close'] for r in rows),
                    'closed': sum(r['closed'] for r in rows),
                    'commands_at_P': sum(r['commands_at_P'] for r in rows),
                    'modes': {outcome: {m: sum(1 for r in rows if r['mode'] == m and r['first_success'] == ok)
                                        for m in ('never_closed', 'touched_before_close', 'closed_first')}
                              for outcome, ok in (('success', True), ('failure', False))},
                    'metrics': {name: compare(rows, name) for name in metrics(rows)}})
    return out


def verdict(group):
    """The plan's reading for one group, from the two primary stage contrasts at P."""
    m = group['metrics']
    s1, s2 = m.get('P.wrist_pos.1', {}).get('separates'), m.get('P.wrist_pos.2', {}).get('separates')
    if 'too_few' in (s1, s2) or None in (s1, s2):
        return 'cannot conclude (too few failures or successes)'
    return {('yes', 'yes'): 'both stages separate', ('yes', 'no'): 'stage 1 separates (generation side)',
            ('no', 'yes'): 'stage 2 separates (tracker side)', ('no', 'no'): 'neither stage separates'}[(s1, s2)]


# ---- entry points ------------------------------------------------------------------------------------------
_FK = None


def _work(args):
    global _FK
    run, threshold = args
    if _FK is None:
        from paths import G1_XML
        from kimodo_vla.core.types import G1_JOINT_NAMES
        from kimodo_vla.oracle.geometry import XmlFK
        _FK = XmlFK(G1_XML, G1_JOINT_NAMES)
    try:
        return episode(run, _FK, threshold)
    except Exception as exc:   # one broken folder must not stop the batch; it is listed as skipped
        return {'run': Path(run).name, 'skip': f'{type(exc).__name__}: {exc}'}


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('command', choices=('episodes', 'summary'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=CONTACT_N)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == 'episodes':
        from paths import RUNS
        root = args.runs or RUNS
        runs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(PREFIXES)
                      and (p / 'grasp-result.json').exists())[:args.limit]
        from multiprocessing import Pool
        with Pool(args.workers) as pool, (args.out / 'episodes.jsonl').open('w') as f:
            for record in pool.imap_unordered(_work, [(str(r), args.threshold) for r in runs], chunksize=4):
                f.write(json.dumps(record) + '\n')
        print(f'{len(runs)} run folders -> {args.out / "episodes.jsonl"}')
    else:
        records = read_jsonl(args.out / 'episodes.jsonl')
        groups = summarize(records)
        for g in groups:
            g['verdict'] = verdict(g)
        skipped = defaultdict(int)
        for r in records:
            if 'skip' in r:
                skipped[r['skip'].split(':')[0]] += 1
        (args.out / 'summary.json').write_text(json.dumps({'contact_n': args.threshold, 'primary': PRIMARY,
                                                          'skipped': skipped, 'groups': groups}, indent=1))
        print(json.dumps({'skipped': skipped, 'groups': [(g['family'], g['condition'], g['n'], g['first_success'],
                                                          g['verdict']) for g in groups]}, indent=1))


if __name__ == '__main__':
    main()
