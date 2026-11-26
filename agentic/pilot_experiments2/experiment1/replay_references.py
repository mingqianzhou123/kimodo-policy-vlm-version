"""Experiment 1 follow-up, step 2: replay logged grasp references on SONIC and on HEFT, open loop.

The reference SONIC consumed in a historical episode (``execution.jsonl``: reference root, root
orientation and 29 joints, from the anchor to P) becomes a replay segment
(``kimodo_vla/oracle/demo_walks.py`` format) and is played on the open floor with
``run_oracle.py --reference-source demo``: a 1 s blend from the settled stance, the reference,
its last frame held 1 s. No table, no box, no feedback.

Which references (fixed before any replay): in the box · static-stand group, the never-closed
failures whose reference was in reach and holdable (``reach_holdable.py``), four from
``exp1-map`` evenly spaced in run-name order and the first two from ``exp1-stance``; as
controls, two ``exp1-map`` successes evenly spaced and the first ``exp1-stance`` success.

What counts (fixed before any replay), at the end of the reference (the last frame of the
window) and averaged over its last 0.5 s: a tracker *follows* a reference when the wrists are
within 5 cm of the reference wrists (the grasp gate's tolerance), the pelvis is no more than
5 cm behind the reference pelvis along the heading, and it has not fallen. Per failing
reference: SONIC does not follow and HEFT does -> SONIC; neither follows -> the reference (or
both trackers); SONIC follows -> the failure does not reproduce open loop.

Usage (dev box):
  python replay_references.py segments --stage-out DIR --reach-out DIR --out DIR
  python replay_references.py launch --out DIR --gpus 0 3 [--date YYYYMMDD]
  python replay_references.py analyze --out DIR [--date YYYYMMDD]
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from stage_split import key, quat_matrix, read_jsonl, round_targets

PREFIX = 'p2e1replay'
SCENE_PREFIX = 'p2e1scene'   # the same references replayed in the grasp scene (table and box), SONIC only
CONTACT_N = 20.              # a non-foot body pressing at least this hard counts as a contact (scene replays)
LEAD = 10                 # standing frames before the reference, so the blend ends on a still pose
START_XY = (-4., -4.)     # open floor of the pp_box scene, as the HEFT record's replays
FOLLOW_WRIST_M, FOLLOW_BACKOFF_M = .05, .05
RATE = 50


def pick(rows, n, spread):
    rows = sorted(rows, key=lambda r: r['run'])
    if spread and len(rows) > n:
        return [rows[int(round(i))] for i in np.linspace(0, len(rows) - 1, n)]
    return rows[:n]


def select(reach):
    box = [r for r in reach if 'skip' not in r and r['group'] == 'box · static-stand']
    fail = [r for r in box if not r['first_success'] and r['mode'] == 'never_closed' and r['class'] == 'holdable']
    ok = [r for r in box if r['first_success']]
    chosen = [('fail', r) for r in pick([r for r in fail if r['cell'] == 'exp1-map'], 4, True)]
    chosen += [('fail', r) for r in pick([r for r in fail if r['cell'] == 'exp1-stance'], 2, False)]
    chosen += [('ok', r) for r in pick([r for r in ok if r['cell'] == 'exp1-map'], 2, True)]
    chosen += [('ok', r) for r in pick([r for r in ok if r['cell'] == 'exp1-stance'], 1, False)]
    return chosen


def to_actions(rows):
    """Reference rows -> Arena actions [T, 40]: root steps in the body frame, height, rot6d relative to the
    first heading, 29 joints, hands open (the layout of heft_segments.encode)."""
    pos = np.array([r['reference_root_world'] for r in rows], float)
    rot = np.stack([quat_matrix(r['reference_root_quat']) for r in rows])
    yaw0 = np.arctan2(rot[0, 1, 0], rot[0, 0, 0])
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    head = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rot = np.einsum('ij,tjk->tik', head, rot)
    local = (head @ (pos - pos[0]).T).T
    step = np.zeros_like(local)
    step[1:] = local[1:] - local[:-1]
    out = np.zeros((len(rows), 40))
    out[:, :2] = np.einsum('tji,tj->ti', rot, step)[:, :2]
    out[:, 2] = pos[:, 2]
    out[:, 3:9] = rot[:, :, :2].reshape(len(rows), 6)
    out[:, 9:38] = np.array([r['reference_joints'] for r in rows], float)
    return out


def segment_of(run, record):
    rounds = read_jsonl(run / 'rounds.jsonl')
    anchor = round_targets(rounds)[0][0]
    execution = {key(r['after']['time']): r for r in read_jsonl(run / 'execution.jsonl')}
    rows = [execution[k] for k in range(key(anchor), key(record['time_P']) + 1)
            if k in execution and 'reference_joints' in execution[k]]
    actions = to_actions([rows[0]] * LEAD + rows)
    return {'source': 'logged_reference', 'run': run.name, 'direction': 'in_place', 'motion': 'grasp-reach',
            'anchor_time': anchor, 'time_P': record['time_P'], 'window': [LEAD, len(actions) - 1],
            'actions': actions.tolist()}


def run_id(tag, tracker, date, prefix=PREFIX):
    return f'{prefix}-{tracker}-{tag}-{date}'


def placement(original_run):
    """run_oracle flags that rebuild an original grasp episode's scene: its map cell or stance, and its seed."""
    r = json.loads((original_run / 'run.json').read_text())
    flags = ['--seed', str(r['seed'])]
    cell = (r.get('layout_request') or {}).get('cell')
    if cell:
        return flags + ['--box-at', *(str(cell[k]) for k in ('depth', 'lateral', 'yaw', 'extra'))]
    if r.get('start_at_stance'):
        return flags + ['--start-at-stance']
    raise ValueError(f'{original_run.name}: neither a map cell nor a start at the stance')


def contacts(rows, t0, t1, heading, start_xy):
    """Scene replays: peak net force per non-foot body (box contacts reported apart), and whether the first
    contact of at least CONTACT_N came before the pelvis was 5 cm behind its start."""
    peak, box_peak, first, onset = {}, {}, None, None
    for r in rows:
        t = r['after']['time']
        if t < t0 - 1e-6 or t > t1 + 1e-6:
            continue
        for name, f in (r.get('body_contacts_n') or {}).items():
            peak[name] = max(peak.get(name, 0.), f)
            if f >= CONTACT_N and first is None:
                first = (t, name)
        for name, f in (r.get('box_contact_n') or {}).items():
            box_peak[name] = max(box_peak.get(name, 0.), f)
        back = -float((np.asarray(r['after']['root_pos'][:2]) - start_xy) @ heading)
        if back >= .05 and onset is None:
            onset = t
    top = sorted(peak.items(), key=lambda x: -x[1])[:5]
    return {'peak_contact_n': dict(top), 'box_contact_peak_n': box_peak,
            'first_contact': None if first is None else {'s': first[0] - t0, 'body': first[1]},
            'backoff_onset_s': None if onset is None else onset - t0,
            'contact_before_backoff': None if onset is None else bool(first is not None and first[0] <= onset)}


def task_in_replay(original, record_time_P, replay_rows, replay_time, segment_start):
    """The original task target (box: both ``pre`` wrists; hammer: ``grip``) carried into the replay's frame.

    The segment starts on the original reference at the anchor; the replay's reference root at the
    window start plays that role. Positions are moved with the yaw difference between the two.
    Added after the first replays (see README): the replayed reference carries the closed-loop
    correction, so "within 5 cm of the reference" is not "within the grasp gate of the target"."""
    from stage_split import target_at
    rounds = read_jsonl(original / 'rounds.jsonl')
    targets = round_targets(rounds)
    anchor = targets[0][0]
    execution = {key(r['after']['time']): r for r in read_jsonl(original / 'execution.jsonl')}
    a = execution[key(anchor)]
    wrists, _ = target_at(targets, record_time_P)
    task = np.asarray(wrists['pre'] if 'pre' in wrists else [wrists['grip']] * 2, float)
    i = int(np.argmin(abs(replay_time - segment_start)))
    b = replay_rows[i]
    yaw = lambda q: np.arctan2(quat_matrix(q)[1, 0], quat_matrix(q)[0, 0])
    d = yaw(b['reference_root_quat']) - yaw(a['reference_root_quat'])
    turn = np.array([[np.cos(d), -np.sin(d), 0], [np.sin(d), np.cos(d), 0], [0, 0, 1]])
    return (turn @ (task - np.asarray(a['reference_root_world'])).T).T + np.asarray(b['reference_root_world'])


def analyze_run(run, fk, original=None, time_P=None):
    target = json.loads((run / 'target.json').read_text())
    rows = [r for r in read_jsonl(run / 'execution.jsonl') if 'reference_root_world' in r]
    time_ = np.array([r['after']['time'] for r in rows])
    t0, t1 = target['window_time']
    end = int(np.argmin(abs(time_ - t1)))
    last = [i for i in range(len(rows)) if t1 - .5 - 1e-6 <= time_[i] <= t1 + 1e-6]
    heading = np.asarray(target['heading'])[:2, 0]

    def at(i):
        r = rows[i]
        wrist = float(np.mean(r['tracking_wrist_m']))
        behind = float((np.asarray(r['reference_root_world'][:2]) - np.asarray(r['after']['root_pos'][:2])) @ heading)
        ref = fk(np.asarray(r['reference_joints']), np.asarray(r['reference_root_world']),
                 quat_matrix(r['reference_root_quat']), body_names=['torso_link'])[1][0]
        act = fk(np.asarray(r['after']['joints']), np.asarray(r['after']['root_pos']),
                 quat_matrix(r['after']['root_quat']), body_names=['torso_link'])[1][0]
        lean = lambda m: float(np.degrees(np.arctan2(m[0, 2] * heading[0] + m[1, 2] * heading[1], m[2, 2])))
        return wrist, behind, lean(ref), lean(act)

    upto = time_ <= t1 + .5
    root_z = np.array([r['after']['root_pos'][2] for r in rows])
    tilt = np.array([np.degrees(np.arccos(np.clip(quat_matrix(r['after']['root_quat'])[2, 2], -1, 1))) for r in rows])
    fell = bool((root_z[upto] < .4).any() or (tilt[upto] > 60).any())
    w, b, lr, la = at(end)
    window = np.array([at(i) for i in last])
    out = {'run': run.name, 'fell': fell, 'wrist_end_m': w, 'behind_end_m': b, 'lean_ref_end_deg': lr, 'lean_end_deg': la,
           'wrist_last_m': float(window[:, 0].mean()), 'behind_last_m': float(window[:, 1].mean())}
    out['follows'] = bool(not fell and out['wrist_last_m'] <= FOLLOW_WRIST_M and out['behind_last_m'] <= FOLLOW_BACKOFF_M)
    start = rows[int(np.argmin(abs(time_ - t0)))]['after']['root_pos'][:2]
    out['max_backoff_m'] = max(-float((np.asarray(r['after']['root_pos'][:2]) - np.asarray(start)) @ heading)
                               for r, t in zip(rows, time_) if t0 - 1e-6 <= t <= t1 + 1e-6)
    if any('body_contacts_n' in r for r in rows):
        out.update(contacts(rows, t0, t1, heading, np.asarray(start)))
    if original is not None:
        task = task_in_replay(original, time_P, rows, time_, target['window_time'][0])
        hands = (0, 1) if 'exp1b-' not in original.name else (1,)
        to_task = lambda i, which: float(np.mean([np.linalg.norm(np.asarray(rows[i][which][h] if which == 'reference_wrist_world'
                                                                            else rows[i]['after']['wrist_pos'][h]) - task[h]) for h in hands]))
        out['robot_to_task_last_m'] = float(np.mean([to_task(i, 'after') for i in last]))
        out['reference_to_task_last_m'] = float(np.mean([to_task(i, 'reference_wrist_world') for i in last]))
    return out


def scene_reading(sonic):
    """Fixed before the scene replays (coordinating session's reading): back-off reappears with contact -> the
    table; back-off without contact -> SONIC in that scene; no back-off -> the replay still misses something."""
    if sonic is None:
        return 'incomplete'
    if sonic['max_backoff_m'] < .05:
        return 'no back-off'
    return 'back-off with contact first' if sonic.get('contact_before_backoff') else 'back-off without prior contact'


def verdict(sonic, heft):
    if sonic is None or heft is None:
        return 'incomplete'
    if sonic['follows']:
        return 'does not reproduce open loop'
    return 'SONIC' if heft['follows'] else 'reference (or both trackers)'


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiment2'))
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('action', choices=('segments', 'launch', 'analyze'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage-out', type=Path)
    p.add_argument('--reach-out', type=Path)
    p.add_argument('--gpus', type=int, nargs='+', default=[])
    p.add_argument('--date', default=time.strftime('%Y%m%d'))
    p.add_argument('--dds-domain', type=int, default=80)
    p.add_argument('--in-scene', action='store_true', help='launch / analyze the replays in the grasp scene (SONIC only)')
    a = p.parse_args(argv)
    from paths import AGENTIC, RUNS
    a.out.mkdir(parents=True, exist_ok=True)
    if a.action == 'segments':
        records = {r['run']: r for r in read_jsonl(a.stage_out / 'episodes.jsonl') if 'skip' not in r}
        manifest = []
        for i, (outcome, r) in enumerate(select(read_jsonl(a.reach_out / 'reach.jsonl'))):
            tag = f"{'f' if outcome == 'fail' else 's'}{i}"
            segment = segment_of(RUNS / r['run'], records[r['run']])
            (a.out / f'{tag}.json').write_text(json.dumps(segment))
            manifest.append({'tag': tag, 'outcome': outcome, 'run': r['run'], 'cell': r['cell'], 'frames': len(segment['actions']),
                             'original_max_backoff_m': r['max_backoff_m'], 'original_stage2_root_m': r.get('P.root_pos.2')})
        (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
        print(json.dumps(manifest, indent=1))
    elif a.action == 'launch':
        from exp2_replay import wait_free
        manifest = json.loads((a.out / 'manifest.json').read_text())
        jobs = [(m, t) for m in manifest for t in (('sonic',) if a.in_scene else ('sonic', 'heft'))]
        lock = threading.Lock()

        def worker(slot, gpu):
            while True:
                with lock:
                    if not jobs:
                        return
                    m, tracker = jobs.pop(0)
                rid = run_id(m['tag'], tracker, a.date, SCENE_PREFIX if a.in_scene else PREFIX)
                if (RUNS / rid).exists():
                    print('EXISTS', rid, flush=True)
                    continue
                if not wait_free(gpu):
                    print('GPU-BUSY', rid, flush=True)
                    continue
                steps = 50 + m['frames'] + 50 + 30
                cmd = [sys.executable, 'scripts/run_oracle.py', '--run-id', rid, '--reference-source', 'demo',
                       '--demo-segment', str(a.out / f"{m['tag']}.json"),
                       *(['--demo-in-scene', *placement(RUNS / m['run'])] if a.in_scene else ['--start-xy', *map(str, START_XY)]),
                       '--steps', str(steps), '--tracker', tracker, '--sim-gpu', str(gpu), '--policy-gpu', str(gpu),
                       '--share-policy-sim-gpu', '--gpu-min-free-mib', '16384']
                print('START', rid, 'gpu', gpu, flush=True)
                done = subprocess.run(cmd, cwd=AGENTIC, env=dict(os.environ, KVLA_DDS_DOMAIN=str(a.dds_domain + slot)),
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print('COMPLETE' if done.returncode == 0 else 'FAILED', rid, flush=True)

        threads = [threading.Thread(target=worker, args=(slot, gpu)) for slot, gpu in enumerate(a.gpus)]
        for thread in threads:
            thread.start()
            time.sleep(20)
        for thread in threads:
            thread.join()
    else:
        from kimodo_vla.core.types import G1_JOINT_NAMES
        from kimodo_vla.oracle.geometry import XmlFK
        from paths import G1_XML
        fk = XmlFK(G1_XML, G1_JOINT_NAMES)
        rows = []
        for m in json.loads((a.out / 'manifest.json').read_text()):
            result = {}
            time_P = json.loads((a.out / f"{m['tag']}.json").read_text())['time_P']
            for tracker in (('sonic',) if a.in_scene else ('sonic', 'heft')):
                run = RUNS / run_id(m['tag'], tracker, a.date, SCENE_PREFIX if a.in_scene else PREFIX)
                ok = (run / 'run.json').exists() and json.loads((run / 'run.json').read_text()).get('status') == 'completed'
                result[tracker] = analyze_run(run, fk, RUNS / m['run'], time_P) if ok else None
            if a.in_scene:
                rows.append({**m, **result, 'reading': scene_reading(result['sonic'])})
            else:
                rows.append({**m, **result, 'verdict': verdict(result['sonic'], result['heft']) if m['outcome'] == 'fail' else
                             ('controls: both follow' if all(x and x['follows'] for x in result.values()) else 'control not followed')})
        (a.out / ('replays-scene.json' if a.in_scene else 'replays.json')).write_text(json.dumps(rows, indent=1))
        print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
