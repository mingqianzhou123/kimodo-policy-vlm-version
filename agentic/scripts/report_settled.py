"""Compare matched tasks with different consumed references, including fixed marker truth."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from analyze_root_dynamics import analyze
from analyze_sonic_replay import vector_stats
from arena.dynamics_override import FIELDS
from arena.settled_start import WINDOW, stable
from kimodo_vla.oracle.geometry import rotation_wxyz
from kimodo_vla.oracle.recorded_reference import read_rows, sha, export_reference
from paths import RUNS
from run_settled_suite import CASES

OLD = 'dynamics-actuator-20261119-r1'


def load(run, filename):
    return json.loads((run/filename).read_text())


def wrist_stats(delta):
    stats = vector_stats(delta)
    norms = np.linalg.norm(delta, axis=-1)*100
    stats['strict_pass'] = bool(np.all(norms <= 3))
    stats['violating_frames'] = int(np.any(norms > 3, axis=1).sum())
    return stats


def measure(run, rows):
    after = [r['after'] for r in rows]
    times = np.array([a['time'] for a in after])
    start = rows[0]['before']['time']
    target = load(run,'target.json')
    assert len(rows)==400 and np.allclose(times-start, np.arange(1,401)/50, atol=1e-6,rtol=0)
    assert abs(target['arrival_time']-start-1.2)<1e-6
    mask = times > target['arrival_time']+1e-6
    assert mask.sum()==340
    def actual(key):
        return np.asarray([a[key] for a in after])[mask]
    def ref(key):
        return np.asarray([r[key] for r in rows])[mask]
    a = actual('wrist_pos')
    b = ref('reference_wrist_world')
    visual = np.asarray(target['targets_world'])[:,0]
    marker = np.asarray(load(run,'visual-evaluation-truth.json')['targets_world'])
    root = actual('root_pos')
    rroot = ref('reference_root_world')
    rot = np.stack([rotation_wxyz(q) for q in actual('root_quat')])
    rrot = np.stack([rotation_wxyz(q) for q in ref('reference_root_quat')])
    angles = Rotation.from_matrix(rrot.transpose(0,2,1)@rot).as_rotvec(degrees=True)
    results = {key:wrist_stats(delta) for key,delta in dict(ref_visual=b-visual,
        tracking=a-b, actual_visual=a-visual, ref_marker=b-marker, actual_marker=a-marker).items()}
    results['visual_marker_cm'] = (100*np.linalg.norm(visual-marker,axis=-1)).tolist()
    results['root_cm'] = vector_stats(root-rroot)
    results['root_xyz_max_abs_cm'] = (100*abs(root-rroot).max(axis=0)).tolist()
    results['orientation_deg'] = vector_stats(angles,scale=1)
    for key, mat in [('actual',rot),('reference',rrot)]:
        results[key+'_world_rpy_mean_deg'] = Rotation.from_matrix(mat).as_euler('xyz',degrees=True).mean(axis=0).tolist()
    for name,act,reference in [('feet',actual('foot_pos'),ref('reference_foot_world')),('wrists',a,b)]:
        results[name+'_root_frame_cm'] = vector_stats((act-root[:,None])@rot-(reference-rroot[:,None])@rrot)
        if name=='feet':results['feet_world_cm'] = vector_stats(act-reference)
    results['root_dynamics'] = analyze(run)['windows']
    return results


def compare(case, suite, dest):
    baseline = RUNS/(OLD+'-'+case)
    run = RUNS/(suite+'-'+case)
    source = RUNS/('vlm-refined-20261118-r1-'+case)
    old_target, target = load(source,'target.json'), load(run,'target.json')
    meta = load(run,'run.json')
    oldmeta = load(baseline,'run.json')
    assert meta['status']=='completed' and meta['generator_loaded'] and not meta['grasp_attempted']
    assert meta['archived_goal_sha256']==sha(source/'target.json')
    assert meta['dynamics_profile']=='actuator' and meta['root_support'] is None
    for key in ('arena_sha256','tracker_sha256','dynamics_profile_sha256'):
        assert meta[key]==oldmeta[key], key
    assert meta['checkpoint_sha256']==load(source,'run.json')['checkpoint_sha256']
    for key in ('targets_world','wrist_rotation_world'):
        assert target[key]==old_target[key]==load(baseline,'target.json')[key]
    assert load(run,'visual-evaluation-truth.json')==load(source,'visual-evaluation-truth.json')
    cfg, oldcfg = load(run,'actuator-config.json'),load(baseline,'actuator-config.json')
    for key in ('robot_asset','robot_cfg','sim_cfg','runtime_env','provider','body_names','joint_names','physics_dt','control_dt'):
        assert cfg[key]==oldcfg[key], key
    for key in (*FIELDS,'material_properties'):
        assert cfg['active_first_physics_step'][key]==oldcfg['active_first_physics_step'][key], key
    before = read_rows(baseline/'execution.jsonl')
    rows = read_rows(run/'execution.jsonl')
    settle = read_rows(run/'settling.jsonl')
    assert len(settle)>=WINDOW and all(stable(row) for row in settle[-WINDOW:])
    assert settle[-1]['consecutive']==WINDOW and rows[0]['before']['time']==settle[-1]['time']
    sample = rows[0]['before']
    support = target['support_target']
    for k,v in [('root_world','root_pos'),('ankle_world','foot_pos'),('ankle_rotation_world','foot_rot')]:
        assert support[k]==sample[v]
    assert np.allclose(support['root_rotation_world'],rotation_wxyz(sample['root_quat']),atol=1e-9)
    produced = read_rows(run/'commands.jsonl')[:395]
    delta = {}
    for k,v in [('sent_root_world','reference_root_world'),('sent_wrist','reference_wrist_world'),('sent_joints','reference_joints')]:
        delta[k]=float(np.max(np.abs(np.array([r[k] for r in produced])-np.array([r[v] for r in rows[5:]]))))
        assert delta[k]<2e-5, (case,k,delta[k])
    delta['sent_root_rotation_matrix'] = float(max(np.max(np.abs(
        np.asarray(a['sent_root_rotation_world'])-rotation_wxyz(b['reference_root_quat'])))
        for a,b in zip(produced,rows[5:])))
    assert delta['sent_root_rotation_matrix']<2e-5
    for row in rows[:5]:
        for k, v in [('reference_root_world','root_pos'),('reference_root_quat','root_quat'),('reference_joints','joints')]:
            assert np.allclose(row[k],sample[v],atol=2e-6,rtol=0), ('standing handoff',k)
    assert all(r['after']['gripper_command']==[0,0] for r in rows)
    physics = read_rows(run/'actuator-physics.jsonl')
    assert len(physics)==1600 and all((r['control'],r['substep'])==divmod(i,4) for i,r in enumerate(physics))
    assert all(np.array_equal(r['physx_position_target'],r['decoder_target']) for r in physics)
    fixture = dest/'fixtures'/case
    if not fixture.exists():export_reference(run,fixture)
    for name,h in load(fixture,'replay.json')['fixture_sha256'].items():assert sha(fixture/name)==h
    old_fixture=RUNS/OLD/'fixtures'/case
    for name,h in load(old_fixture,'replay.json')['fixture_sha256'].items():assert sha(old_fixture/name)==h
    reference_change=vector_stats(np.array([r['reference_wrist_world'] for r in rows])-np.array([r['reference_wrist_world'] for r in before]))
    return dict(old=measure(baseline,before),new=measure(run,rows),settling_frames=len(settle),
        settled_sample=sample, stable_last=settle[-1], handoff_max_abs=delta,
        old_reference_sha256=sha(old_fixture/'reference.npz'),new_reference_sha256=sha(fixture/'reference.npz'),
        reference_wrist_change_cm=reference_change)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',default='settled-20261123-r1')
    args=p.parse_args()
    dest=RUNS/args.suite
    dest.mkdir(exist_ok=True)
    results={case:compare(case,args.suite,dest) for case in CASES}
    (dest/'comparison.json').write_text(json.dumps(results,indent=2,allow_nan=False))
