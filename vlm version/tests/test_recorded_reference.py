import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from kimodo_vla.oracle.recorded_reference import RecordedReference,export_reference,read_rows
from kimodo_vla.oracle.geometry import XmlFK
from kimodo_vla.core.types import G1_JOINT_NAMES

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_sonic_replay import components


def test_original_chunks_exact_and_no_feedback_replanning(tmp_path):
    source=ROOT/'runs/vlm-refined-20261118-r1-base-center'
    if not source.exists():pytest.skip('Archived experiment unavailable')
    fixture=tmp_path/'fixture';export_reference(source,fixture)
    run=tmp_path/'run';run.mkdir();policy=RecordedReference(run,fixture)
    rows=read_rows(source/'execution.jsonl')
    output=[];offset=0
    for index,length in enumerate(policy.lengths):
        sample=copy.deepcopy(rows[offset]['before'])
        if index:
            # Actions must not depend on subsequent measured pose or visual contents.
            sample['joints']=[123.]*29;sample['root_pos']=[10.,20.,30.]
        result=policy({'observation':{'oracle_history':[sample],'vision':{'not_used':True}}})
        output.extend(result['action_chunk']);offset+=int(length)
    np.testing.assert_array_equal(output,policy.actions)
    original=np.concatenate([np.load(p)['actions'] for p in sorted(source.glob('plan-*.npz'))])
    np.testing.assert_array_equal(np.array(output)[5:],original)
    np.testing.assert_array_equal(np.array(output)[:5,9:38],np.array([rows[0]['before']['joints']]*5))
    assert list(policy.lengths)==[5]+[25]*16
    with pytest.raises(ValueError,match='exhausted'):policy({})
    policy.reset();bad=copy.deepcopy(rows[0]['before']);bad['root_pos'][0]+=.01
    with pytest.raises(ValueError,match='Initial state'):policy({'observation':{'oracle_history':[bad]}})
    good=copy.deepcopy(rows[0]['before']);policy({'observation':{'oracle_history':[good]}})
    with pytest.raises(ValueError,match='timing'):policy({'observation':{'oracle_history':[good]}})
    with (fixture/'reference.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='hash'):RecordedReference(run,fixture)


def test_rigid_root_motion_and_arm_error_are_separate():
    fk=XmlFK(ROOT/'kimodo_vla/skeleton/assets/g1skel34/xml/g1.xml',G1_JOINT_NAMES)
    q=np.zeros(29);p=np.array([0.,0.,.8]);r=Rotation.from_euler('xyz',[.1,-.2,.3])
    quat=r.as_quat()[[3,0,1,2]].tolist();actualp=p+np.array([.03,-.04,.01])
    ref=fk(q,p,np.eye(3))[0];actual=fk(q,actualp,r.as_matrix())[0]
    e=dict(after=dict(joints=q.tolist(),root_pos=actualp.tolist(),root_quat=quat,wrist_pos=actual.tolist()),
        reference_joints=q.tolist(),reference_root_world=p.tolist(),reference_root_quat=[1.,0.,0.,0.],reference_wrist_world=ref.tolist())
    c,_=components(e,fk)
    np.testing.assert_allclose(c['root_relative'],0,atol=1e-12)
    np.testing.assert_allclose(c['root_translation']+c['root_rotation'],c['tracking'],atol=1e-12)
    q[G1_JOINT_NAMES.index('right_shoulder_pitch_joint')]=.2
    e['after']['joints']=q.tolist();e['after']['wrist_pos']=fk(q,actualp,r.as_matrix())[0].tolist()
    c,_=components(e,fk)
    np.testing.assert_allclose(c['torso'],0,atol=1e-12)
    np.testing.assert_allclose(c['arm'][0],0,atol=1e-12)
    assert np.linalg.norm(c['arm'][1])>.01


def test_audit_rejects_changed_consumed_reference(tmp_path):
    from analyze_sonic_replay import audit
    source=ROOT/'runs/vlm-refined-20261118-r1-base-center'
    if not source.exists():pytest.skip('Archived experiment unavailable')
    fixture=tmp_path/'fixture';export_reference(source,fixture)
    run=tmp_path/'run';run.mkdir()
    meta=json.loads((source/'run.json').read_text())
    meta.update(generator_loaded=False,vlm_model=None,reference_source='replay')
    (run/'run.json').write_text(json.dumps(meta))
    for name in ('execution.jsonl','sim-contract.json'):
        (run/name).write_bytes((source/name).read_bytes())
    (run/'target.json').write_bytes((fixture/'target.json').read_bytes())
    with np.load(fixture/'reference.npz') as data:
        (run/'commands.jsonl').write_text(''.join(json.dumps({'action':a.tolist()})+'\n' for a in data['actions']))
    assert audit(run,fixture,source)['passed']
    rows=read_rows(run/'execution.jsonl');rows[100]['reference_wrist_world'][1][0]+=.01
    (run/'execution.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='Consumed reference mismatch'):audit(run,fixture,source)
