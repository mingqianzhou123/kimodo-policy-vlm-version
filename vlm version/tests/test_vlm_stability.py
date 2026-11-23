import hashlib
import json
import numpy as np
import pytest
from kimodo_vla.vlm.stability import decompose,inspect_intent,stats


def test_signed_decomposition_does_not_sum_norms():
    truth=np.zeros((2,3));visual=np.ones((2,3));ref=np.tile(visual,(4,1,1));actual=np.zeros((4,2,3))
    result=decompose(actual,ref,visual,truth)
    np.testing.assert_allclose(result['total']['mean'],[0,0])
    np.testing.assert_allclose(result['tracking']['mean'],np.sqrt(3)*100)
    assert stats([]) is None


def test_rejected_intent_is_counted_and_capture_provenance_checked(tmp_path):
    capture=tmp_path/'capture.npz'
    np.savez(capture,rgb=np.zeros((5,5,3),dtype=np.uint8),depth=np.ones((5,5)),
             K=np.array([[10,0,2],[0,10,2],[0,0,1]]),camera_position=np.zeros(3),camera_rotation=np.eye(3),
             depth_kind='distance_to_image_plane',camera_convention='ros_optical')
    record={'capture_sha256':hashlib.sha256(capture.read_bytes()).hexdigest(),'load_s':1,'inference_s':2,
            'raw_output':'null','error':'left target is not visible'}
    path=tmp_path/'intent.json';path.write_text(json.dumps(record))
    truth=np.array([[0,0,1.],[0,0,1.]])
    out=inspect_intent(path,capture,truth,truth)
    assert out['accepted'] is False and out['rejection']==record['error']
    assert [x['error_cm'] for x in out['truth_center_depth_check']]==[0.,0.]
    record['capture_sha256']='wrong';path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='hash mismatch'):inspect_intent(path,capture,truth,truth)


def test_valid_depth_outside_workspace_is_still_rejected(tmp_path):
    capture=tmp_path/'capture.npz'
    np.savez(capture,rgb=np.zeros((5,5,3),dtype=np.uint8),depth=np.ones((5,5)),
             K=np.array([[10,0,2],[0,10,2],[0,0,1]]),camera_position=np.zeros(3),camera_rotation=np.eye(3),
             depth_kind='distance_to_image_plane',camera_convention='ros_optical')
    record={'capture_sha256':hashlib.sha256(capture.read_bytes()).hexdigest(),'load_s':1,'inference_s':2,
            'raw_output':'valid','intent':{'left':[500,500],'right':[500,500]}}
    path=tmp_path/'intent.json';path.write_text(json.dumps(record))
    truth=np.array([[0,0,1.],[0,0,1.]])
    out=inspect_intent(path,capture,truth,np.zeros((2,3)))
    assert not out['accepted'] and 'workspace' in out['rejection']
    assert out['world_error_cm']==[0,0]


def test_one_depth_failure_keeps_other_hand_localization_error(tmp_path):
    capture=tmp_path/'capture.npz';depth=np.ones((9,9));depth[4,6]=np.nan
    np.savez(capture,rgb=np.zeros((9,9,3),dtype=np.uint8),depth=depth,
             K=np.array([[10,0,4],[0,10,4],[0,0,1]]),camera_position=np.zeros(3),camera_rotation=np.eye(3),
             depth_kind='distance_to_image_plane',camera_convention='ros_optical')
    record={'capture_sha256':hashlib.sha256(capture.read_bytes()).hexdigest(),'load_s':1,'inference_s':2,
            'raw_output':'valid','intent':{'left':[250,500],'right':[750,500]}}
    path=tmp_path/'intent.json';path.write_text(json.dumps(record))
    truth=np.array([[-.2,0,1.],[.2,0,1.]])
    out=inspect_intent(path,capture,truth,truth)
    assert not out['accepted']
    assert out['hands'][0]['world_error_cm']==0
    json.dumps(out)
    assert 'right:' in out['rejection'] and 'world' not in out


def test_quality_rejected_candidate_remains_measurable(tmp_path):
    capture=tmp_path/'capture.npz'
    np.savez(capture,rgb=np.zeros((5,5,3),dtype=np.uint8),depth=np.ones((5,5)),
             K=np.array([[10,0,2],[0,10,2],[0,0,1]]),camera_position=np.zeros(3),camera_rotation=np.eye(3),
             depth_kind='distance_to_image_plane',camera_convention='ros_optical')
    record={'capture_sha256':hashlib.sha256(capture.read_bytes()).hexdigest(),'load_s':1,'inference_s':2,
            'raw_output':'candidate','candidate_intent':{'left':[500,500],'right':[500,500]},
            'error':'left: crop estimates disagree'}
    path=tmp_path/'intent.json';path.write_text(json.dumps(record))
    truth=np.array([[0,0,1.],[0,0,1.]])
    out=inspect_intent(path,capture,truth,truth)
    assert not out['accepted'] and out['rejection']==record['error']
    assert out['world_error_cm']==[0,0] and out['valid_protocol']
