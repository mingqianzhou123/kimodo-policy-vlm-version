import json
import numpy as np
import pytest
from kimodo_vla.vlm.refinement import crop_box,crop_to_full,parse_crop_point,combine_and_verify,verify_marker_pixel


def test_crop_mapping_at_clamped_edges_and_center():
    for coarse in ([0,0],[1000,1000],[500,500]):
        box=crop_box(coarse,(640,480),192);x,y,x2,y2=box
        assert 0<=x<x2<=640 and 0<=y<y2<=480
        np.testing.assert_allclose(crop_to_full([0,0],box),[x,y])
        np.testing.assert_allclose(crop_to_full([1000,1000],box),[x2-1,y2-1])
        np.testing.assert_allclose(crop_to_full([500,500],box),[(x+x2-1)/2,(y+y2-1)/2])


@pytest.mark.parametrize('point',[None,[True,20],[float('nan'),20],[-1,500],[1001,500]])
def test_bad_or_missing_crop_points_are_rejected(point):
    with pytest.raises(ValueError):parse_crop_point(json.dumps({'point':point}))


def test_background_wrong_color_and_disagreement_rejected_without_snapping():
    rgb=np.full((40,40,3),100,dtype=np.uint8);rgb[10:20,10:20]=[255,0,0];rgb[25:35,25:35]=[0,0,255]
    for point,side in [([5,5],'left'),([15,15],'right'),([30,30],'left')]:
        with pytest.raises(ValueError):verify_marker_pixel(rgb,point,side)
    with pytest.raises(ValueError,match='disagree'):combine_and_verify(rgb,[[11,11],[19,19]],'left')
    point,check=combine_and_verify(rgb,[[14,14],[16,15]],'left')
    np.testing.assert_array_equal(point,[15,14.5]) # Exact mean of VLM points, not color centroid.
    assert check['disagreement_px']<6


def test_old_ground_mispoint_rejected_by_rgb_check():
    from pathlib import Path
    from kimodo_vla.vlm.geometry import load_capture
    p=Path(__file__).resolve().parents[1]/'runs/vlm-stability-20261117-r1-near-left'
    if not (p/'vlm-capture.npz').exists():pytest.skip('Archived real capture not available')
    d=load_capture(p/'vlm-capture.npz');i=json.loads((p/'vlm-intent.json').read_text())['intent']
    with pytest.raises(ValueError,match='expected colored marker'):
        verify_marker_pixel(d['rgb'],np.array(i['right'])/1000*[639,479],'right')
