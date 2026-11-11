import numpy as np
import pytest
import torch
from kimodo_vla.benchmarks.toolbench.demos import ToolBenchDemo, BODY_JOINT_NAMES
from kimodo_vla.core.types import joint_indices
from kimodo_vla.data.build import training_episode
from kimodo_vla.data.episodes import KimodoWindowDataset, save_episode
from kimodo_vla.model.features import KimodoFeatures
from test_toolbench_demos import recording

@pytest.mark.parametrize('length',[1,2,49,80,151])
def test_terminal_command_pose_padding_and_observed_cut_range(tmp_path,length):
    raw=recording(length)
    raw['action'][-1,15]=.8
    raw['teleop.base_height_command'][-1]=.81
    demo=ToolBenchDemo(raw,'x')
    demo.images=lambda steps:np.zeros((len(steps),1,8,8,3),np.uint8)
    features=KimodoFeatures(50)
    record,frames=training_episode(demo,features,task='G1BallMove-L0-S',episode_id='x')
    n=len(record['target_motion'])
    assert n==round((length-1)*.6)+1
    assert record['target_motion_padded'].shape==(n+49,417)
    assert 'current_observed_motion' not in record
    assert demo.observations()[-1].joint_mask.all()
    np.testing.assert_allclose(demo.observations()[-1].joint_pos[joint_indices(BODY_JOINT_NAMES)],raw['observation.state'][-1,:29])
    full=demo.commanded_motion(include_terminal=True)
    assert len(full.hand)==length+1
    assert full.joint_pos[-1,joint_indices(['left_shoulder_pitch_joint'])[0]]==pytest.approx(.8)
    decoded=features.representation.inverse(record['target_motion_padded'],is_normalized=False,return_numpy=False)
    assert float(decoded['root_positions'][-1,1])==pytest.approx(.81,abs=1e-5)
    velocity=record['target_motion_padded'][-20:,features.representation.slice_dict['velocities']]
    assert velocity.abs().max()<1e-5
    save_episode(tmp_path/'x',record,frames)
    dataset=KimodoWindowDataset([tmp_path],action_history=100,action_chunk=50)
    assert len(dataset)==n
    item=dataset[n-1]
    assert not item['condition_motion_mask'][100:].any()
    assert 'current_hand_state' not in item
    assert item['gt_mask'][100:].all()
    torch.testing.assert_close(item['gt_motion'][100:,3:],record['target_motion_padded'][n-1:n+49,3:])


def test_e20_replanning_images_match_training_timestamps_without_drift():
    from kimodo_vla.model.planner import KimodoPlanner
    planner=object.__new__(KimodoPlanner)
    planner.model_fps,planner.control_fps,planner.execution_frames=30,50,20
    planner.align_execution_grid=True
    step=0;lengths=[]
    for cut in range(0,3000,20):
        assert round(step*.6)==cut
        assert step==round(cut*50/30)
        offsets=planner.control_offsets(step,50)
        absolute=(offsets+cut)/30
        np.testing.assert_allclose(absolute.numpy(),np.arange(step,step+len(offsets))/50,atol=2e-5)
        lengths.append(len(offsets)-1);step+=lengths[-1]
    assert lengths[:6]==[33,34,33,33,34,33]
