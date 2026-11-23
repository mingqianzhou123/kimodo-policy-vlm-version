import numpy as np
from kimodo_vla.oracle.hand_path import joint_goal_path,execution_frames
from kimodo_vla.oracle.policy import OraclePolicy
from kimodo_vla.oracle.geometry import Frame,rotation_wxyz,BASIS
from kimodo_vla.model.features import KimodoFeatures
import json
from pathlib import Path


def test_clock_path_and_execution_coverage():
    q=np.zeros(29);goal=q.copy();goal[0]=-.15
    path=joint_goal_path(q,goal,np.array([0,.6,1.2,4]))
    np.testing.assert_allclose(path[:,0],[0,-.075,-.15,-.15])
    for step in range(5,400,25):
        assert execution_frames(step,round(step*.6))==list(range(1,16))
    assert execution_frames(6,3)==list(range(1,17))


def test_feasible_path_endpoint_matches_saved_oracle_and_is_absolute_clock():
    run=Path('runs/support-root-reach-20261117-r1')
    if not run.exists():return  # Remote source-only deployment has the run too.
    target=json.loads((run/'target.json').read_text());sample=json.loads((run/'execution.jsonl').read_text().splitlines()[0])['before']
    p=OraclePolicy.__new__(OraclePolicy);p.features=KimodoFeatures(50)
    p.frame=Frame(target['origin'],np.array(target['heading']));p.initial_root=np.array(sample['root_pos'])
    p.initial_rotation=rotation_wxyz(sample['root_quat']);p.initial_joints=np.array(sample['joints'])
    p.goal_joints=np.array(target['witness_joint_pos'])
    names=p.features.representation.skeleton.bone_order_names
    p.ids=[[names.index(f'{s}_{j}_skel') for j in ('wrist_yaw','hand_roll')] for s in ('left','right')]
    pos,rot=p.path_goals(np.array([0,.6,1.2,4.]))
    np.testing.assert_allclose(p.frame.world(pos[2:]),np.tile(target['targets_world'],(2,1,1,1)),atol=2e-6)
    assert np.max(abs(pos[1]-pos[0]))>0.01
    np.testing.assert_allclose(rot@np.swapaxes(rot,-1,-2),np.broadcast_to(np.eye(3),rot.shape),atol=2e-6)
    world_rot=p.frame.world_to_model.T@rot@BASIS
    np.testing.assert_allclose(world_rot[0],sample['wrist_rot'],atol=3e-6)
    np.testing.assert_allclose(world_rot[2],target['wrist_rotation_world'],atol=3e-6)
