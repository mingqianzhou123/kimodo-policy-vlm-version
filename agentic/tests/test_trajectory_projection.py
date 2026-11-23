import numpy as np
from kimodo_vla.oracle.geometry import XmlFK
from kimodo_vla.oracle.projection import project_trajectory
from kimodo_vla.model.features import G1_XML_PATH
from kimodo_vla.core.types import G1_JOINT_NAMES


def test_projection_corrects_noisy_prior_with_legal_continuous_joints():
    fk=XmlFK(G1_XML_PATH,G1_JOINT_NAMES)
    q=np.clip(np.zeros(29),fk.joint_limits[:,0]+.01,fk.joint_limits[:,1]-.01)
    goal=q.copy()
    for side in ('left','right'):goal[G1_JOINT_NAMES.index(side+'_shoulder_pitch_joint')]-=.15
    root=np.array([0,0,.8]);rotation=np.eye(3)
    hand,hr=fk(goal,root,rotation);feet,fr=fk(q,root,rotation,body_names=['left_ankle_roll_link','right_ankle_roll_link'])
    # Deliberately wrong generation; the solver does not receive goal joint angles.
    prior=np.tile(q,(8,1))+np.random.default_rng(7).normal(0,.15,(8,29))
    out,info=project_trajectory(fk,prior,q,root,rotation,np.tile(hand,(8,1,1)),np.tile(hr,(8,1,1,1)),feet,fr)
    assert np.max(np.abs(np.diff(np.vstack((q,out)),axis=0)))<=4/50+1e-7
    assert np.all(out>=fk.joint_limits[:,0]-1e-7) and np.all(out<=fk.joint_limits[:,1]+1e-7)
    assert max(info['frames'][-1]['wrist_error_m'])<1e-4
    assert max(info['frames'][-1]['foot_error_m'])<1e-4
    assert np.linalg.norm(out[-1]-prior[-1])>.1
    assert all(f['success'] for f in info['frames'])


def fixture_pose():
    fk=XmlFK(G1_XML_PATH,G1_JOINT_NAMES)
    q=np.clip(np.zeros(29),fk.joint_limits[:,0]+.01,fk.joint_limits[:,1]-.01)
    root=np.array([0,0,.8]);rot=np.eye(3)
    hand,hr=fk(q,root,rot)
    feet,fr=fk(q,root,rot,body_names=['left_ankle_roll_link','right_ankle_roll_link'])
    return fk,q,root,rot,hand,hr,feet,fr


def test_hold_does_not_follow_noisy_prior_across_replanning_boundary():
    fk,q,root,rot,hand,hr,feet,fr=fixture_pose()
    rng=np.random.default_rng(22)
    for _ in range(2):
        prior=np.tile(q,(5,1))+rng.normal(0,.3,(5,29))
        out,info=project_trajectory(fk,prior,q,root,rot,np.tile(hand,(5,1,1)),np.tile(hr,(5,1,1,1)),feet,fr,
            stabilize_hold=True,previous_hands=hand,previous_hand_rotations=hr)
        np.testing.assert_array_equal(out,np.tile(q,(5,1)))
        assert all(f['holding_target'] and f['reused_reference'] for f in info['frames'])
        q=out[-1]  # consumed reference, not a generated or measured pose


def test_hold_corrects_unreached_goal_and_releases_for_changed_target():
    fk,q,root,rot,hand,hr,feet,fr=fixture_pose()
    shifted=q.copy()
    for side in ('left','right'):shifted[G1_JOINT_NAMES.index(side+'_shoulder_pitch_joint')]-=.15
    target,tr=fk(shifted,root,rot);prior=np.tile(q,(6,1))
    # Target already stationary, but previous reference does not yet reach it.
    out,info=project_trajectory(fk,prior,q,root,rot,np.tile(target,(6,1,1)),np.tile(tr,(6,1,1,1)),feet,fr,
        stabilize_hold=True,previous_hands=target,previous_hand_rotations=tr)
    assert not info['frames'][0]['reused_reference']
    assert max(info['frames'][-1]['wrist_error_m'])<1e-5
    assert np.linalg.norm(out[-1]-q)>.1
    # A new target must release the hold even when it starts in a later chunk.
    back,second=project_trajectory(fk,prior,out[-1],root,rot,np.tile(hand,(6,1,1)),np.tile(hr,(6,1,1,1)),feet,fr,
        stabilize_hold=True,previous_hands=target,previous_hand_rotations=tr)
    assert not second['frames'][0]['holding_target']
    assert max(second['frames'][-1]['wrist_error_m'])<1e-5
    assert np.max(abs(np.diff(np.vstack((out[-1],back)),axis=0)))<=.08+1e-7


def test_hold_survives_float32_fk_roundoff_but_not_final_reach_step():
    from scipy.spatial.transform import Rotation
    fk,q,root,rot,hand,hr,feet,fr=fixture_pose()
    previous_hand=hand+np.array([2e-7,-1e-7,0.])
    previous_rot=hr@Rotation.from_rotvec([0.,2e-7,0.]).as_matrix()
    out,info=project_trajectory(fk,np.array([q+.1]),q,root,rot,hand[None],hr[None],feet,fr,
        stabilize_hold=True,previous_hands=previous_hand,previous_hand_rotations=previous_rot)
    assert info['frames'][0]['reused_reference']
    np.testing.assert_array_equal(out[0],q)
    # The last 20 ms of the actual 1.2 s reach is small, but still motion.
    from kimodo_vla.oracle.hand_path import joint_goal_path
    goal=q.copy()
    for side in ('left','right'):goal[G1_JOINT_NAMES.index(side+'_shoulder_pitch_joint')]-=.15
    before=joint_goal_path(q,goal,np.array([1.18]))[0]
    ph,pr=fk(before,root,rot);end,er=fk(goal,root,rot)
    _,info=project_trajectory(fk,goal[None],before,root,rot,end[None],er[None],feet,fr,
        stabilize_hold=True,previous_hands=ph,previous_hand_rotations=pr)
    assert not info['frames'][0]['holding_target']
