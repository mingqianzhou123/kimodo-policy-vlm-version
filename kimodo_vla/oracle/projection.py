"""Bounded per-frame IK projection of Kimodo joints onto standing hand/foot goals.

The generated joints remain the posture prior. No reach witness is accepted by
this solver. This is a kinematic projection, not a collision/dynamics solver.
"""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def project_trajectory(fk, prior, previous, root, root_rotation, hands, hand_rotations,
                       feet, foot_rotations, speed=4., rate=50., stabilize_hold=False,
                       previous_hands=None, previous_hand_rotations=None):
    names=['left_wrist_yaw_link','right_wrist_yaw_link','left_ankle_roll_link','right_ankle_roll_link']
    output=[];diagnostics=[];previous=np.asarray(previous,dtype=float).copy()
    if stabilize_hold and (previous_hands is None or previous_hand_rotations is None):
        raise ValueError('Holding detection requires the preceding hand target, not measured hand drift')
    start=time.perf_counter()
    for candidate,hand,hand_rot in zip(prior,hands,hand_rotations):
        target=np.concatenate((hand,feet));target_rot=np.concatenate((hand_rot,foot_rotations))
        # Single-pose and batched float32 FK can differ by ~1e-7 despite the
        # same goal. Do not release a hold on that numerical difference.
        holding=bool(stabilize_hold
            and np.max(np.linalg.norm(hand-previous_hands,axis=-1))<=1e-6
            and np.max(np.linalg.norm(Rotation.from_matrix(
                np.swapaxes(previous_hand_rotations,-1,-2)@hand_rot).as_rotvec(),axis=-1))<=1e-6)
        # A changing model prior must not move redundant joints at a fixed goal.
        # Always start from the last CONSUMED reference at chunk boundaries.
        posture_prior=previous if holding else candidate
        lo=np.maximum(fk.joint_limits[:,0],previous-speed/rate)
        hi=np.minimum(fk.joint_limits[:,1],previous+speed/rate)
        if np.any(lo>=hi):raise ValueError('Projection has no legal slew interval')
        def residual(q):
            pos,rot=fk(q,root,root_rotation,body_names=names)
            angle=Rotation.from_matrix(np.swapaxes(target_rot,-1,-2)@rot).as_rotvec()
            return np.r_[100*(pos-target).ravel(),20*angle.ravel(),.1*(q-posture_prior),.1*(q-previous)]
        reuse=False
        if holding:
            prev_pos,prev_rot=fk(previous,root,root_rotation,body_names=names)
            pos_error=np.linalg.norm(prev_pos-target,axis=-1)
            rot_error=np.linalg.norm(Rotation.from_matrix(np.swapaxes(target_rot,-1,-2)@prev_rot).as_rotvec(),axis=-1)
            reuse=bool(np.all(pos_error<=1e-5) and np.all(rot_error<=1e-4)
                       and np.all(previous>=lo) and np.all(previous<=hi))
        if reuse:
            q=previous.copy();success=True;nfev=0
        else:
            fit=least_squares(residual,np.clip(previous,lo+1e-10,hi-1e-10),bounds=(lo,hi),
                              max_nfev=60,ftol=1e-8,xtol=1e-8,gtol=1e-7)
            q=fit.x;success=bool(fit.success);nfev=fit.nfev
        pos,rot=fk(q,root,root_rotation,body_names=names)
        error=np.linalg.norm(pos-target,axis=-1)
        diagnostics.append({'success':success,'nfev':nfev,'holding_target':holding,'reused_reference':reuse,
            'wrist_error_m':error[:2].tolist(),'foot_error_m':error[2:].tolist(),
            'rotation_error_deg':np.rad2deg(np.linalg.norm(Rotation.from_matrix(np.swapaxes(target_rot,-1,-2)@rot).as_rotvec(),axis=-1)).tolist(),
            'max_joint_step_rad':float(abs(q-previous).max()),
            'prior_joint_rmse_rad':float(np.sqrt(np.mean((q-candidate)**2)))})
        output.append(q);previous=q
        previous_hands=hand;previous_hand_rotations=hand_rot
    return np.array(output),{'elapsed_s':time.perf_counter()-start,'frames':diagnostics,
        'root_mode':'initial measured root fixed',
        'prior':'consumed reference at fixed goals; Kimodo while moving' if stabilize_hold else 'bounded handoff Kimodo joints',
        'collision_checked':False,'dynamics_checked':False}
