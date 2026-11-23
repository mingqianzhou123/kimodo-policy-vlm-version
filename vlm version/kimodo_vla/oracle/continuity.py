"""Explicit reference handoff and bounded hinge projection; no target-aware IK."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def condition_boundary(history_mask, values, mask, current_features, representation):
    """Keep real last-history conditions and condition frame zero on measured pose."""
    history_mask[:-1] = False
    end = representation.slice_dict['global_rot_data'].stop
    values[0, :end] = current_features[:end]
    mask[0, :end] = True


def blend_reference(local, root, boundary_local, boundary_root, elapsed, duration):
    """SO(3)/position blend from the last consumed reference over a fixed interval."""
    if duration <= 0:
        raise ValueError('Handoff duration must be positive')
    u=np.clip(np.asarray(elapsed,dtype=float)/duration,0.,1.)
    w=u**3*(10.-15.*u+6.*u*u)
    r=local.numpy().astype(float); b=boundary_local.numpy().astype(float)
    delta=np.swapaxes(b,-1,-2)[None]@r
    tangent=Rotation.from_matrix(delta.reshape(-1,3,3)).as_rotvec().reshape(*r.shape[:2],3)
    blended=b[None]@Rotation.from_rotvec((tangent*w[:,None,None]).reshape(-1,3)).as_matrix().reshape(r.shape)
    position=boundary_root.numpy()[None]+w[:,None]*(root.numpy()-boundary_root.numpy()[None])
    return torch.as_tensor(blended,dtype=local.dtype),torch.as_tensor(position,dtype=root.dtype)


def bounded_hinge_trajectory(local, codec, limits, previous, max_speed, dt=.02):
    """Closest rotation within joint limits AND the previous command's slew interval.

    Solve each scalar hinge exactly over its allowed interval, considering angle
    branches. A finite joint never jumps through +/-pi by blindly unwrapping.
    This changes reference geometry; callers must measure the resulting FK error.
    """
    if max_speed <= 0 or dt <= 0:raise ValueError('Positive speed and dt required')
    previous=np.asarray(previous,dtype=float).copy();limits=np.asarray(limits,dtype=float)
    if np.any(previous<limits[:,0]-1e-5) or np.any(previous>limits[:,1]+1e-5):
        raise ValueError('Consumed reference is outside joint limits; cannot guarantee a continuous repair')
    previous=np.clip(previous,limits[:,0],limits[:,1])
    rotations=local.numpy().astype(float);out=np.zeros((len(rotations),29))
    for t in range(len(rotations)):
        low=np.maximum(limits[:,0],previous-max_speed*dt)
        high=np.minimum(limits[:,1],previous+max_speed*dt)
        for xi,si in enumerate(codec.xml_to_skeleton.tolist()):
            ci=int(codec.canonical_to_xml[xi]);axis=codec.axes_kimodo[xi].numpy().astype(float)
            axis/=np.linalg.norm(axis)
            r=codec.rot_offsets_f2q[si].numpy()@rotations[t,si]
            skew=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])
            sine=skew@axis;cosine=np.trace(r)-axis@r@axis
            theta=np.arctan2(sine,cosine)
            candidates=np.r_[np.clip(theta+2*np.pi*np.arange(-1,2),low[ci],high[ci]),previous[ci]]
            scores=sine*np.sin(candidates)+cosine*np.cos(candidates)
            ties=np.flatnonzero(scores>=scores.max()-1e-12)
            chosen=ties[np.argmin(np.abs(candidates[ties]-previous[ci]))]
            out[t,ci]=candidates[chosen]
        previous=out[t]
    return out
