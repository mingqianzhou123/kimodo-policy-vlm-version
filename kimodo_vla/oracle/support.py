"""Standing support conditions in native Kimodo features, never output clamping."""
import torch


def support_channels(representation, mode):
    if mode not in ('none', 'root', 'root-feet', 'root-feet-keyframes'):
        raise ValueError('Unknown support condition mode')
    mask = torch.zeros(representation.motion_rep_dim, dtype=torch.bool)
    if mode == 'none':
        return mask
    for name in ('smooth_root_pos', 'global_root_heading'):
        mask[representation.slice_dict[name]] = True
    skeleton = representation.skeleton
    joints = [skeleton.root_idx]
    if mode.startswith('root-feet'):
        joints += list(skeleton.left_foot_joint_idx[:2]) + list(skeleton.right_foot_joint_idx[:2])
        mask[representation.slice_dict['foot_contacts']] = True
    for name, width in (('local_joints_positions', 3), ('global_rot_data', 6), ('velocities', 3)):
        for joint in joints:
            start = representation.slice_dict[name].start + width * joint
            mask[start:start + width] = True
    return mask


def condition_support(values, mask, standing_features, representation, mode, keyframes=None):
    """Hold initial root at frames 1..T-1; feet are dense or at hand keyframes.

    Frame zero remains the CURRENT measured boundary. Standing features come
    from a repeated initial pose, including zero velocities and native heuristic
    contact labels. Contacts here are desired labels, not measured forces.
    """
    channels = support_channels(representation, mode)
    if mode == 'root-feet-keyframes':
        root_channels=condition_support(values,mask,standing_features,representation,'root')
        foot_channels=channels & ~root_channels
        if not keyframes or any(k<0 or k>=len(values) for k in keyframes):
            raise ValueError('Valid hand keyframes required for sparse support')
        for k in keyframes:
            if k==0:continue
            values[k,foot_channels]=standing_features[foot_channels]
            mask[k,foot_channels]=True
        return channels
    values[1:, channels] = standing_features[channels]
    mask[1:, channels] = True
    return channels
