"""Generate motion the way NVIDIA's Kimodo does, with our ported modules (no history canvas).

Mirrors github.com/nv-tlabs/kimodo (e1c219c): ``Kimodo._generate`` / ``denoising_step`` (Gaussian
noise for the whole segment, one separated-CFG pass per DDIM step, constraints as observed motion
and a concatenated mask), ``ClassifierFreeGuidedModel`` with ``cfg_type='separated'`` (one batched
pass over: text only, constraints only, neither), the constraint sets ``Root2DConstraintSet``,
``FullBodyConstraintSet`` and ``EndEffectorConstraintSet``, and ``Kimodo._multiprompt`` without
post-processing (a new segment starts from the last few frames of the previous one, constrained in
full and blended linearly). Nothing here imports the official package.

Kimodo-policy's canvas (100 history frames + 50 generated, ``predict_future``) spoils the frozen
model's gait (the walking attribution record, 2026-11-24); this module is the alternative.

Frames are Kimodo's: 30 fps, Y up, heading 0 faces +Z, X to the left. Feature values passed in
and returned are unnormalized; constraint masks are bool.
"""
from __future__ import annotations

import torch

from kimodo_vla.motion.feature_utils import compute_heading_angle
from kimodo_vla.utils.geometry import matrix_to_cont6d
from .native_policy import ConstraintGuidance, SeparatedGuidance
from .sparse_conditions import normalized_future_condition

LLM_DIM = 4096
TRANSITION_FRAMES = 5      # the official multi-prompt default


def _inner_denoiser(model):
    d = model.denoiser
    return d.model if isinstance(d, (ConstraintGuidance, SeparatedGuidance)) else d


def separated_guidance(denoiser, x, pad, text, text_mask, timesteps, heading, motion_mask, observed,
                       text_weight, constraint_weight):
    """The official separated CFG in one batched pass: (text only, constraints only, neither)."""
    out = denoiser(torch.cat([x, x, x]), torch.cat([pad, pad, pad]),
                   torch.cat([text, 0*text, 0*text]),
                   torch.cat([text_mask, torch.zeros_like(text_mask), torch.zeros_like(text_mask)]),
                   torch.cat([timesteps, timesteps, timesteps]),
                   first_heading_angle=torch.cat([heading, heading, heading]),
                   motion_mask=torch.cat([0*motion_mask, motion_mask, 0*motion_mask]),
                   observed_motion=torch.cat([observed, observed, observed]))
    out_text, out_constraint, out_uncond = torch.chunk(out, 3)
    return out_uncond + text_weight*(out_text-out_uncond) + constraint_weight*(out_constraint-out_uncond)


@torch.inference_mode()
def sample_segment(model, values, mask, *, first_heading=0., text_feat=None, text_weight=2., constraint_weight=2.,
                   steps=100, generator=None, noise=None):
    """A segment of ``len(values)`` frames from noise; returns ``(features [T, D] unnormalized, decoded dict of [T, ...])``.

    ``model`` is a ``NativeKimodoPolicy``. ``first_heading`` is the heading angle (radians) the
    motion starts with. Without ``text_feat`` the text token is empty (zero and masked), as the
    official code treats an empty prompt; its guidance term is then exactly zero. ``noise``
    ([T, D] or [1, T, D]) replaces the Gaussian draw, e.g. to repeat an official run.
    """
    rep, device = model.representation, model.device
    denoiser = _inner_denoiser(model)
    dtype = next(denoiser.parameters()).dtype
    frames, dim = values.shape
    observed, motion_mask = normalized_future_condition(rep, values, mask, batch_size=1, frames=frames,
                                                        device=device, dtype=dtype)
    motion_mask = motion_mask.to(dtype)
    if text_feat is None:
        text = torch.zeros(1, 1, LLM_DIM, device=device, dtype=dtype)
        text_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
    else:
        text = torch.as_tensor(text_feat).reshape(1, 1, LLM_DIM).to(device=device, dtype=dtype)
        text_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
    use_timesteps, timestep_map = model.diffusion.space_timesteps(steps)
    model.diffusion.calc_diffusion_vars(use_timesteps)
    if noise is None:
        x = torch.randn((1, frames, dim), device=device, dtype=dtype, generator=generator)
    else:
        x = torch.as_tensor(noise).reshape(1, frames, dim).to(device=device, dtype=dtype)
    pad = torch.ones(1, frames, dtype=torch.bool, device=device)
    heading = torch.tensor([float(first_heading)], device=device, dtype=dtype)
    for i in range(steps-1, -1, -1):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        pred = separated_guidance(denoiser, x, pad, text, text_mask, timestep_map[t], heading, motion_mask, observed,
                                  float(text_weight), float(constraint_weight))
        x = model.sampler(use_timesteps, x, pred, t)
    decoded = rep.inverse(x.float(), is_normalized=True, return_numpy=False)
    decoded = {k: v[0] if torch.is_tensor(v) and v.ndim >= 2 else v for k, v in decoded.items()}
    return rep.unnormalize(x.float())[0], decoded


def _device(rep):
    parents = getattr(rep.skeleton, 'joint_parents', None)
    return parents.device if torch.is_tensor(parents) else torch.device('cpu')


def decode(rep, features):
    """Unnormalized features [T, D] -> the representation's decoded dict of [T, ...] tensors on the CPU."""
    out = rep.inverse(torch.as_tensor(features).float()[None].to(_device(rep)), is_normalized=False, return_numpy=False)
    return {k: v[0].cpu() if torch.is_tensor(v) and v.ndim >= 2 else v for k, v in out.items()}


def empty_condition(rep, frames):
    """Zero values and an all-False mask for ``frames`` frames."""
    values = torch.zeros(frames, rep.motion_rep_dim)
    return values, torch.zeros_like(values, dtype=torch.bool)


def _write(values, mask, rep, name, frames, data, columns=None):
    s = rep.slice_dict[name]
    cols = torch.arange(s.start, s.stop) if columns is None else torch.tensor([s.start+c for c in columns])
    frames = torch.as_tensor(frames, dtype=torch.long)
    values[frames[:, None], cols] = torch.as_tensor(data, dtype=values.dtype).reshape(len(frames), -1)
    mask[frames[:, None], cols] = True


def root_path_condition(rep, values, mask, frames, xz, heading=None):
    """Official ``Root2DConstraintSet``: the smooth root's ground point (x, z) on ``frames``, and the
    global heading (an angle in radians per frame) if given. Writes into ``values`` / ``mask``."""
    xz = torch.as_tensor(xz, dtype=torch.float32).reshape(len(frames), 2)
    _write(values, mask, rep, 'smooth_root_pos', frames, xz, columns=[0, 2])
    if heading is not None:
        angle = torch.as_tensor(heading, dtype=torch.float32).reshape(len(frames))
        _write(values, mask, rep, 'global_root_heading', frames, torch.stack([angle.cos(), angle.sin()], -1))
    return values, mask


def end_effector_rotation_joints(skeleton, names=('left_hand', 'right_hand', 'left_foot', 'right_foot')):
    """Joints whose global rotation an official end-effector constraint fixes (the base of each chain)."""
    return [getattr(skeleton, f'{n}_joint_names')[0] for n in names]


def body_condition(rep, values, mask, frames, posed_joints, global_rot_mats, smooth_root_xz=None, rotation_joints=()):
    """Official ``FullBodyConstraintSet`` on ``frames``: every joint's position, the smooth root (x, z)
    (the pelvis's when not given), the root height and the hip heading; plus, like
    ``EndEffectorConstraintSet``, the global rotation of ``rotation_joints``. Writes into ``values`` / ``mask``."""
    skeleton = rep.skeleton
    posed = torch.as_tensor(posed_joints, dtype=torch.float32)
    rots = torch.as_tensor(global_rot_mats, dtype=torch.float32)
    root_xz = posed[:, skeleton.root_idx][:, [0, 2]] if smooth_root_xz is None else torch.as_tensor(smooth_root_xz).float()
    _write(values, mask, rep, 'smooth_root_pos', frames, root_xz, columns=[0, 2])
    _write(values, mask, rep, 'smooth_root_pos', frames, posed[:, skeleton.root_idx, 1:2], columns=[1])
    angle = compute_heading_angle(posed[None], skeleton)[0]
    _write(values, mask, rep, 'global_root_heading', frames, torch.stack([angle.cos(), angle.sin()], -1))
    local = posed.clone()
    local[..., 0] -= root_xz[:, None, 0]
    local[..., 2] -= root_xz[:, None, 1]
    _write(values, mask, rep, 'local_joints_positions', frames, local.reshape(len(frames), -1))
    s = rep.slice_dict['global_rot_data']
    for name in rotation_joints:
        j = skeleton.bone_order_names.index(name)
        cols = torch.arange(s.start+6*j, s.start+6*j+6)
        f = torch.as_tensor(frames, dtype=torch.long)
        values[f[:, None], cols] = matrix_to_cont6d(rots[:, j])
        mask[f[:, None], cols] = True
    return values, mask


def transition_condition(rep, previous, frames, *, transition=TRANSITION_FRAMES, values=None, mask=None):
    """Conditions of the official multi-prompt continuation, before sampling.

    The last ``transition`` frames of ``previous`` [T, D] (unnormalized features) are constrained in
    full (``FullBodyConstraintSet``) plus the hand and foot rotations (``EndEffectorConstraintSet``),
    followed by ``frames`` new frames with their own ``values`` / ``mask`` (world frame, as
    ``previous``). Everything is moved so the first transition frame's smooth root is at the ground
    origin. Returns ``(values, mask, first_heading, origin_xz)``.
    """
    if not 1 <= transition <= len(previous):
        raise ValueError('need 1 <= transition <= len(previous)')
    latest = torch.as_tensor(previous)[-transition:].float().cpu()
    last = decode(rep, latest)
    root_xz = last['smooth_root_pos'][:, [0, 2]]
    v, m = empty_condition(rep, transition+frames)
    body_condition(rep, v, m, torch.arange(transition), last['posed_joints'], last['global_rot_mats'], root_xz,
                   end_effector_rotation_joints(rep.skeleton))
    if values is not None:
        v[transition:], m[transition:] = torch.as_tensor(values).float().cpu(), torch.as_tensor(mask).cpu()
    origin = root_xz[0].clone()
    s = rep.slice_dict['smooth_root_pos'].start
    v[:, s] -= origin[0]
    v[:, s+2] -= origin[1]
    heading = float(compute_heading_angle(last['posed_joints'][None], rep.skeleton)[0, 0])
    return v.masked_fill(~m, 0), m, heading, origin


def join_segment(rep, previous, generated, origin, *, transition=TRANSITION_FRAMES):
    """Stitch a continuation as the official code does: move ``generated`` [transition+frames, D] back by
    ``origin``, blend its first ``transition`` frames linearly with ``previous``'s last ones (old
    weight 1 -> 0), append the rest. Returns the whole sequence."""
    previous = torch.as_tensor(previous).float().cpu()
    feats = torch.as_tensor(generated).float().cpu().clone()
    s = rep.slice_dict['smooth_root_pos'].start
    feats[:, s] += origin[0]
    feats[:, s+2] += origin[1]
    alpha = torch.linspace(1, 0, transition)[:, None]
    blended = previous[-transition:]*alpha + (1-alpha)*feats[:transition]
    return torch.cat([previous[:-transition], blended, feats[transition:]])


@torch.inference_mode()
def continue_segment(model, previous, frames, *, transition=TRANSITION_FRAMES, values=None, mask=None, **sample_kwargs):
    """The next ``frames`` frames the way the official multi-prompt generation makes them (no
    post-processing): ``transition_condition`` -> ``sample_segment`` -> ``join_segment``. Returns the
    whole stitched sequence [len(previous)+frames, D]; ``sample_kwargs`` go to ``sample_segment``."""
    rep = model.representation
    v, m, heading, origin = transition_condition(rep, previous, frames, transition=transition, values=values, mask=mask)
    feats, _ = sample_segment(model, v, m, first_heading=heading, **sample_kwargs)
    return join_segment(rep, previous, feats, origin, transition=transition)


@torch.inference_mode()
def sample_sequence(model, segments, *, transition=TRANSITION_FRAMES, first_heading=0., noises=None, **sample_kwargs):
    """Official multi-prompt generation: ``segments`` is a list of dicts with ``frames`` and optional
    ``text_feat``, ``values`` / ``mask`` (that segment's constraints, frames counted from its start).
    Returns the stitched unnormalized features [sum(frames), D]."""
    out = None
    for i, seg in enumerate(segments):
        kw = dict(sample_kwargs, text_feat=seg.get('text_feat'))
        if noises is not None:
            kw['noise'] = noises[i]
        if out is None:
            values, mask = seg.get('values'), seg.get('mask')
            if values is None:
                values, mask = empty_condition(model.representation, seg['frames'])
            out, _ = sample_segment(model, values, mask, first_heading=first_heading, **kw)
            out = out.cpu()
        else:
            out = continue_segment(model, out, seg['frames'], transition=transition,
                                   values=seg.get('values'), mask=seg.get('mask'), **kw)
    return out
