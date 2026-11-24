"""Sample a whole motion segment the way NVIDIA's Kimodo does, with our ported modules.

Mirrors ``Kimodo._generate`` / ``denoising_step`` of github.com/nv-tlabs/kimodo (e1c219c): Gaussian
noise for the whole segment, then per DDIM step one call of the separated-CFG denoiser with the
constraints passed as observed motion and a concatenated mask (no history substituted into the
canvas), then the DDIM update. Constraints are unnormalized Kimodo features with a bool mask,
as ``predict_future``'s ``future_condition`` takes them.

Kimodo-policy's canvas (100 history frames + 50 generated) is what ``predict_future`` does; a
100-frame full-body history with 50 free frames spoils the frozen model's gait (the walking
attribution record, 2026-11-24). This is the alternative: the present enters only as a frame-0
keyframe.
"""
from __future__ import annotations

import torch

from .native_policy import ConstraintGuidance, SeparatedGuidance
from .sparse_conditions import normalized_future_condition

LLM_DIM = 4096


def _inner_denoiser(model):
    d = model.denoiser
    return d.model if isinstance(d, (ConstraintGuidance, SeparatedGuidance)) else d


@torch.inference_mode()
def sample_segment(model, values, mask, *, first_heading=0., text_feat=None, text_weight=2., constraint_weight=2.,
                   steps=100, generator=None):
    """A segment of ``len(values)`` frames; returns ``(features [T, D] unnormalized, decoded dict of [T, ...])``.

    ``model`` is a ``NativeKimodoPolicy``. ``first_heading`` is the heading angle (radians) the
    denoiser is told the motion starts with. Without ``text_feat`` the text branch is off
    (weight 0) and the text token is masked, as Kimodo treats an empty prompt.
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
        text_weight = 0.
    else:
        text = torch.as_tensor(text_feat).reshape(1, 1, LLM_DIM).to(device=device, dtype=dtype)
        text_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
    guided = SeparatedGuidance(denoiser, text_weight, constraint_weight)
    use_timesteps, timestep_map = model.diffusion.space_timesteps(steps)
    model.diffusion.calc_diffusion_vars(use_timesteps)
    x = torch.randn((1, frames, dim), device=device, dtype=dtype, generator=generator)
    pad = torch.ones(1, frames, dtype=torch.bool, device=device)
    heading = torch.tensor([float(first_heading)], device=device, dtype=dtype)
    for i in range(steps-1, -1, -1):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        pred = guided(x=x, x_pad_mask=pad, text_feat=text, text_feat_pad_mask=text_mask, timesteps=timestep_map[t],
                      first_heading_angle=heading, motion_mask=motion_mask, observed_motion=observed)
        x = model.sampler(use_timesteps, x, pred, t)
    decoded = rep.inverse(x.float(), is_normalized=True, return_numpy=False)
    decoded = {k: v[0] if torch.is_tensor(v) and v.ndim >= 2 else v for k, v in decoded.items()}
    return rep.unnormalize(x.float())[0], decoded
