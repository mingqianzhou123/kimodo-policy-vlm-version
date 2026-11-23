"""Frozen native Kimodo for experiment A. No DINO, ControlNet or learned hand head."""
from pathlib import Path

import torch
from torch import nn
from omegaconf import OmegaConf

from .kimodo_policy import KimodoPolicy, KimodoPolicyConfig
from .modules.denoiser import TwostageDenoiser
from .modules.diffusion import Diffusion, DDIMSampler
from kimodo_vla.motion.representation.kimodo_motionrep import KimodoMotionRep
from kimodo_vla.skeleton.definitions import G1Skeleton34


class ConstraintGuidance(nn.Module):
    """Constraint-only CFG using Kimodo's documented zero-text convention.

    Equivalent to separated CFG with text weight 0. Keep the same noisy input
    and heading in both passes; only the condition mask differs. No IK/clamping.
    Reference: NVIDIA Kimodo docs, _modules/kimodo/model/cfg.html.
    """
    def __init__(self, model, weight=2.):
        super().__init__()
        self.model, self.weight = model, float(weight)

    def forward(self, **kwargs):
        kwargs = dict(kwargs)
        kwargs['text_feat'] = torch.zeros_like(kwargs['text_feat'])
        kwargs['text_feat_pad_mask'] = torch.zeros_like(kwargs['text_feat_pad_mask'])
        conditional = self.model(**kwargs)
        if self.weight == 1.:
            return conditional
        kwargs['motion_mask'] = torch.zeros_like(kwargs['motion_mask'])
        unconditional = self.model(**kwargs)
        return unconditional + self.weight * (conditional - unconditional)


class NativeKimodoPolicy(KimodoPolicy):
    def __init__(self, checkpoint: str | Path, constraint_cfg=None):
        nn.Module.__init__(self)
        checkpoint = Path(checkpoint).resolve()
        self.config = KimodoPolicyConfig(load_text_encoder=False, enable_hand_head=False)
        # Native CFG needs the same *noisy* input for both branches. Known history
        # is substituted inside the conditional denoiser, never into x itself.
        self.native_noisy_history = True
        self.fps = self.config.fps
        self.g1_skeleton_34 = G1Skeleton34()
        self.representation = KimodoMotionRep(self.g1_skeleton_34, fps=self.fps,
                                              stats_path=str(checkpoint / "stats/motion"))
        self.denoiser = TwostageDenoiser(self.representation, "concat", str(checkpoint))
        if constraint_cfg is not None:
            self.denoiser = ConstraintGuidance(self.denoiser, constraint_cfg)
        self.diffusion = Diffusion(num_base_steps=OmegaConf.load(checkpoint / "config.yaml").num_base_steps)
        self.sampler = DDIMSampler(self.diffusion)
        self.text_encoder = self.controlnet = self.hand_head = None
        self.image_encoder = nn.Identity()  # inherited eval(), never called by native sampling
        self.requires_grad_(False)

    def training_kimodo_policy_controlnet(self, *args, **kwargs):
        raise RuntimeError("Experiment A is inference-only")
