# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-stage transformer denoiser: root stage then body stage for motion diffusion."""

import os
import torch
from torch import nn
import contextlib
from typing import Mapping, Optional
from omegaconf import OmegaConf

from .backbone import TransformerEncoderBlock
from ..utils.loading import load_checkpoint_state_dict


class TwostageDenoiser(nn.Module):
    """Two-stage denoiser: first predicts global root features, then body features conditioned on local root."""

    def __init__(
        self,
        motion_rep,
        motion_mask_mode,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        # 1. 加载配置
        cfg = OmegaConf.load(os.path.join(ckpt_path, "config.yaml")).denoiser
        self.motion_rep = motion_rep
        self.motion_mask_mode = motion_mask_mode
        input_dim = motion_rep.motion_rep_dim
        will_concatenate = motion_mask_mode == "concat"
        # 2. 加载根节点去噪网络
        root_input_dim = input_dim * 2 if will_concatenate else input_dim
        root_output_dim = motion_rep.global_root_dim
        self.root_model = TransformerEncoderBlock(
            input_dim=root_input_dim,
            output_dim=root_output_dim,
            skeleton=self.motion_rep.skeleton,
            llm_shape=cfg.llm_shape,
            use_text_mask=cfg.use_text_mask,
            latent_dim=cfg.latent_dim,
            ff_size=cfg.ff_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            activation=cfg.activation,
            dropout=cfg.dropout,
            pe_dropout=cfg.pe_dropout,
            norm_first=cfg.norm_first,
            num_text_tokens_override=cfg.num_text_tokens_override,
            input_first_heading_angle=cfg.input_first_heading_angle
        )
        # 3. 加载身体节点去噪网络
        # global_root_dim: (全局x, 全局y, 全局z, cos θ, sin θ)
        # local_root_dim: (旋转角速度, 平移x, 平移z, 全局高度Y)
        local_motion_rep_dim = input_dim - motion_rep.global_root_dim + motion_rep.local_root_dim
        body_input_dim = local_motion_rep_dim + (
            input_dim if will_concatenate else 0 )  # body stage always takes in local root info for motion (but still the global mask)
        body_output_dim = input_dim - motion_rep.global_root_dim
        self.body_model = TransformerEncoderBlock(
            input_dim=body_input_dim,
            output_dim=body_output_dim,
            skeleton=self.motion_rep.skeleton,
            llm_shape=cfg.llm_shape,
            use_text_mask=cfg.use_text_mask,
            latent_dim=cfg.latent_dim,
            ff_size=cfg.ff_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            activation=cfg.activation,
            dropout=cfg.dropout,
            pe_dropout=cfg.pe_dropout,
            norm_first=cfg.norm_first,
            num_text_tokens_override=cfg.num_text_tokens_override,
            input_first_heading_angle=cfg.input_first_heading_angle
        )   
        # 4. 加载kimodo预训练权重
        if ckpt_path:
            self._load_ckpt(ckpt_path)


    def _load_ckpt(self, ckpt_path: str) -> None:
        """Load checkpoint from path; state dict keys are stripped of 'denoiser.backbone.'
        prefix."""
        state_dict = load_checkpoint_state_dict(ckpt_path)
        state_dict = {key.replace("denoiser.backbone.", ""): val for key, val in state_dict.items()}
        self.load_state_dict(state_dict)


    def forward(
        self,
        x: torch.Tensor,
        x_pad_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_feat_pad_mask: torch.Tensor,
        timesteps: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor] = None,
        motion_mask: Optional[torch.Tensor] = None,
        observed_motion: Optional[torch.Tensor] = None,
        root_hints: Optional[list] = None,
        body_hints: Optional[list] = None,
        root_control_tokens: Optional[Mapping[int, torch.Tensor]] = None,
        body_control_tokens: Optional[Mapping[int, torch.Tensor]] = None,
        root_hint_fuser: Optional[nn.Module] = None,
        body_hint_fuser: Optional[nn.Module] = None,
        control_future_start: Optional[int] = None,
        detach_root_for_body: Optional[bool] = None,
        return_body_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): [B, T, dim_motion] current noisy motion
            x_pad_mask (torch.Tensor): [B, T] attention mask, positions with True are allowed to attend, False are not
            text_feat (torch.Tensor): [B, max_text_len, llm_dim] embedded text prompts
            text_feat_pad_mask (torch.Tensor): [B, max_text_len] attention mask, positions with True are allowed to attend, False are not
            timesteps (torch.Tensor): [B,] current denoising step
            motion_mask
            observed_motion
            detach_root_for_body: Whether Body loss should stop at the predicted
                Root motion instead of updating the Root stage through the local
                Root conversion.

        Returns:
            torch.Tensor: same size as input x
        """

        if self.motion_mask_mode == "concat":
            if motion_mask is None or observed_motion is None:
                motion_mask = torch.zeros_like(x)
                observed_motion = torch.zeros_like(x)
            x = x * (1 - motion_mask) + observed_motion * motion_mask
            x_extended = torch.cat([x, motion_mask], axis=-1)
        else:
            x_extended = x

        # Stage 1: predict root motion in global
        root_motion_pred = self.root_model(
            x_extended,
            x_pad_mask,
            text_feat,
            text_feat_pad_mask,
            timesteps,
            first_heading_angle,
            hints=root_hints,
            control_visual_tokens=root_control_tokens,
            hint_fuser=root_hint_fuser,
            future_start=control_future_start,
        )  # [B, T, 5]

        if detach_root_for_body is None:
            detach_root_for_body = self.training
        convert_ctx = torch.no_grad() if detach_root_for_body else contextlib.nullcontext()
        with convert_ctx:
            root_motion_local = self.motion_rep.global_root_to_local_root(
                root_motion_pred,
                normalized=True,
                lengths=None,
                valid_mask=x_pad_mask,
            )
        if detach_root_for_body:
            root_motion_local = root_motion_local.detach()

        # concatenate the predicted local root with the body motion
        body_x = x[..., self.motion_rep.body_slice]
        x_new = torch.cat([root_motion_local, body_x], axis=-1)

        if self.motion_mask_mode == "concat":
            x_new_extended = torch.cat([x_new, motion_mask], axis=-1)
        else:
            x_new_extended = x_new

        # Stage 2: predict local body motion based on local root
        body_output = self.body_model(
            x_new_extended,
            x_pad_mask,
            text_feat,
            text_feat_pad_mask,
            timesteps,
            first_heading_angle,
            hints=body_hints,
            control_visual_tokens=body_control_tokens,
            hint_fuser=body_hint_fuser,
            future_start=control_future_start,
            return_hidden=return_body_hidden,
        )
        if return_body_hidden:
            predicted_body, body_hidden = body_output
        else:
            predicted_body = body_output

        # concatenate the predicted local body with the predicted root
        output = torch.cat([root_motion_pred, predicted_body], axis=-1)
        if return_body_hidden:
            return output, body_hidden
        return output
