# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transformer backbone: padding, masking, and encoder stack for the denoiser."""

import logging
from typing import Mapping, Optional, Union

import torch
from omegaconf import ListConfig
from pydantic.dataclasses import dataclass
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from kimodo_vla.utils.tools import validate

log = logging.getLogger(__name__)


def pad_x_and_mask_to_fixed_size(x: Tensor, mask: Tensor, size: int):
    """Pad a feature vector x and the mask to always have the same size.

    Args:
        x (torch.Tensor): [B, T, D]
        mask (torch.Tensor): [B, T]
        size (int)
    Returns:
        torch.Tensor: [B, size, D]
        torch.Tensor: [B, size]
    """

    batch_size, cur_max_size, dim = x.shape[0], x.shape[1], x.shape[2]

    if cur_max_size == size:
        # already padded to this size, probably in the collate function
        return x, mask

    if cur_max_size > size:
        log.warning("The tensor is larger than the maximum size; cropping from %d to %d", cur_max_size, size)
        return x[:, :size], mask[:, :size]

    new_x = torch.zeros(
        (batch_size, size, dim),
        dtype=x.dtype,
        device=x.device,
    )
    new_x[:, :cur_max_size] = x

    # same for the mask
    new_mask = torch.zeros(
        (batch_size, size),
        dtype=mask.dtype,
        device=mask.device,
    )
    new_mask[:, :cur_max_size] = mask
    return new_x, new_mask


@dataclass(frozen=True, config=dict(extra="forbid", arbitrary_types_allowed=True))
class TransformerEncoderBlockConfig:
    """Configuration for the transformer encoder backbone."""

    # input features dimension
    input_dim: int
    # output features dimension
    output_dim: int

    # skeleton object
    skeleton: object

    # dimension of the text embeddings
    llm_shape: Union[list[int], ListConfig]

    # mask the text or not
    use_text_mask: bool

    # latent dimension of the model
    latent_dim: int
    # dimension of the feedforward network in transformer
    ff_size: int
    # num layers in transformer
    num_layers: int
    # num heads in transformer
    num_heads: int
    # activation in transformer
    activation: str
    # dropout rate for the transformer
    dropout: float
    # dropout rate for the positional embeddings
    pe_dropout: float
    # use norm first or not
    norm_first: bool = False
    # artificially extend the number of text tokens
    num_text_tokens_override: Optional[int] = None

    # Input first heading angle
    input_first_heading_angle: bool = False


class TransformerEncoderBlock(nn.Module):
    @validate(TransformerEncoderBlockConfig, save_args=True, super_init=True)
    def __init__(self, conf):
        self.nbjoints = self.skeleton.nbjoints

        # 定义 positional encoding 层
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.pe_dropout)

         # 定义 time 映射层
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        
        # 定义 motion 映射层
        self.input_linear = nn.Linear(self.input_dim, self.latent_dim)
        self.output_linear = nn.Linear(self.latent_dim, self.output_dim)

        # 定义 first_heading_angle 映射层
        self.linear_first_heading_angle = nn.Linear(2, self.latent_dim)

        # 定义 text 映射层
        llm_dim = self.llm_shape[-1]
        self.embed_text = nn.Linear(llm_dim, self.latent_dim)

        # 定义 text token 的最大数量
        self.num_text_tokens = self.llm_shape[0]
        if self.num_text_tokens_override is not None:
            self.num_text_tokens = self.num_text_tokens_override

        # 定义 Encoder-only Transformer Layer (Self-Attention 看所有 token)
        trans_enc_layer = TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            norm_first=self.norm_first,
        )

        # 定义 Transformer Block ( 多 Transformer Layer 堆叠)
        self.seqTransEncoder = TransformerEncoder(
            trans_enc_layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        x: Tensor,
        x_pad_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_feat_pad_mask: torch.Tensor,
        timesteps: Tensor,
        first_heading_angle: Optional[Tensor] = None,
        hints: Optional[Mapping[int, Tensor]] = None,
        control_visual_tokens: Optional[Mapping[int, Tensor]] = None,
        hint_fuser: Optional[nn.Module] = None,
        future_start: Optional[int] = None,
        return_hidden: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """
        Args:
            x (torch.Tensor): [B, T, dim_motion] current noisy motion
            x_pad_mask (torch.Tensor): [B, T] attention mask, positions with True are allowed to attend, False are not
            text_feat (torch.Tensor): [B, max_text_len, llm_dim] embedded text prompts
            text_feat_pad_mask (torch.Tensor): [B, max_text_len] attention mask, positions with True are allowed to attend, False are not
            timesteps (torch.Tensor): [B,] current denoising step

        Returns:
            torch.Tensor: [B, T, output_dim]
        """
        # 1. 将带噪声的 motion token 化
        batch_size = len(x)
        x = self.input_linear(x)  # [B, T, D]
        
        # 2. 填充 text token & text token 化
        if self.num_text_tokens is not None:
            text_feat, text_feat_pad_mask = pad_x_and_mask_to_fixed_size(
                text_feat,
                text_feat_pad_mask,
                self.num_text_tokens,
            )
        emb_text = self.embed_text(text_feat)  # [B, max_text_len, D]
        
        # 3. time token 化 & 创建 time mask 
        emb_time = self.embed_timestep(timesteps)  # [B, 1, D]
        time_mask = torch.ones((batch_size, 1), dtype=bool, device=x.device)

        # 4. 组合 text+time 的tokne 和 mask -> prefix token 和 prefix attention mask
        prefix_feats = torch.cat((emb_text, emb_time), axis=1)
        if not self.use_text_mask:
            text_feat_pad_mask = torch.ones(
                (batch_size, emb_text.shape[1]),
                dtype=torch.bool,
                device=x.device,
            )
        prefix_mask = torch.cat((text_feat_pad_mask, time_mask), axis=1)

        # 5. 向 prefix token 和 prefix attention mask 添加 first_heading_angle token & mask
        if self.input_first_heading_angle:
            assert first_heading_angle is not None, "The first heading angle is mandatory for this model"
            # cos(angle) / sin(angle)
            first_heading_angle_feats = torch.stack(
                [
                    torch.cos(first_heading_angle),
                    torch.sin(first_heading_angle),
                ],
                axis=-1,
            )

            first_heading_angle_feats = self.linear_first_heading_angle(first_heading_angle_feats)
            first_heading_angle_feats = first_heading_angle_feats[:, None]  # for cat
            first_heading_angle_mask = torch.ones(
                (batch_size, 1),
                dtype=bool,
                device=x.device,
            )
            prefix_feats = torch.cat((prefix_feats, first_heading_angle_feats), axis=1)
            prefix_mask = torch.cat((prefix_mask, first_heading_angle_mask), axis=1)

        # 6. 记录 prefix 的 token 数量
        pose_start_ind = prefix_feats.shape[1]

        # 7. 组织最终的输入 token 和 attention mask
        xseq = torch.cat((prefix_feats, x), axis=1)
        src_key_padding_mask = ~torch.cat((prefix_mask, x_pad_mask), axis=1)

        # 8. 向输入 token 添加位置编码
        xseq = self.sequence_pos_encoder(xseq)
        if isinstance(self.seqTransEncoder, nn.TransformerEncoder):
            assert not self.seqTransEncoder.use_nested_tensor, "Flash attention should be disabled due to bug!"
        
        # 9. 过 transformer block，并在指定层将视觉信息只注入 future motion tokens。
        if hints is not None and control_visual_tokens is not None:
            raise ValueError("Static hints and dynamic control fusion cannot be used together")
        if (control_visual_tokens is None) != (hint_fuser is None):
            raise ValueError(
                "control_visual_tokens and hint_fuser must either both be provided or both be None"
            )
        if control_visual_tokens is not None and future_start is None:
            raise ValueError("future_start is required for dynamic control fusion")

        if hints is not None or control_visual_tokens is not None:
            for i, layer in enumerate(self.seqTransEncoder.layers):
                xseq = layer(xseq, src_key_padding_mask=src_key_padding_mask)
                if hints is not None and i in hints:
                    xseq = torch.cat(
                        (xseq[:, :pose_start_ind], xseq[:, pose_start_ind:] + hints[i]), dim=1
                    )
                if control_visual_tokens is not None and i in control_visual_tokens:
                    motion_tokens = xseq[:, pose_start_ind:]
                    if future_start < 0 or future_start > motion_tokens.shape[1]:
                        raise ValueError(
                            f"Invalid future_start={future_start} for motion length "
                            f"{motion_tokens.shape[1]}"
                        )
                    future_motion_tokens = motion_tokens[:, future_start:]
                    future_hint = hint_fuser(
                        i,
                        control_visual_tokens[i],
                        future_motion_tokens,
                    )
                    if future_hint.shape != future_motion_tokens.shape:
                        raise RuntimeError(
                            "Dynamic future hint must match the future motion hidden shape, "
                            f"got {tuple(future_hint.shape)} and "
                            f"{tuple(future_motion_tokens.shape)}"
                        )
                    motion_tokens = torch.cat(
                        (
                            motion_tokens[:, :future_start],
                            future_motion_tokens + future_hint,
                        ),
                        dim=1,
                    )
                    xseq = torch.cat((xseq[:, :pose_start_ind], motion_tokens), dim=1)
            output = xseq
        else:
            output = self.seqTransEncoder(xseq, src_key_padding_mask=src_key_padding_mask)

        # 10. 提取 motion tokne 输出
        hidden_output = output[:, pose_start_ind:]  # [B, T, D]

        # 11. 输出 motion 标准输出
        output = self.output_linear(hidden_output)  # [B, T, OD]

        if return_hidden:
            return output, hidden_output
        return output


class PositionalEncoding(nn.Module):
    """Non-learned positional encoding."""

    def __init__(
        self,
        d_model: int,
        dropout: Optional[float] = 0.1,
        max_len: Optional[int] = 5000,
    ):
        """
        Args:
            d_model (int): input dim
            dropout (Optional[float] = 0.1): dropout probability on output
            max_len (Optional[int] = 5000): maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Note: have to replace torch.exp() and math.log() with torch.pow()
        # due to MKL exp() and ln() throws floating point exceptions on certain CPUs
        # see corresponding commit and MR
        div_term = torch.pow(10000.0, -torch.arange(0, d_model, 2).float() / d_model)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        # )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, T, D]

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to input sequence.

        Args:
            x (torch.Tensor): [B, T, D] input motion sequence

        Returns:
            torch.Tensor: [B, T, D] input motion with PE added to it (and optionally dropout)
        """
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    """Encoder for diffusion step."""

    def __init__(self, latent_dim: int, sequence_pos_encoder: PositionalEncoding):
        """
        Args:
            latent_dim (int): dim to encode to
            sequence_pos_encoder (PositionalEncoding): the PE to use on timesteps
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed timesteps by adding PE then going through linear layers.

        Args:
            timesteps (torch.Tensor): [B]

        Returns:
            torch.Tensor: [B, 1, D]
        """
        return self.time_embed(self.sequence_pos_encoder.pe.transpose(0, 1)[timesteps])
