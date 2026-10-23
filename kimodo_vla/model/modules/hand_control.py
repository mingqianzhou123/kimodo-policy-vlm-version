"""Detached hand diffusion head conditioned on visual and body-motion tokens."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .backbone import PositionalEncoding, TimestepEmbedder


class HandTokenDiffusionDenoiser(nn.Module):
    """Predict clean future binary states or continuous closure residuals."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        future_token_count: int = 50,
        image_token_count: int = 196,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        max_diffusion_steps: int = 1000,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if future_token_count <= 0 or image_token_count <= 0:
            raise ValueError("future_token_count and image_token_count must be positive")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.future_token_count = int(future_token_count)
        self.image_token_count = int(image_token_count)
        self.sequence_token_count = 1 + image_token_count + future_token_count + (
            future_token_count + 1
        )

        self.vision_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        self.body_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        self.hand_projection = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, hidden_dim),
        )

        timestep_encoding = PositionalEncoding(
            hidden_dim,
            dropout=0.0,
            max_len=max_diffusion_steps,
        )
        self.timestep_embedding = TimestepEmbedder(hidden_dim, timestep_encoding)
        self.modality_embedding = nn.Parameter(torch.zeros(4, hidden_dim))
        self.body_position_embedding = nn.Parameter(
            torch.zeros(1, future_token_count, hidden_dim)
        )
        self.hand_position_embedding = nn.Parameter(
            torch.zeros(1, future_token_count + 1, hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2),
        )

        nn.init.normal_(self.modality_embedding, std=0.02)
        nn.init.normal_(self.body_position_embedding, std=0.02)
        nn.init.normal_(self.hand_position_embedding, std=0.02)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        body_future_tokens: torch.Tensor,
        noisy_future_hand: torch.Tensor,
        current_hand_state: torch.Tensor,
        timesteps: torch.Tensor,
        future_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = visual_tokens.shape[0]
        expected_shapes = {
            "visual_tokens": (
                batch_size,
                self.image_token_count,
                self.input_dim,
            ),
            "body_future_tokens": (
                batch_size,
                self.future_token_count,
                self.input_dim,
            ),
            "noisy_future_hand": (
                batch_size,
                self.future_token_count,
                2,
            ),
        }
        actual_tensors = {
            "visual_tokens": visual_tokens,
            "body_future_tokens": body_future_tokens,
            "noisy_future_hand": noisy_future_hand,
        }
        for name, expected_shape in expected_shapes.items():
            if tuple(actual_tensors[name].shape) != expected_shape:
                raise ValueError(
                    f"Expected {name} shape {expected_shape}, got "
                    f"{tuple(actual_tensors[name].shape)}"
                )
        if current_hand_state.ndim == 3 and current_hand_state.shape[1] == 1:
            current_hand_state = current_hand_state[:, 0]
        if tuple(current_hand_state.shape) != (batch_size, 2):
            raise ValueError(
                f"Expected current_hand_state shape {(batch_size, 2)}, got "
                f"{tuple(current_hand_state.shape)}"
            )
        if tuple(timesteps.shape) != (batch_size,):
            raise ValueError(
                f"Expected timesteps shape {(batch_size,)}, got {tuple(timesteps.shape)}"
            )

        if future_mask is None:
            future_mask = torch.ones(
                batch_size,
                self.future_token_count,
                dtype=torch.bool,
                device=visual_tokens.device,
            )
        else:
            future_mask = torch.as_tensor(
                future_mask, device=visual_tokens.device, dtype=torch.bool
            )
            if tuple(future_mask.shape) != (
                batch_size,
                self.future_token_count,
            ):
                raise ValueError(
                    "Expected future_mask shape "
                    f"{(batch_size, self.future_token_count)}, got "
                    f"{tuple(future_mask.shape)}"
                )

        # This head is intentionally isolated from ControlNet and the body denoiser.
        visual_tokens = visual_tokens.detach()
        body_future_tokens = body_future_tokens.detach()
        current_hand_state = current_hand_state.to(dtype=noisy_future_hand.dtype)

        time_tokens = self.timestep_embedding(timesteps) + self.modality_embedding[0]
        vision_tokens = self.vision_projection(visual_tokens) + self.modality_embedding[1]
        body_tokens = (
            self.body_projection(body_future_tokens)
            + self.body_position_embedding
            + self.modality_embedding[2]
        )
        hand_input = torch.cat(
            (current_hand_state.unsqueeze(1), noisy_future_hand), dim=1
        )
        hand_tokens = (
            self.hand_projection(hand_input)
            + self.hand_position_embedding
            + self.modality_embedding[3]
        )

        sequence = torch.cat(
            (time_tokens, vision_tokens, body_tokens, hand_tokens), dim=1
        )
        if sequence.shape[1] != self.sequence_token_count:
            raise RuntimeError(
                f"Expected {self.sequence_token_count} hand-head tokens, got "
                f"{sequence.shape[1]}"
            )

        prefix_mask = torch.ones(
            batch_size,
            1 + self.image_token_count,
            dtype=torch.bool,
            device=sequence.device,
        )
        hand_mask = torch.cat(
            (
                torch.ones(batch_size, 1, dtype=torch.bool, device=sequence.device),
                future_mask,
            ),
            dim=1,
        )
        valid_mask = torch.cat(
            (prefix_mask, future_mask, hand_mask), dim=1
        )
        encoded = self.encoder(sequence, src_key_padding_mask=~valid_mask)

        # The first hand token is current state; only the following 50 are predictions.
        future_hand_tokens = encoded[:, -self.future_token_count :]
        return self.output(future_hand_tokens)
