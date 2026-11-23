import copy

import torch
from torch import nn

from .backbone import TimestepEmbedder


_VALID_CONTROL_FUSION_MODES = frozenset({"both", "cross_attn", "mlp"})


def _normalize_control_fusion_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in _VALID_CONTROL_FUSION_MODES:
        valid_modes = ", ".join(sorted(_VALID_CONTROL_FUSION_MODES))
        raise ValueError(
            f"control_fusion_mode must be one of {{{valid_modes}}}, got {mode!r}"
        )
    return normalized


def _zero_linear(dim: int) -> nn.Linear:
    layer = nn.Linear(dim, dim)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class _SharedCrossAttention(nn.Module):
    """Read visual tokens using the current future-motion hidden tokens as queries."""

    def __init__(self, latent_dim: int, num_heads: int):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError(
                f"latent_dim={latent_dim} must be divisible by num_heads={num_heads}"
            )
        self.latent_dim = int(latent_dim)
        self.query_norm = nn.LayerNorm(latent_dim)
        self.context_norm = nn.LayerNorm(latent_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(
        self,
        future_motion_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if future_motion_tokens.ndim != 3 or visual_tokens.ndim != 3:
            raise ValueError(
                "Expected future motion and visual tokens to be rank-3 tensors, got "
                f"{tuple(future_motion_tokens.shape)} and {tuple(visual_tokens.shape)}"
            )
        if future_motion_tokens.shape[0] != visual_tokens.shape[0]:
            raise ValueError(
                "Future motion and visual tokens must share the batch dimension, got "
                f"{future_motion_tokens.shape[0]} and {visual_tokens.shape[0]}"
            )
        if (
            future_motion_tokens.shape[-1] != self.latent_dim
            or visual_tokens.shape[-1] != self.latent_dim
        ):
            raise ValueError(
                f"Expected latent dim {self.latent_dim}, got "
                f"{future_motion_tokens.shape[-1]} and {visual_tokens.shape[-1]}"
            )

        query = self.query_norm(future_motion_tokens)
        context = self.context_norm(visual_tokens)
        attended, _ = self.attention(
            query=query,
            key=context,
            value=context,
            need_weights=False,
        )
        return attended


class _VisualTokenMLP(nn.Module):
    """Compress spatial visual tokens into future-position visual tokens."""

    def __init__(
        self,
        image_token_count: int,
        future_token_count: int,
        hidden_token_dims: tuple[int, ...],
    ):
        super().__init__()
        token_dims = (
            int(image_token_count),
            *(int(dim) for dim in hidden_token_dims),
            int(future_token_count),
        )
        if any(dim <= 0 for dim in token_dims):
            raise ValueError(f"All visual token MLP dimensions must be positive, got {token_dims}")

        layers = []
        for layer_index, (input_dim, output_dim) in enumerate(
            zip(token_dims[:-1], token_dims[1:])
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if layer_index < len(token_dims) - 2:
                layers.append(nn.GELU())
        self.network = nn.Sequential(*layers)
        self.image_token_count = token_dims[0]
        self.future_token_count = token_dims[-1]

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        if visual_tokens.ndim != 3:
            raise ValueError(
                "Expected visual tokens [B, image_tokens, latent_dim], got "
                f"{tuple(visual_tokens.shape)}"
            )
        if visual_tokens.shape[1] != self.image_token_count:
            raise ValueError(
                f"Expected {self.image_token_count} visual tokens, "
                f"got {visual_tokens.shape[1]}"
            )

        # Apply the MLP over the token axis while preserving every latent channel.
        future_tokens = self.network(visual_tokens.transpose(1, 2)).transpose(1, 2)
        if future_tokens.shape[1] != self.future_token_count:
            raise RuntimeError(
                f"Visual token MLP produced {future_tokens.shape[1]} tokens, "
                f"expected {self.future_token_count}"
            )
        return future_tokens


class _LayerHintProjection(nn.Module):
    """Fuse configured visual paths and produce a zero-initialized hint."""

    def __init__(
        self,
        image_token_count: int,
        future_token_count: int,
        latent_dim: int,
        token_mlp_hidden_dims: tuple[int, ...],
        control_fusion_mode: str = "both",
    ):
        super().__init__()
        self.control_fusion_mode = _normalize_control_fusion_mode(control_fusion_mode)
        self.future_token_count = int(future_token_count)
        self.latent_dim = int(latent_dim)
        if self.control_fusion_mode in {"both", "mlp"}:
            # Keep the original module name and construction order in the
            # default ``both`` mode so existing checkpoints are unchanged.
            self.visual_token_mlp = _VisualTokenMLP(
                image_token_count=image_token_count,
                future_token_count=self.future_token_count,
                hidden_token_dims=token_mlp_hidden_dims,
            )
        self.fusion_zero_projection = _zero_linear(latent_dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        future_motion_tokens: torch.Tensor,
        shared_cross_attention: _SharedCrossAttention | None,
    ) -> torch.Tensor:
        if future_motion_tokens.ndim != 3:
            raise ValueError(
                "Expected future motion tokens [B, future_tokens, latent_dim], got "
                f"{tuple(future_motion_tokens.shape)}"
            )
        if future_motion_tokens.shape[1:] != (
            self.future_token_count,
            self.latent_dim,
        ):
            raise ValueError(
                "Expected future motion token shape "
                f"[B, {self.future_token_count}, {self.latent_dim}], got "
                f"{tuple(future_motion_tokens.shape)}"
            )

        if self.control_fusion_mode == "both":
            # Keep the original dual-path operation order intact.
            mlp_future_tokens = self.visual_token_mlp(visual_tokens)
            if shared_cross_attention is None:
                raise RuntimeError(
                    "Cross-Attention is required for control_fusion_mode='both'"
                )
            attended_visual_tokens = shared_cross_attention(
                future_motion_tokens,
                visual_tokens,
            )
            if mlp_future_tokens.shape != attended_visual_tokens.shape:
                raise RuntimeError(
                    "Visual MLP and shared cross-attention outputs must have identical shapes, "
                    f"got {tuple(mlp_future_tokens.shape)} and "
                    f"{tuple(attended_visual_tokens.shape)}"
                )
            fused_future_tokens = mlp_future_tokens + attended_visual_tokens
        elif self.control_fusion_mode == "mlp":
            fused_future_tokens = self.visual_token_mlp(visual_tokens)
        else:
            if shared_cross_attention is None:
                raise RuntimeError(
                    "Cross-Attention is required for control_fusion_mode='cross_attn'"
                )
            fused_future_tokens = shared_cross_attention(
                future_motion_tokens,
                visual_tokens,
            )
        return self.fusion_zero_projection(fused_future_tokens)


class _StageHintFusion(nn.Module):
    """Fuse per-layer visual tokens with the configured visual paths."""

    def __init__(
        self,
        injection_layers: tuple[int, ...],
        image_token_count: int,
        future_token_count: int,
        latent_dim: int,
        num_heads: int,
        token_mlp_hidden_dims: tuple[int, ...],
        control_fusion_mode: str = "both",
    ):
        super().__init__()
        self.control_fusion_mode = _normalize_control_fusion_mode(control_fusion_mode)
        self.injection_layers = tuple(int(layer) for layer in injection_layers)
        if self.control_fusion_mode in {"both", "cross_attn"}:
            # In ``both`` this is intentionally constructed exactly as before;
            # in ``mlp`` it is absent from the module/state-dict entirely.
            self.shared_cross_attention = _SharedCrossAttention(
                latent_dim=latent_dim,
                num_heads=num_heads,
            )
        self.projections = nn.ModuleList(
            _LayerHintProjection(
                image_token_count=image_token_count,
                future_token_count=future_token_count,
                latent_dim=latent_dim,
                token_mlp_hidden_dims=token_mlp_hidden_dims,
                control_fusion_mode=self.control_fusion_mode,
            )
            for _ in self.injection_layers
        )

    def forward(
        self,
        layer_index: int,
        visual_tokens: torch.Tensor,
        future_motion_tokens: torch.Tensor,
    ) -> torch.Tensor:
        try:
            projection_index = self.injection_layers.index(int(layer_index))
        except ValueError as error:
            raise ValueError(
                f"Layer {layer_index} is not a configured injection layer "
                f"{self.injection_layers}"
            ) from error
        if self.control_fusion_mode == "mlp":
            shared_cross_attention = None
        else:
            shared_cross_attention = self.shared_cross_attention
        return self.projections[projection_index](
            visual_tokens,
            future_motion_tokens,
            shared_cross_attention,
        )


class ControlNet(nn.Module):
    """Sequential sparse visual ControlNet for the frozen two-stage Kimodo denoiser."""

    def __init__(
        self,
        denoiser,
        image_feat_dim: int,
        image_token_count: int = 196,
        motion_token_count: int = 150,
        num_control_layers: int = 8,
        future_token_count: int | None = None,
        token_mlp_hidden_dims: tuple[int, ...] = (256, 128),
        detach_root_control_for_body: bool = False,
        control_fusion_mode: str = "both",
        controlnet_scale: int = 1,
    ):
        super().__init__()
        self.control_fusion_mode = _normalize_control_fusion_mode(control_fusion_mode)
        root_model = denoiser.root_model
        body_model = denoiser.body_model
        if root_model.latent_dim != body_model.latent_dim:
            raise ValueError("Root and body ControlNet stages require the same latent dimension")

        latent_dim = root_model.latent_dim
        self.image_token_count = int(image_token_count)
        self.motion_token_count = int(motion_token_count)
        self.num_control_layers = int(num_control_layers)
        self.controlnet_scale = int(controlnet_scale)
        if self.controlnet_scale <= 0:
            raise ValueError(
                f"controlnet_scale must be a positive integer, got {controlnet_scale}"
            )
        self.future_token_count = int(
            motion_token_count if future_token_count is None else future_token_count
        )
        if self.future_token_count <= 0 or self.future_token_count > self.motion_token_count:
            raise ValueError(
                "future_token_count must be in "
                f"[1, {self.motion_token_count}], got {self.future_token_count}"
            )
        self.token_mlp_hidden_dims = tuple(int(dim) for dim in token_mlp_hidden_dims)
        self.detach_root_control_for_body = bool(detach_root_control_for_body)
        if any(dim <= 0 for dim in self.token_mlp_hidden_dims):
            raise ValueError(
                "token_mlp_hidden_dims must contain only positive dimensions, got "
                f"{self.token_mlp_hidden_dims}"
            )
        root_num_heads = int(getattr(root_model, "num_heads", 8))
        body_num_heads = int(getattr(body_model, "num_heads", root_num_heads))
        self.root_injection_layers = self._injection_layers(
            len(root_model.seqTransEncoder.layers), self.num_control_layers
        )
        self.body_injection_layers = self._injection_layers(
            len(body_model.seqTransEncoder.layers), self.num_control_layers
        )
        self.root_branch_injection_layers = self._branch_injection_layers(
            self.root_injection_layers, self.controlnet_scale
        )
        self.body_branch_injection_layers = self._branch_injection_layers(
            self.body_injection_layers, self.controlnet_scale
        )
        self.sequence_pos_encoder = copy.deepcopy(root_model.sequence_pos_encoder)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)
        self.image_projection = nn.Linear(image_feat_dim, latent_dim)

        self.root_layers = self._build_branch_layers(
            root_model.seqTransEncoder.layers,
            self.root_injection_layers,
            self.controlnet_scale,
        )
        self.body_layers = self._build_branch_layers(
            body_model.seqTransEncoder.layers,
            self.body_injection_layers,
            self.controlnet_scale,
        )
        self.root_hint_fusion = _StageHintFusion(
            injection_layers=self.root_injection_layers,
            image_token_count=self.image_token_count,
            future_token_count=self.future_token_count,
            latent_dim=latent_dim,
            num_heads=root_num_heads,
            token_mlp_hidden_dims=self.token_mlp_hidden_dims,
            control_fusion_mode=self.control_fusion_mode,
        )
        self.body_hint_fusion = _StageHintFusion(
            injection_layers=self.body_injection_layers,
            image_token_count=self.image_token_count,
            future_token_count=self.future_token_count,
            latent_dim=latent_dim,
            num_heads=body_num_heads,
            token_mlp_hidden_dims=self.token_mlp_hidden_dims,
            control_fusion_mode=self.control_fusion_mode,
        )

    @staticmethod
    def _injection_layers(backbone_depth: int, num_control_layers: int) -> tuple[int, ...]:
        backbone_depth = int(backbone_depth)
        num_control_layers = int(num_control_layers)
        if num_control_layers <= 0 or num_control_layers > backbone_depth:
            raise ValueError(
                f"num_control_layers must be in [1, {backbone_depth}], got {num_control_layers}"
            )
        if backbone_depth % num_control_layers != 0:
            raise ValueError(
                f"Backbone depth {backbone_depth} must be divisible by "
                f"num_control_layers={num_control_layers}"
            )
        stride = backbone_depth // num_control_layers
        return tuple(range(stride - 1, backbone_depth, stride))

    @staticmethod
    def _branch_injection_layers(
        injection_layers: tuple[int, ...], controlnet_scale: int
    ) -> tuple[int, ...]:
        """Return the branch-layer indices at which visual hints are emitted.

        ``injection_layers`` describes the frozen Kimodo backbone positions and
        is intentionally kept separate from the number of layers in the
        ControlNet branch.  Each branch segment has ``controlnet_scale``
        layers, so the hint is emitted after every segment.
        """
        if controlnet_scale <= 0:
            raise ValueError(
                f"controlnet_scale must be a positive integer, got {controlnet_scale}"
            )
        return tuple(
            (segment_index + 1) * controlnet_scale - 1
            for segment_index in range(len(injection_layers))
        )

    @staticmethod
    def _randomized_layer(template):
        """Clone a Transformer layer and reset its learnable submodules.

        The branch uses the same layer type/configuration as Kimodo, but these
        layers must not inherit the copied Kimodo weights.  PyTorch's
        ``TransformerEncoderLayer`` does not expose a public reset method, so
        reset every child module that provides ``reset_parameters``.
        """
        layer = copy.deepcopy(template)
        for child in layer.modules():
            if child is layer:
                continue
            reset_parameters = getattr(child, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()
        return layer

    @staticmethod
    def _copied_backbone_indices(
        backbone_depth: int, branch_depth: int
    ) -> tuple[int, ...]:
        """Choose copied Kimodo layers from the total ControlNet depth.

        This makes equivalent total-depth configurations share the same
        copied layers.  For example, a 16-layer Kimodo with branch depth 8
        selects layers ``(1, 3, ..., 15)`` whether the branch is configured as
        8 injections x 1 block or 4 injections x 2 blocks.
        """
        if branch_depth <= backbone_depth:
            if backbone_depth % branch_depth != 0:
                raise ValueError(
                    f"Kimodo depth {backbone_depth} must be divisible by "
                    f"ControlNet copied depth {branch_depth}"
                )
            stride = backbone_depth // branch_depth
            return tuple(range(stride - 1, backbone_depth, stride))
        return tuple(range(backbone_depth))

    @classmethod
    def _build_branch_layers(
        cls,
        backbone_layers,
        injection_layers: tuple[int, ...],
        controlnet_scale: int,
    ) -> nn.ModuleList:
        """Build one branch segment per main-backbone injection.

        A segment contains the random prefix required when the requested
        branch is deeper than the corresponding Kimodo span, followed by the
        copied Kimodo layers.  Copied layers are selected from the *total*
        branch depth, so equivalent configurations use the same Kimodo
        layers.  For example, ``num_control_layers=4, scale=2`` and
        ``num_control_layers=8, scale=1`` both copy Kimodo layers
        ``(2, 4, ..., 16)`` (1-based).

        At ``scale=1`` this reduces exactly to the legacy construction: one
        copied layer (the final layer of each Kimodo segment) per injection.
        """
        backbone_depth = len(backbone_layers)
        num_control_layers = len(injection_layers)
        if num_control_layers <= 0:
            raise ValueError("At least one ControlNet injection layer is required")
        if backbone_depth % num_control_layers != 0:
            raise ValueError(
                f"Backbone depth {backbone_depth} must be divisible by "
                f"num_control_layers={num_control_layers}"
            )
        if controlnet_scale <= 0:
            raise ValueError(
                f"controlnet_scale must be a positive integer, got {controlnet_scale}"
            )

        branch_depth = num_control_layers * controlnet_scale
        backbone_segment_depth = backbone_depth // num_control_layers
        copied_per_segment = min(controlnet_scale, backbone_segment_depth)
        random_per_segment = controlnet_scale - copied_per_segment
        copied_indices = cls._copied_backbone_indices(backbone_depth, branch_depth)
        expected_copied_depth = num_control_layers * copied_per_segment
        if len(copied_indices) != expected_copied_depth:
            raise RuntimeError(
                f"Selected {len(copied_indices)} copied Kimodo layers, "
                f"expected {expected_copied_depth}"
            )
        layers = nn.ModuleList()
        random_template = backbone_layers[0]
        for segment_index in range(num_control_layers):
            for _ in range(random_per_segment):
                layers.append(cls._randomized_layer(random_template))

            copied_start = segment_index * copied_per_segment
            copied_end = copied_start + copied_per_segment
            for layer_index in copied_indices[copied_start:copied_end]:
                layers.append(copy.deepcopy(backbone_layers[layer_index]))

        if len(layers) != branch_depth:
            raise RuntimeError(
                f"Built {len(layers)} ControlNet layers, expected {branch_depth}"
            )
        return layers

    def _collect_visual_tokens(
        self,
        xseq: torch.Tensor,
        layers: nn.ModuleList,
        branch_injection_layers: tuple[int, ...],
        output_injection_layers: tuple[int, ...],
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        visual_tokens_by_layer = {}
        branch_injection_lookup = {
            branch_layer_index: output_layer_index
            for branch_layer_index, output_layer_index in zip(
                branch_injection_layers, output_injection_layers
            )
        }
        for branch_layer_index, layer in enumerate(layers):
            xseq = layer(xseq, src_key_padding_mask=padding_mask)
            output_layer_index = branch_injection_lookup.get(branch_layer_index)
            if output_layer_index is not None:
                visual_tokens_by_layer[output_layer_index] = xseq[:, 1:]
        return xseq, visual_tokens_by_layer

    def forward(
        self,
        timesteps: torch.Tensor,
        image_features: torch.Tensor,
        sequence_length: int,
        future_start: int,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        if image_features.shape[1] != self.image_token_count:
            raise ValueError(
                f"Expected {self.image_token_count} image tokens, got {image_features.shape[1]}"
            )
        if sequence_length != self.motion_token_count:
            raise ValueError(
                f"Expected motion length {self.motion_token_count}, got {sequence_length}"
            )
        if future_start < 0 or future_start > sequence_length:
            raise ValueError(f"Invalid future_start={future_start} for length {sequence_length}")
        available_future_length = sequence_length - future_start
        if available_future_length != self.future_token_count:
            raise ValueError(
                "The configured future token count must exactly match the future motion "
                f"range: configured={self.future_token_count}, "
                f"available={available_future_length}"
            )

        time_token = self.embed_timestep(timesteps)
        image_tokens = self.image_projection(image_features)
        xseq = self.sequence_pos_encoder(torch.cat([time_token, image_tokens], dim=1))
        padding_mask = torch.zeros(
            xseq.shape[:2], dtype=torch.bool, device=xseq.device
        )

        xseq, root_visual_tokens = self._collect_visual_tokens(
            xseq,
            self.root_layers,
            self.root_branch_injection_layers,
            self.root_injection_layers,
            padding_mask,
        )
        body_xseq = (
            xseq.detach()
            if self.training and self.detach_root_control_for_body
            else xseq
        )
        _, body_visual_tokens = self._collect_visual_tokens(
            body_xseq,
            self.body_layers,
            self.body_branch_injection_layers,
            self.body_injection_layers,
            padding_mask,
        )
        return root_visual_tokens, body_visual_tokens
