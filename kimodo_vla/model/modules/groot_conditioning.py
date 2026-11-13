"""GR00T N1.7 features and future-only residual attention for Kimodo.

The official Qwen3Backbone and SelfAttentionTransformer implementations are
loaded from a pinned, local Isaac-GR00T checkout. No GR00T action head is built.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.nn.utils.rnn import pad_sequence


GROOT_REVISION = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
GROOT_SOURCE_REVISION = "c56f54128d92f743889b7e4cecfc1ceafd88739e"


@dataclass
class VLCondition:
    tokens: torch.Tensor
    mask: torch.Tensor  # True means a real token, never padding.

    def __post_init__(self):
        if self.tokens.ndim != 3 or self.mask.shape != self.tokens.shape[:2]:
            raise ValueError("Expected VL tokens [B,L,D] and mask [B,L]")
        if self.mask.dtype != torch.bool or not self.mask.any(dim=1).all():
            raise ValueError("Every VL sequence must have at least one valid token")

    def to(self, **kwargs):
        tokens = self.tokens.to(**kwargs)
        return VLCondition(tokens, self.mask.to(device=tokens.device))


def masked_mean_tokens(condition: VLCondition) -> torch.Tensor:
    """Summarise variable length VLM tokens without allowing padding to leak in."""
    weights = condition.mask.unsqueeze(-1).to(dtype=condition.tokens.dtype)
    return (condition.tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def official_module(repo: str, relative: str):
    """Import a source file without importing the GR00T training/deployment stack."""
    path = Path(repo).expanduser().resolve() / relative
    name = "_kimodo_groot_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load GR00T source {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checkpoint_parts(directory: str, modules: dict[str, nn.Module]) -> dict:
    """Load only requested prefixes, strictly, without allocating action-head tensors."""
    root = Path(directory)
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())["weight_map"]
        shards = sorted({shard for key, shard in index.items()
                         if any(key.startswith(prefix) for prefix in modules)})
    else:
        shards = ["model.safetensors"]
    states = {prefix: {} for prefix in modules}
    for shard in shards:
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                for prefix in modules:
                    if key.startswith(prefix):
                        states[prefix][key.removeprefix(prefix)] = handle.get_tensor(key)
                        break
    report = {}
    for prefix, module in modules.items():
        if not states[prefix]:
            raise RuntimeError(f"No checkpoint tensors for {prefix} in {root}")
        module.load_state_dict(states[prefix], strict=True)
        report[prefix] = {"tensors": len(states[prefix]),
                          "parameters": sum(v.numel() for v in states[prefix].values())}
    return report


class SharedVLMAdapter(nn.Module):
    """Checkpoint LayerNorm -> four self-attention layers; width stays 2048.

    The official adapter has no padding-mask argument. Process equal-length
    unpadded sequences together so padding cannot contaminate valid features.
    """

    def __init__(self, vlln: nn.Module, vl_self_attention: nn.Module):
        super().__init__()
        self.vlln = vlln
        self.vl_self_attention = vl_self_attention

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> VLCondition:
        condition = VLCondition(features, mask)
        lengths = condition.mask.sum(dim=1)
        rows = [None] * len(features)
        for length in lengths.unique().tolist():
            indices = (lengths == length).nonzero().flatten().tolist()
            group = torch.stack([features[i, mask[i]] for i in indices])
            adapted = self.vl_self_attention(self.vlln(group))
            for i, row in zip(indices, adapted.unbind()):
                rows[i] = row
        tokens = pad_sequence(rows, batch_first=True)
        valid = torch.arange(tokens.shape[1], device=tokens.device)[None] < lengths[:, None]
        return VLCondition(tokens, valid)


class GrootN17Encoder(nn.Module):
    output_dim = 2048
    token_count = None  # image AND text tokens, variable length

    def __init__(self, checkpoint: str, repo: str, vlm_model: str | None = None,
                 vision_lora: bool = False, vision_lora_rank: int = 8,
                 vision_lora_alpha: float = 16.0, vision_lora_dropout: float = 0.0,
                 vision_lora_targets: tuple[str, ...] = ("qkv", "proj")):
        super().__init__()
        from transformers import AutoProcessor

        root = Path(checkpoint).expanduser().resolve()
        cfg = json.loads((root / "config.json").read_text())
        adapter_cfg = cfg.get("vl_self_attention_cfg", {})
        if (cfg.get("backbone_embedding_dim") != 2048
                or adapter_cfg.get("num_layers") != 4
                or adapter_cfg.get("num_attention_heads", 0)
                * adapter_cfg.get("attention_head_dim", 0) != 2048
                or not cfg.get("use_vlln")):
            raise ValueError("Expected N1.7 checkpoint with LayerNorm and four 2048D adapter layers")
        if int(cfg.get("select_layer", 0)) <= 0:
            raise ValueError("Checkpoint must explicitly specify a positive VLM select_layer")
        self.checkpoint_config = cfg
        self.vlm_model = vlm_model or cfg["model_name"]
        self.processor = AutoProcessor.from_pretrained(self.vlm_model)
        self.processor.tokenizer.padding_side = "left"
        qwen = official_module(repo, "gr00t/model/modules/qwen3_backbone.py")
        # GR00T targets transformers 4.57.3; Kimodo uses 5.1.0, which removed
        # the public language_model/visual aliases. Restore READ-ONLY aliases
        # in a local subclass. Forward computation and checkpoint key paths
        # remain the original HF implementation (no duplicate module entries).
        base_class = qwen.Qwen3VLForConditionalGeneration
        if not hasattr(base_class, "language_model"):
            class CompatibleQwen3VL(base_class):
                @property
                def language_model(self):
                    return self.model.language_model

                @property
                def visual(self):
                    return self.model.visual

            qwen.Qwen3VLForConditionalGeneration = CompatibleQwen3VL
        dit = official_module(repo, "gr00t/model/modules/dit.py")
        augment = official_module(repo, "gr00t/model/gr00t_n1d7/image_augmentations.py")
        _, self.image_transform = augment.build_image_transformations_albumentations(
            cfg.get("image_target_size", [256, 256]), cfg.get("image_crop_size", [230, 230]),
            0, None, cfg.get("shortest_image_edge", 256), cfg.get("crop_fraction", 0.95),
            letter_box_transform=cfg.get("letter_box_transform", False),
        )
        # Load the official base layout, then overwrite ALL retained backbone
        # parameters with the N1.7 policy-pretrained weights (not just Cosmos).
        self.backbone = qwen.Qwen3Backbone(
            model_name=self.vlm_model, select_layer=int(cfg["select_layer"]),
            tune_llm=False, tune_visual=False, load_bf16=True,
            use_flash_attention=False,
        )
        self.adapter = SharedVLMAdapter(nn.LayerNorm(2048), dit.SelfAttentionTransformer(**adapter_cfg))
        self.load_report = load_checkpoint_parts(str(root), {
            "backbone.": self.backbone,
            "action_head.vlln.": self.adapter.vlln,
            "action_head.vl_self_attention.": self.adapter.vl_self_attention,
        })
        self.backbone._reset_rotary_inv_freq()
        self.source_hashes = {
            name: hashlib.sha256((Path(repo) / name).read_bytes()).hexdigest()
            for name in ("gr00t/model/modules/qwen3_backbone.py", "gr00t/model/modules/dit.py")
        }
        self.requires_grad_(False)
        self.vision_lora = bool(vision_lora)
        self.vision_lora_rank = int(vision_lora_rank)
        self.vision_lora_alpha = float(vision_lora_alpha)
        self.vision_lora_dropout = float(vision_lora_dropout)
        self.vision_lora_targets = tuple(vision_lora_targets)
        if self.vision_lora:
            self._install_vision_lora()
        self.tune_visual = False

    def _install_vision_lora(self):
        """Attach PEFT LoRA modules to the visual attention projections only."""
        if getattr(self, "_vision_lora_installed", False):
            return
        from peft import LoraConfig, get_peft_model
        targets = list(self.vision_lora_targets)
        if not targets:
            raise ValueError("vision_lora_targets must contain at least one module suffix")
        self.backbone.model.visual = get_peft_model(
            self.backbone.model.visual,
            LoraConfig(
                r=self.vision_lora_rank,
                lora_alpha=self.vision_lora_alpha,
                lora_dropout=self.vision_lora_dropout,
                target_modules=targets,
                bias="none",
                task_type=None,
            ),
        )
        self._vision_lora_installed = True
        self.backbone.model.visual.requires_grad_(False)

    def _set_vision_lora_trainable(self, enabled: bool):
        if not self.vision_lora:
            if enabled:
                raise ValueError("vision_lora stage requested but vision_lora is disabled")
            return
        self.backbone.model.visual.requires_grad_(False)
        for name, parameter in self.backbone.model.visual.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(enabled)

    def set_stage(self, stage: str):
        if stage not in {"injection", "adapter", "vision", "vision_lora"}:
            raise ValueError(f"Unknown VLM training stage {stage!r}")
        self.requires_grad_(False)
        self.adapter.requires_grad_(stage not in {"injection"})
        self.tune_visual = stage == "vision"
        self._set_vision_lora_trainable(stage == "vision_lora")
        self.backbone.tune_visual = self.tune_visual
        if self.tune_visual:
            self.backbone.model.visual.requires_grad_(True)
        elif self.vision_lora:
            self._set_vision_lora_trainable(stage == "vision_lora")
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        if self.tune_visual or (self.vision_lora and any("lora_" in n and p.requires_grad
                                                         for n, p in self.backbone.model.visual.named_parameters())):
            self.backbone.model.visual.train(mode)
        if not any(p.requires_grad for p in self.adapter.parameters()):
            self.adapter.eval()
        return self

    @torch.compiler.disable
    def preprocess(self, images: torch.Tensor, instructions: list[str]):
        if images.ndim != 4 or images.shape[1] != 3 or len(instructions) != len(images):
            raise ValueError("N1.7 expects one current RGB image and instruction per sample")
        if images.dtype != torch.uint8:
            if not torch.is_floating_point(images) or not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
                raise ValueError("Images must be uint8 [0,255] or finite float [0,1]")
            images = (images * 255).round().to(torch.uint8)
        # Use the official deterministic resize/center-crop/resize for BOTH
        # training and planning, preserving the original Arena no-augmentation
        # recipe. The Cosmos processor then owns rescale and normalization.
        frames = [torch.from_numpy(self.image_transform(image=frame.permute(1, 2, 0).numpy())["image"])
                  .permute(2, 0, 1) for frame in images.detach().cpu().unbind()]
        texts = []
        for frame, instruction in zip(frames, instructions):
            if self.checkpoint_config.get("formalize_language", True):
                instruction = re.sub(r"[^\w\s]", "", instruction.lower())
            conversation = [{"role": "user", "content": [
                {"type": "image", "image": frame}, {"type": "text", "text": instruction},
            ]}]
            texts.append(self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False))
        return self.processor(text=texts, images=frames, padding=True, return_tensors="pt")

    def forward(self, images: torch.Tensor, instructions: list[str]) -> VLCondition:
        inputs = self.preprocess(images, instructions)
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype
        inputs = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
                  for key, value in inputs.items()}
        ctx = contextlib.nullcontext() if self.tune_visual else torch.no_grad()
        with ctx:
            output = self.backbone(inputs)
        features = output["backbone_features"].to(dtype=next(self.adapter.parameters()).dtype)
        return self.adapter(features, output["backbone_attention_mask"])


class CrossAttentionResidual(nn.Module):
    def __init__(self, motion_dim=1024, condition_dim=2048, num_heads=8):
        super().__init__()
        self.query_norm = nn.LayerNorm(motion_dim)
        self.attention = nn.MultiheadAttention(
            motion_dim, num_heads, kdim=condition_dim, vdim=condition_dim,
            dropout=0.0, batch_first=True,
        )
        self.output_projection = nn.Linear(motion_dim, motion_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, query: torch.Tensor, condition: VLCondition):
        update, _ = self.attention(
            self.query_norm(query), condition.tokens, condition.tokens,
            key_padding_mask=~condition.mask, need_weights=False,
        )
        return self.output_projection(update)


class AdaLNZeroResidual(nn.Module):
    """External AdaLN-Zero modulation for one frozen Kimodo hidden sequence.

    The modulation is intentionally applied outside the checkpointed
    TransformerEncoderLayer.  A zero-initialised head therefore makes the
    adapter an exact identity at construction time while keeping the action
    backbone unchanged and frozen.
    """

    def __init__(self, motion_dim: int = 1024, condition_dim: int = 512):
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.condition_dim = int(condition_dim)
        self.norm = nn.LayerNorm(self.motion_dim, elementwise_affine=False)
        self.modulation = nn.Linear(self.condition_dim, 3 * self.motion_dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, query: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if query.ndim != 3 or query.shape[-1] != self.motion_dim:
            raise ValueError(
                f"Expected query [B,T,{self.motion_dim}], got {tuple(query.shape)}"
            )
        if condition.ndim != 2 or condition.shape != (query.shape[0], self.condition_dim):
            raise ValueError(
                f"Expected condition [B,{self.condition_dim}], got {tuple(condition.shape)}"
            )
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        normalized = self.norm(query)
        update = normalized * (1.0 + scale[:, None, :]) + shift[:, None, :]
        return gate[:, None, :] * update


class AdaLNZeroInjectionStack(nn.Module):
    """Root or body stack of independent, timestep-conditioned AdaLN heads."""

    accepts_timesteps = True

    def __init__(
        self,
        layers=(3, 7, 11, 15),
        motion_dim: int = 1024,
        condition_input_dim: int = 2048,
        condition_dim: int = 512,
        max_diffusion_steps: int = 1000,
        pool: str = "masked_mean",
    ):
        super().__init__()
        if pool != "masked_mean":
            raise ValueError(f"Unsupported AdaLN VLM pooling {pool!r}; use 'masked_mean'")
        if condition_dim <= 0 or max_diffusion_steps <= 0:
            raise ValueError("condition_dim and max_diffusion_steps must be positive")
        # Import lazily to keep the official GR00T import path independent of
        # the Kimodo backbone module.
        from .backbone import PositionalEncoding, TimestepEmbedder

        self.layers = tuple(int(layer) for layer in layers)
        self.motion_dim = int(motion_dim)
        self.condition_input_dim = int(condition_input_dim)
        self.condition_dim = int(condition_dim)
        self.pool = pool
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(self.condition_input_dim),
            nn.Linear(self.condition_input_dim, self.condition_dim),
        )
        time_encoding = PositionalEncoding(
            self.condition_dim, dropout=0.0, max_len=int(max_diffusion_steps)
        )
        self.timestep_embedding = TimestepEmbedder(self.condition_dim, time_encoding)
        self.condition_mlp = nn.Sequential(
            nn.Linear(2 * self.condition_dim, self.condition_dim),
            nn.SiLU(),
            nn.Linear(self.condition_dim, self.condition_dim),
        )
        self.blocks = nn.ModuleDict({
            str(layer): AdaLNZeroResidual(self.motion_dim, self.condition_dim)
            for layer in self.layers
        })

    def _condition(self, condition: VLCondition, timesteps: torch.Tensor) -> torch.Tensor:
        if not isinstance(condition, VLCondition):
            raise TypeError("AdaLN-Zero expects a VLCondition")
        if timesteps.ndim != 1 or timesteps.shape[0] != condition.tokens.shape[0]:
            raise ValueError(
                "timesteps must have shape [B] matching the VLM condition batch"
            )
        if timesteps.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            timesteps = timesteps.long()
        visual = self.visual_projection(masked_mean_tokens(condition))
        temporal = self.timestep_embedding(timesteps.to(device=visual.device)).squeeze(1)
        return self.condition_mlp(torch.cat((visual, temporal), dim=-1))

    def forward(
        self,
        layer: int,
        condition: VLCondition,
        query: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        try:
            block = self.blocks[str(int(layer))]
        except KeyError as error:
            raise ValueError(f"Layer {layer} is not configured for AdaLN-Zero") from error
        return block(query, self._condition(condition, timesteps))


class InjectionStack(nn.Module):
    def __init__(self, layers=(3, 7, 11, 15), motion_dim=1024, condition_dim=2048, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleDict({str(i): CrossAttentionResidual(motion_dim, condition_dim, num_heads)
                                     for i in layers})

    def forward(self, layer: int, condition: VLCondition, query: torch.Tensor):
        return self.blocks[str(layer)](query, condition)


class GrootInjection(nn.Module):
    """Eight independent modules sharing one VLM condition.

    ``mode='cross_attention'`` is the original model-improvement path.  The
    ``adaln_zero`` path replaces only its external fuser and leaves the frozen
    Kimodo Transformer layers untouched.
    """

    root_injection_layers = body_injection_layers = (3, 7, 11, 15)
    image_token_count = None

    def __init__(
        self,
        denoiser,
        future_token_count=50,
        mode: str = "cross_attention",
        adaln_condition_dim: int = 512,
        adaln_pool: str = "masked_mean",
        max_diffusion_steps: int = 1000,
    ):
        super().__init__()
        for model in (denoiser.root_model, denoiser.body_model):
            if model.latent_dim != 1024 or len(model.seqTransEncoder.layers) != 16:
                raise ValueError("N1.7 injection requires Kimodo's 16-layer, 1024D backbones")
        if future_token_count != 50:
            raise ValueError("This experiment requires a 50-frame future horizon")
        mode = str(mode).strip().lower()
        if mode not in {"cross_attention", "adaln_zero"}:
            raise ValueError(
                "groot_injection_mode must be 'cross_attention' or 'adaln_zero', "
                f"got {mode!r}"
            )
        self.mode = mode
        self.future_token_count = future_token_count
        if mode == "cross_attention":
            self.root_hint_fusion = InjectionStack()
            self.body_hint_fusion = InjectionStack()
        else:
            self.root_hint_fusion = AdaLNZeroInjectionStack(
                condition_dim=adaln_condition_dim,
                pool=adaln_pool,
                max_diffusion_steps=max_diffusion_steps,
            )
            self.body_hint_fusion = AdaLNZeroInjectionStack(
                condition_dim=adaln_condition_dim,
                pool=adaln_pool,
                max_diffusion_steps=max_diffusion_steps,
            )

    def forward(self, timesteps, condition: VLCondition, motion_token_count, future_start):
        if motion_token_count - future_start != self.future_token_count:
            raise ValueError("Inject only the 50 future motion tokens")
        # Reuse the same features across stages, locations AND diffusion steps.
        return ({i: condition for i in self.root_injection_layers},
                {i: condition for i in self.body_injection_layers})
