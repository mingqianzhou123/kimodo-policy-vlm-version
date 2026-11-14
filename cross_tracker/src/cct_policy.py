"""CCT conditioning through Kimodo's existing text-prefix interface."""
from pathlib import Path
import torch
from kimodo_vla.model.kimodo_policy import KimodoPolicy
from cct_context import InteractionEncoder


class CCTPolicy(KimodoPolicy):
    def __init__(self, config=None):
        super().__init__(config)
        self.cct_encoder = InteractionEncoder(
            visual_dim=self.image_encoder.output_dim,
            output_dim=self.denoiser.root_model.embed_text.in_features,
            patches=self.image_encoder.token_count,
        )
        # The original pooled-text prefix has one token. Do not truncate CCT's 256 tokens.
        self.denoiser.root_model.num_text_tokens = None
        self.denoiser.body_model.num_text_tokens = None
        self._deployment_context = None
        self.has_trained_context = False

    def context_prefix(self, context):
        dtype = next(self.cct_encoder.parameters()).dtype
        visual = context["visual"].to(self.device, dtype=dtype)
        transitions = context["transitions"].to(self.device, dtype=dtype)
        trajectory_ids = context["trajectory_ids"].to(self.device)
        tokens = self.cct_encoder(visual, transitions, trajectory_ids)
        metadata = context["metadata"].to(self.device, dtype=tokens.dtype)
        if metadata.ndim != 3 or metadata.shape[1] != 1:
            raise ValueError("Controller text must be a cached LLM2Vec token")
        return torch.cat((tokens, metadata), dim=1)

    @staticmethod
    def prepend(prefix, text, lengths):
        # The existing planner passes [1, D]; training passes [B, 1, D].
        if text is not None and text.ndim == 2:
            text = text.unsqueeze(0)
        if text is None or text.ndim != 3 or text.shape[1] != 1:
            raise ValueError("Use the unchanged single-token Kimodo task embedding")
        if lengths is not None and not (torch.as_tensor(lengths) == 1).all():
            raise ValueError("Unexpected task-text padding")
        text = text.to(device=prefix.device, dtype=prefix.dtype)
        joined = torch.cat((prefix, text), dim=1)
        return joined, torch.full((len(text),), joined.shape[1], device=text.device, dtype=torch.long)

    def forward(self, instruction, egoview, gt_motion, gt_mask, **kwargs):
        if not isinstance(egoview, dict):
            raise ValueError("CCT training requires explicit interaction context")
        prefix = self.context_prefix(egoview["context"])
        kwargs["text_feat"], kwargs["text_length"] = self.prepend(
            prefix, kwargs.get("text_feat"), kwargs.get("text_length"))
        return super().forward(instruction, egoview["image"].to(self.device), gt_motion, gt_mask, **kwargs)

    @torch.inference_mode()
    def set_context(self, context):
        """Encode once before a rollout; replace explicitly when the controller changes."""
        if self.training:
            raise RuntimeError("Call eval() before setting deployment context")
        self._deployment_context = self.context_prefix(context).detach()

    @torch.inference_mode()
    def predict_future(self, instruction, egoview, history_motion, **kwargs):
        if self._deployment_context is None:
            raise RuntimeError("Set controller context before inference")
        kwargs["text_feat"], kwargs["text_length"] = self.prepend(
            self._deployment_context, kwargs.get("text_feat"), kwargs.get("text_length"))
        return super().predict_future(instruction, egoview, history_motion, **kwargs)

    def load_controlnet_checkpoint(self, checkpoint_path, strict=True, weights="model"):
        step = super().load_controlnet_checkpoint(checkpoint_path, strict=strict, weights=weights)
        path = Path(checkpoint_path)
        if path.is_dir():
            path = path / "training_state.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload.get("model", payload) if weights == "model" else payload["ema"][weights.removeprefix("ema")]
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        encoder = {k.removeprefix("cct_encoder."): v for k, v in state.items() if k.startswith("cct_encoder.")}
        self.has_trained_context = bool(encoder)
        if encoder:
            self.cct_encoder.load_state_dict(encoder, strict=True)
        elif step != 1000000:
            raise RuntimeError("CCT encoder missing from a non-initial checkpoint")
        return step
