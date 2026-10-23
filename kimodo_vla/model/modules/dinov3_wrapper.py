import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel


class DINOv3Encoder(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(checkpoint_path)
        self.model = AutoModel.from_pretrained(checkpoint_path)
        self.image_size = self.model.config.image_size
        self.output_dim = self.model.config.hidden_size
        self.num_register_tokens = int(getattr(self.model.config, "num_register_tokens", 4))

    @torch.compiler.disable
    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Run the checkpoint's official image processor outside the compiled graph."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected images with shape [B, 3, H, W], got {tuple(images.shape)}"
            )
        if images.dtype == torch.uint8:
            do_rescale = True
        elif torch.is_floating_point(images):
            # Float callers use the documented [0, 1] input convention.
            do_rescale = False
        else:
            raise TypeError(
                "DINOv3 images must be uint8 in [0, 255] or floating point in [0, 1], "
                f"got {images.dtype}"
            )
        return self.image_processor(
            images=images,
            return_tensors="pt",
            do_rescale=do_rescale,
        )["pixel_values"]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] uint8 tensor in [0,255] or float tensor in [0,1]
        Returns:
            [B, 196, 1024] patch token features
        """
        x = self._preprocess(images)
        model_dtype = next(self.model.parameters()).dtype
        if model_dtype == torch.float16:
            raise RuntimeError(
                "DINOv3 FP16 inference is numerically unstable for this checkpoint; "
                "use BF16 or FP32 for the image encoder"
            )
        x = x.to(device=images.device, dtype=model_dtype)
        outputs = self.model(pixel_values=x)
        prefix_tokens = 1 + self.num_register_tokens
        return outputs.last_hidden_state[:, prefix_tokens:]
