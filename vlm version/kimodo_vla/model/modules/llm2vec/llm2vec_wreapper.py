# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM2Vec encoder wrapper for Kimodo text conditioning."""

import os

import numpy as np
import torch

from .llm2vec import LLM2Vec


class LLM2VecEncoder(torch.nn.Module):
    """LLM2Vec text embeddings."""

    def __init__(
        self,
        checkpoint_path: str,
    ) -> None:
        super().__init__()
        # 1. 模型的基本配置
        self.llm_dim = 4096
        # 2. 加载模型
        self.model = LLM2Vec.from_pretrained(
            base_model_name_or_path=os.path.join(checkpoint_path, "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"),
            peft_model_name_or_path=os.path.join(checkpoint_path, "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"),
            torch_dtype=torch.bfloat16,
        )
        # 3. 冻结模型
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, text: list[str] | str):
        is_string = False
        if isinstance(text, str):
            text = [text]
            is_string = True

        with torch.no_grad():
            device = next(self.model.parameters()).device
            encoded_text = self.model.encode(
                text,
                batch_size=len(text),
                show_progress_bar=False,
                device=str(device),
            )

        assert len(encoded_text.shape)
        assert self.llm_dim == encoded_text.shape[-1]

        encoded_text = encoded_text[:, None]
        lengths = np.ones(len(encoded_text), dtype=int).tolist()

        if is_string:
            encoded_text = encoded_text[0]
            lengths = lengths[0]

        encoded_text = torch.as_tensor(encoded_text, device=device)
        return encoded_text, lengths
