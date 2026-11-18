# GR00T N1.7 embedded 32-block CrossAttention

This experiment is an additional model-improvement variant. The existing
external sparse CrossAttention and AdaLN-Zero configurations are unchanged.

`configs/arena_sonicx7_groot_embedded32.yaml` selects
`model.groot_injection_mode: embedded_cross_attention`. Each pretrained
Kimodo Root and Body self-attention layer is followed by one ordinary residual
CrossAttention block, giving 16 + 16 = 32 blocks. Only the 50 future motion
tokens are queries; the text, timestep, heading prefix and 100-frame history
continue through Kimodo's original self-attention path. GR00T N1.7 image-text
features are keys/values and the full VLM, including its four-layer feature
adapter, remains frozen. The attention projections use an internal width of
512 to keep the 32-block run within the four-GPU memory budget.

The first 30,000 updates train the embedded CrossAttention blocks and HandNet
with the pretrained Kimodo weights frozen. The second stage unfreezes all 16
Root and 16 Body Kimodo self-attention layers and output projections using a
smaller learning rate. No external zero-initialized residual is used.

Before a production run, execute the real-data preflight with the new config:

```bash
python scripts/check_groot.py \
  --config configs/arena_sonicx7_groot_embedded32.yaml \
  --device cuda \
  --report outputs/sonicx7_embedded32_preflight/groot_preflight.json
```

Then launch the four-GPU joint Arena run with `CUDA_VISIBLE_DEVICES=0,1,2,3`
and the same `torchrun` environment used by the existing Arena recipes:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv-groot/bin/python \
  scripts/run_arena_sonicx7_groot.py \
  --config configs/arena_sonicx7_groot_embedded32.yaml \
  --gpus 0,1,2,3 --gpu 0 \
  --validation-root outputs/sonicx7_embedded32_preflight \
  --detach
```
