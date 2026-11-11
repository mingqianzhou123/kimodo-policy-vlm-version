# HumanoidArena seven-task AdaLN-Zero experiment

`configs/arena_sonicx7_groot_adaln.yaml` is the matched successor to the
Cross-Attention recipe in `arena_sonicx7_groot_stable.yaml`.  It keeps the
official GR00T N1.7 encoder, raw VLM tokens for HandNet, diffusion schedule,
Root→Body detach, and the frozen 16-layer Kimodo action backbone.  Only the
external visual fuser changes:

```text
VLM tokens + valid-token mask -> masked mean [B, 2048]
                              + timestep embedding
                              -> Root/Body condition MLPs
                              -> four independent zero-initialised AdaLN heads
                              -> future 50 motion tokens after layers 3/7/11/15
```

At initialization every AdaLN head emits `shift=scale=gate=0`, so the
denoiser is bitwise-equivalent to the frozen Kimodo path.  Root and Body use
separate condition stems and heads.  The old Cross-Attention path remains
available with `model.groot_injection_mode: cross_attention`; its checkpoints
are intentionally not loaded into AdaLN runs.

The seven-task run uses global batch 128 (four GPUs × micro-batch 32), eight
data workers per GPU, bf16 autocast, fused AdamW, 5,000-step warmup and cosine decay.  Compared with the
previous conservative VLM recipe, the new rates are AdaLN 5e-5, HandNet 3e-5,
and VLM adapter 1e-5.  The Kimodo denoiser and VLM backbone remain frozen;
the VLM adapter joins at step 10,000.  The dedicated AdaLN optimizer group is
saved in checkpoints and included in the architecture fingerprint.

Run the checks in this order on a GPU host:

```bash
.venv-groot/bin/python scripts/check_groot.py \
  --config configs/arena_sonicx7_groot_adaln.yaml \
  --report outputs/sonicx7_adaln_preflight/groot_preflight.json

CUDA_VISIBLE_DEVICES=4,5,6,7 .venv-groot/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 scripts/train.py \
  --config configs/arena_sonicx7_groot_adaln.yaml \
  --run-dir log/arena_sonicx7_groot_adaln/scratch_ddp4_<timestamp>
```

The existing simulator smoke and formal evaluator are unchanged.  Compare
the AdaLN and Cross-Attention runs using the same task seeds, checkpoint
steps, DDIM steps, execution horizon and controller.  The paper reports that
AdaLN-Zero remains robust when the action backbone is frozen, while
Cross-Attention collapses in that setting; this Arena experiment tests that
claim under the local N1.7/Kimodo data path rather than treating the paper's
MimicGen/LIBERO rates as Arena results.

