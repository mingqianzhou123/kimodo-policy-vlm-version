# Seven-task HumanoidArena scratch experiment

The experiment uses `configs/arena_sonicx7_groot.yaml` on `model_improvement_lct1`.
It tests the existing N1.7 VLM/shared-adapter/eight residual-injection design in
joint training. No single-task policy, 419-hour policy, or old multi-task policy
weights initialize this experiment. Frozen Kimodo Root/Body and official N1.7
VLM/adapter weights are retained, as in the single-task design. The hand head and
injections start fresh.

## Original training recipe

Reference: `/root/workspace/kimodo-policy/reproduce_humanoidarena_sonicx7.yaml`,
the original `scripts/HumanoidArena_Multi_Task/` configuration and original
`train.py` / `data/datasetloader.py`.

| Setting | Original and this experiment |
|---|---|
| Dataset | `HumanoidArena_merged_datasets_v3_1/sonic_8_refpose_v3_1` |
| Tasks | Football, DoubleDesk, P&P Box, OpenDoor, SitSofa, Boxing, VisionNavigation |
| Episodes / windows | 700 (100 per task) / 293,864 |
| Excluded merged task | `HOI_grap_cup`, per original loader |
| Sampling | episode-uniform, stride 1, continuous absolute sample ordinals |
| Training | 500,000 optimizer updates; global batch 128 = 4 GPUs × 32 |
| Seeds | 42; hand initialization 3407 |
| Precision | bf16 autocast, fp32 parameters |
| Temporal settings | history 100, future 50, 30 fps |
| Data loading | 24 spawned workers per GPU, prefetch 1, episode cache 700 |
| Augmentation | none |
| Optimizer | fused AdamW, betas .9/.999, eps 1e-8, weight decay .01 |
| Scheduler | 5,000-update warmup, cosine decay to 0.1 of initial LR |
| Gradient clipping | motion conditioning 1.0; hand 1.0, independently |
| Loss | original future-action MSE; Root 4, Body 1, Hand 1, hand transition .2 |
| Save interval | 5,000 updates; additional early 100/1,000 checkpoints |
| Compilation | `torch.compile`, `reduce-overhead`, `dynamic=False` |

The original configuration's `batch_size: 32` is **per GPU**. This repository's
`main.batch_size: 128` is **global** and `micro_batch_size: 32` is per GPU.
`ArenaOrdinalBatchSampler` reproduces Accelerate's contiguous per-rank shards
of the original ordinal stream, including resume offsets. Sampling a shuffled
finite range of 293,864 indices would reuse a limited set of deterministic
episode/cut selections, so this experiment does not use that sampler.

The architecture-specific optimizer schedule remains the already tested
two-stage design: updates 0–9,999 train the eight new injections and HandNet
(62,889,990 parameters, LR 1e-4); at update 10,000 the shared LayerNorm/four-layer
adapter joins training (264,327,174 total, adapter LR 3e-5). Frozen VLM and
Root/Body stay frozen. No vision-unfreeze stage is enabled. Adam moments of
existing parameters and the global LR schedule survive the stage change.

The VLM's processor and variable-length feature/adapter extraction run outside
the compiled motion forward, preserving their autograd connection. The motion
forward, injections, hand head and losses use compilation. This is a runtime
adaptation for the VLM processor; the original sampling and supervision are
unchanged.

## Verification and evaluation

`outputs/sonicx7_preflight/` holds the exact configuration fingerprint and real
checks. `data_audit.json` compares original-loader episode/cut identities at
early, dataset-boundary and late ordinals through 63,999,999. The full
`check_groot.py` check uses a real batch of 32 containing all seven tasks:
different token lengths/masking, exact zero-residual Root/Body parity, detach
boundaries, frozen weights, both training stages, checkpoint round-trip and
one VLM call for a full 10-step DDIM plan. The distributed check additionally
exercises actual four-GPU compilation, stage transition and checkpoint resume,
and compares all saved model tensors across all ranks.

Before launch, all three tasks must complete a short real simulator inference.
These 30-step smoke episodes are **not success-rate measurements**.

Formal evaluation uses the unchanged local HumanoidArena SONIC evaluator:
fp32, DDIM 10, 50 predicted frames, 15 executed model frames, original timing,
RTC off, control 50 Hz, seeds 0/1/2 × 20 repeats (60 episodes per task),
nonpersistent simulator, root rotation layout `row`, root max delta 26°.
Football and DoubleDesk use 2,000-step limits; P&P Box uses the original
**1,450-step limit**. The official P&P Box task name ends in `Wholedoby`.
Each evaluation stores environment/controller/evaluator SHA256 hashes and
rejects server failures or missing successful policy requests.

| Paper Table 2, non-star `Ours` (scratch) | Success rate |
|---|---|
| Football | 35.0 ± 7.1% |
| DoubleDesk | 40.0 ± 0.0% |
| P&P Box | 80.0 ± 4.1% |
| HOI mean | 51.67% |

These are published references, not measurements from this run. The launcher
can additionally evaluate the existing original seven-task scratch 500k model
with the **same wrapper, environment files, seeds, limits and controller**.
That model is only used for evaluation; it never initializes training.
One new training seed with 60 evaluation episodes per task cannot reproduce
the paper's training-seed uncertainty or by itself establish statistical
significance.

```bash
.venv-groot/bin/python scripts/run_arena_sonicx7_groot.py \
  --gpus 4,5,6,7 --gpu 7 --detach \
  --baseline-checkpoint /root/workspace/kimodo-policy/log/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse/2026-10-18_21-07-12/checkpoint_500000
```

The launcher requires data/model/DDP/three simulator checks before training.
It evaluates the new model on the three HOI tasks first, then the matched
original model, writing `experiment_summary.json`. The CPU monitor refreshes
PNG/PDF curves, full metric CSV, health JSON and an HTML dashboard every minute,
with hourly snapshots. The existing xiancun Supervisor remains enabled; it
yields during GPU work and resumes when GPUs become idle. Its status is checked
before training, after training, and on both normal and failed pipeline exits.
