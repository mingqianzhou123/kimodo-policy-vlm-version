# Seven-task VLM scratch restart: stability experiment

**Current execution:** see [GPU allocation and speed migration](SPEED_MIGRATION.md).
The selected continuation uses GPUs0-3 exclusively, starting at19260;
HumanoidToolBench continues on4-7. Active status is pointed to by
`outputs/arena_sonicx7_current.json`. The original run described below is historical.

Run name: `scratch_stable_20261105_lr2e5_vl5e6` (an experiment label;
machine UTC and local log timestamps use different zones).

## Motivation and scope

The earlier run resumed at 123307 and diverged near 136680; the guarded
retry stopped at 162943 after ten consecutive excessive gradient norms.
The latter had typical total loss ~0.033 and control gradient norm ~0.13
before root loss rose from ~0.0018 to ~0.031 and gradient norm to ~12.
There was no logged OOM/CUDA failure at that stop. These observations do
not establish the underlying cause. Lower learning rates are an intervention,
not proof that learning rate alone caused the failures.

This is a **new step-zero run**, not a continuation of either failed run.
It uses seven tasks, 100 demonstrations each, original episode-uniform
ordinal sampling, global batch 128, four GPUs 4/5/6/7, microbatch 32, seed42,
and 500000 logical updates. No `initialization`, `--init-checkpoint` or
`--resume` is used for production. The pretrained frozen Kimodo motion prior
and GR00T/Cosmos encoders/adapter initialization remain part of the model;
scratch here means no 419-hour VLA policy initialization, not random VLMs.

## Changes relative to the failed recipe

| Setting | Previous | New |
|---|---:|---:|
| Injection LR | 1e-4 | 2e-5 |
| Hand LR | 1e-4 | 2e-5 |
| VLM adapter LR | 3e-5 | 5e-6 |
| Global warmup | 5000 | 10000 |
| Adapter unfreeze | 10000 | 10000 |
| Adapter warmup after unfreeze | none | 5000 |
| Control/hand clipping | 1.0 | 0.5 |
| Compile | reduce-overhead | off (eager) |
| Data workers/GPU | 24 | 12 |

BF16 autocast with FP32 parameters/Adam states, root/body/hand objective
weights 4/1/1, transition weight0.2, cosine floor0.1, weight decay0.01,
and frozen vision backbone remain unchanged. Adam moments are retained
when enabling adapter parameters. Starting at logical update10001,
loss>1 or preclip norm>10 skips the update; ten consecutive skips stop the
run. Skips advance sampling and the LR scheduler, are counted persistently,
and must be reported separately from actual optimizer updates.
The loss guard is disabled during early scratch convergence to avoid
rejecting ordinary initial loss (~1.6). Nonfinite losses/norms always stop.

DataLoader has a separate generator so iterator reconstruction does not
consume restored training CPU RNG. Each save retains the preceding full
`resume_previous.pt` in addition to current `resume.pt`; this gives an
optimizer rollback point rather than only a model-only checkpoint.

The first smoke attempt found differences in HandNet already BEFORE the
interruption (continuous vs split step24 loss ~0.777001 vs0.777043), although
control metrics matched and final CPU/CUDA RNG states matched. Max final
HandNet parameter difference was ~2.75e-4. This is not evidence of broken
resume loading. Deterministic algorithms and CUBLAS workspace configuration
are therefore enabled for the subsequent smoke and production run. The first
failed validation remains in `outputs/scratch_stable_20261105_lr2e5_vl5e6`;
the production attempt is labeled with `_det` suffix.

## Validation and operation

Real seven-task forward/backward, frozen-backbone/gradient-detachment,
checkpoint roundtrip and inference checks are in
`outputs/sonicx7_stable_preflight/groot_preflight.json`.
Data inventory and original sample-identity checks are in the adjacent
`data_audit.json`. Unit tests cover guards, ordinal sampling and adapter LR
warmup. No new production initialization is taken from these smoke models.

`scripts/run_arena_stable.py` compares a continuous48-step DDP run with
24+resume24 steps, including unfreezing at3, shortened warmups10/5,
all-rank model equality, final model/Adam/scheduler/CPU/CUDA RNG states.
Validation uses a separate directory and stops on mismatch. A short test
does not rule out a late instability at100k+ updates.

On success the supervisor launches the fresh500k run, with per10-step
metrics, per5000-step checkpoints, previous optimizer backup, and the
existing PNG/PDF/CSV/HTML monitor. Errors are recorded and stop the run;
the supervisor does not silently retry altered experiments. Recover with
`scripts/train.py --resume <run>` using the saved config and four GPUs.

## Effectiveness endpoints and limitations

After500k, the supervisor schedules the existing unchanged SONIC evaluator
for football, doubledesk and pp_box, then the original scratch500k baseline,
each using seeds0/1/2 and20 episodes per seed. Reference manifests enforce
the same assets/controller/inference protocol. All seven tasks are trained;
the existing formal evaluator currently implements these three HOI endpoints.
No success-rate claim is justified by falling training loss or this smoke.
Report per-seed success rates, mean/sample SD across seeds, pooled success
counts and binomial uncertainty separately. The three seed groups are
evaluation seeds, not three independent training runs.

The new recipe changes optimization as well as architecture relative to
the original ControlNet baseline. A positive matched simulation comparison
supports this whole recipe; it does not isolate the causal contribution of
VLM replacement. A strict architecture ablation requires an additional
original-ControlNet run with matched optimization and multiple training seeds.

Old failed runs/checkpoints/logs are preserved for comparison.
