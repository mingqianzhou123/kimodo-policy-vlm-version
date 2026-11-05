# Execution speed and GPU allocation audit

User priority: seven-task Arena uses GPUs0-3 exclusively; the independent
HumanoidToolBench `measured_current_letterbox_v2/formal_seed1000_20261105`
experiment continues on GPUs4-7. Do not stop that experiment.

GPU process accounting found ~12GB/card of ToolBench allocations on GPUs4-7
in addition to the seven-task trainer. High total utilization did not imply
exclusive access. The previous performance explanation was incomplete:
deterministic/eager execution and contention were confounded. No isolated
determinism ablation was performed, so their individual cost is unknown.

Topology: GPUs0-3 share NUMA0/PHB links; GPUs4-7 share NUMA1/PHB links;
cross-group GPU paths are SYS links. No NVLink path is reported.

## Measured candidates

- Original eager4 sharing GPUs4-7: recent100 logged intervals median1.725s/update.
- Full-model compiled4 on GPUs0-3: short intervals1.15-1.22s/update but
  additional intervals8.4-8.6s/update after the first compile. The bounded
  probe stopped at15059. This does not establish a fully warmed steady rate.
- Eager8 across all GPUs, still sharing GPUs4-7 with ToolBench: median1.238s,
  mean1.275s after excluding startup intervals; completed19140 and passed
  all-eight-rank weight hash equality.
- Eager4 exclusive GPUs0-3: median1.144s/update after startup; completed19260
  and passed all-four-rank weight hash equality.

The denoiser-only compiled probe completed19320 with all-rank hash equality,
but intervals after initial compilation were1.14,1.27,1.21 and13.60s/update.
It was not selected. No benchmark process needed forced termination.

Selected run: `log/arena_sonicx7_groot/scratch_stable_20261105_exclusive4`.
It resumes19260 on GPUs0-3, eager deterministic execution. The selected
checkpoint has finite model/Adam tensors, four RNG states, Adam steps19255
for injection/hand and9255 for the adapter (five historical skipped updates).
Original model/training config and the real preflight fingerprint match;
only execution allocation and explicit recorded topology migrations changed.
The scoped compilation candidate's extra60 test updates are retained as a
separate branch and are not incorporated into the selected continuation.

Probe logs and raw metrics are retained under `outputs/speed_20261105/`
and the corresponding run directories. Do not infer success from a single
fast interval. The short measurements do not establish late-training stability.

## Training state continuity

The original seven-task scratch run safely saved/stopped at19015. An8-rank
continuation consumed original absolute ordinals through19140; a4-rank
continuation consumed through19260. Each production-capable handoff hardlinked
the complete previous `resume.pt`, preserving model/Adam/scheduler/stage/
skip counts. Atomic checkpoint replacement leaves the predecessor untouched.

Global batch128, 500000-step target, seed42, seven tasks x100 demonstrations,
loss weights, LR schedules, clip0.5 and update guards are unchanged. No419h
policy or previous divergent policy was used to initialize these continuations.

World-size changes necessarily alter the per-rank stochastic trajectory.
The trainer now rejects mismatched saved RNG-rank counts unless explicitly
allowed; a migration deterministically seeds rank r with
`seed + 1000003 * saved_step + r` and logs the transition. Once a new checkpoint
has the current number of ranks, subsequent resumes restore saved CPU/CUDA
states normally. Four/eight-rank samplers cover the same128 absolute sample
ordinals per logical update. This preserves sampling/optimizer progress,
not bitwise equivalence across different world sizes.

Use `outputs/arena_sonicx7_current.json` as the active-run entry point after
the migration is finalized. Historical runs stay in place and are marked
with their successor. The supervisor still schedules the same three matched
HOI evaluations after500k, including the restored original scratch baseline.
