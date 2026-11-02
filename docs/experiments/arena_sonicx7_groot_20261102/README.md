# Seven-task scratch prelaunch validation, 2026-11-02 UTC

All checks passed before launching the 500,000-update experiment. These files
are validation receipts, **not formal success-rate results**.

- Data: 700 demonstrations, 293,864 windows, seven tasks × 100, with exact
  original-loader episode/cut agreement at ten ordinals through 63,999,999.
- Real mixed batch: 32 samples containing all seven tasks; 98–109 valid tokens;
  zero-injection max error 0; stage 1/2 backward and checkpoint round-trip pass;
  detached Root→Body and Body→Hand boundaries pass; one VLM call per plan.
- Four GPUs 4/5/6/7: production batch 128, micro batch 32, 24 workers/GPU,
  compiled forward. Test-only adapter-unfreeze boundary shortened to update 3;
  train to 6, resume and safely stop at 7, resume to 10. All saved parameters
  have identical SHA256 on all four ranks at all three checkpoints. Adam steps
  are 10 for injections/HandNet, 7 for the newly unfrozen adapter. Formal
  training starts fresh and uses the full 10,000-update stage boundary.
- Simulator: Football, DoubleDesk and P&P Box each completed a real 30-step
  smoke with two successful policy requests and no server failures. Original
  scratch 500k baseline also passed the same Football smoke. Smoke terminations
  do not estimate success rate.
- Regression: 71 passed, 1 skipped (BiGym not installed), 2 deselected known
  pre-existing SIMPLE36D parity failures reproduced at the original base.

`resolved_config.json` is the formal training recipe; `original_recipe.yaml`
is the local original Kimodo reference. `source_sha256.json` identifies tested
training, conditioning, evaluation, monitoring and launch files. See
`../../benchmarks/arena_sonicx7_groot.md` for the protocol and Table 2 references.
Checkpoints and datasets remain local and are not included in Git.

## Launched run

Formal training started at `2026-11-02T10:47 UTC` using source commit `27ed857`
on `model_improvement_lct1`. Pipeline PID: `1331290`; monitor PID: `1331413`.
Run: `log/arena_sonicx7_groot/scratch_ddp4_20261102_104707` (the worktree's log
directory links to shared local storage). `launch_receipt.json` records the
configuration, first 100 updates and verified `checkpoint_100`; the trainer
had reached update 170 when this receipt was collected. All initial losses and
gradient norms were finite, with no monitor alerts. xiancun remained supervised
and yielded its compute worker while the GPUs were busy.

`initial_training_curves.png` is a launch snapshot, not the live chart. The
live run's `monitor/dashboard.html`, `training_curves.png`, `training_curves.pdf`,
`metrics.csv` and `health.json` refresh every minute. Formal success rates are
not yet available. `formal_protocol.json` also verifies all three final
evaluation commands, episode counts, task-specific limits and matching assets.
