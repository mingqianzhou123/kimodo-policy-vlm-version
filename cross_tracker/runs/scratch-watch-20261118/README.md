# Scratch final evaluation watch — 2026-11-18

Last audited: 2026-11-18T22:14:48.978830+08:00. Work remains active; this is not a final result.

- Target: four scratch 200000-step models × two trackers × 60 fixed scenes = 480 valid episodes.
- R4 took over at an idle episode boundary, preserving nine validated R2/R3 episodes. Old scheduler PID 1410037 was terminated only after suspension and a second no-child/no-current-episode check; new scheduler PID 1417347. No active rollout or finetune was terminated.
- New queue: `/root/workspace/humanoidVLA/cross_tracker/runs/scratch-eval-20261118-r4`. It waits for a verified CCT manifest while evaluating the three ready models. CCT final is still pending (latest separately observed step 138800/200000).
- Queue records a heartbeat every five seconds during a rollout; the independent read-only monitor checks every fifteen seconds. Busy GPUs wait. Invalid infrastructure attempts preserve their logs and retry; three failures require investigation and an explicit retry marker. A task failure in a valid completed episode remains in the success-rate denominator.
- No-log-progress limit: 600 seconds. Per-attempt infrastructure wall limit: 2400 seconds after the run folder is created. These limits produce infrastructure errors, not task failures. Stalled attempts terminate only their own new process group.
- Fixed protocol preserved: 3 groups × 20 repeats, paired official hashed scene seeds, FP32, 10 diffusion steps, 15-frame execution, RTC 0, original timing, deterministic Torch, fresh simulator, official YAML, 1450 task steps, 10 post-termination video steps. SONIC uses m2 USD and per-step camera refresh. TWIST2 uses default USD, CPU physics and recorded 38 ms timing; this remains an adapted evaluation, not a published-baseline claim.
- CCT full-training context selection is exactly the SenseCore snapshot: SONIC 81/14/3 and TWIST2 94/35/31 from manifest `940f3cfc1fcc2be86d8ae80c79a8eccc934466132780376d6e91dae2f5797c13`. Baidu pre-existing cached record hashes differed, so six records plus metadata were copied independently from SenseCore, not reused. Seven SHA256 checks and CPU context assembly passed. Final checkpoint loading and GPU serving remain pending.
- CCT serving uses the trained CCT encoder and explicit `--timing original --deterministic`. The three plain models retain their existing serving commands. Both dual-source tracker evaluations are seen-source evaluations, not unseen-source transfer.

Validation: 480 unique tasks, 60 paired seeds and all eight command variants passed local checks. A mocked valid unsuccessful rollout counted as completed; a process failure was excluded and persisted. Remote snapshot contains 144 verified source files. Both CCT contexts assembled finite CPU tensors: visual `[1,128,196,1024]`, transitions `[1,128,110]`, trajectory IDs `[1,128]`, metadata `[1,4096]`.

Evidence: `handoff-audit-2215.json` includes the protocol, source hashes, handoff, nine complete receipts, current runtime commands and monitor state. Byte count 108560; SHA256 `e837e7d0c38448cd39bcacec2d75908e584695b0c7b3b9acd3c90dbc73edcef0`.

Continue watching in the current chat. When CCT final exists, run `verify_cct_final.py` on SenseCore CPU, verify stable finite 200000-step CCT weights, back up independently, relay to Baidu, verify hashes again, and atomically publish `cct-manifest.json`. Do not treat starting the queue as completing this request.
