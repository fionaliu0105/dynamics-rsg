# PC learns to converge, but not to tell time — and there's a separate task bug too

**TL;DR:** Two different problems, not one. Overnight follow-up work found a promising lead on the first, and found that the first attempt at fixing the second didn't work — both logged honestly below, neither fully solved yet.

1. **PC-specific:** even after fixing a real training bug, PC's network gives up on timing entirely — it outputs a flat constant per (prior, effector) combination and ignores the actual interval, at both the output level *and* the internal recurrent-state level. We know this is about PC's learning rule specifically, not the setup, because BPTT trained on the exact same everything and did learn real timing. **Overnight lead:** giving PC's internal relaxation process more steps (`cfg.pc_inference_steps`, 100 instead of the default 20) produces 8-15x more interval-dependent internal structure at the same training budget — a real, validated direction, but not yet shown to close the gap with BPTT.
2. **Task-level, affects both rules:** separately, the target the network is trained toward caps exactly at the response threshold instead of comfortably above it, which makes "did it cross the threshold" partly a coin flip even for a network that learned the task well. **Overnight attempt:** raised the target's hold value well above threshold to fix this — but the margin chosen (3x threshold) turned out to be too big a jump for the current training budget: even BPTT failed to converge under it. Not resolved; needs a gentler value or more training iterations, tried next.

Both are explained in full below, with the evidence for each.

## Background: the bug that got fixed first

PC's training updates were being computed as raw, unscaled sums instead of properly-averaged values, which made them enormous and mismatched with BPTT's scale — this caused instability early in training. That's fixed now (merged from a teammate's branch, `pc_rnn_2`): updates are properly normalized, and PC now shares the same optimizer machinery (Adam) BPTT uses, which was also needed to get PC's recurrent connections to actually move during training at all (they were staying frozen otherwise). Both of these were checked directly against the reference PC implementation the project is based on, not guessed at.

With that fixed, PC trains smoothly: the loss (how far the output is from the target curve) drops from about 0.19 to 0.0003 — a very clean convergence.

## Finding 1: PC converges, but doesn't learn to tell time

Despite that clean convergence, when we look at what the network actually outputs per condition, it isn't tracking the task at all. It settles into exactly **4 flat output levels**, one for each combination of prior (short/long) and effector (eye/hand) — and within each of those 4 groups, the output is *identical* regardless of the actual sample interval (`ts`, ranging 480-1200ms depending on prior). The network isn't ramping toward the target over time; it's just picking a constant answer based on which tonic context signal is present, and ignoring the actual timed input pulses entirely.

**How we know this is about PC specifically, not the task or the setup:** we trained BPTT — the other learning rule this project compares against — on the *exact same* config, task, and random seed, changing nothing except which learning rule was used. BPTT's output visibly tracks each condition's individual target curve, rising at different times for different intervals, closely matching the target shape. Same architecture, same task, same discretization settings — BPTT learns real timing, PC doesn't.

**It's not just the output — the internal recurrent states don't encode timing either.** Comparing the network's hidden-state trajectories (not just its final output) across different `ts` values within the same (prior, effector) group: PC's states differ by only ~0.0001-0.0002 across the whole range of intervals — essentially noise. BPTT's states, same comparison: ~0.07-0.36, **300-1800x larger**, scaling clearly with `ts`. This rules out "PC's recurrent states are fine, only the readout is broken" — the failure is upstream, in what the recurrent connections themselves learn to represent.

**A promising, tested (but not fully proven) lead:** PC's recurrent connections only get a learning signal indirectly — the network briefly relaxes its internal state estimates before computing what to update, and only after that relaxation does the interval information have a chance to reach the recurrent weights. With the default relaxation depth (20 steps), we suspected this indirect path was too short. Tested directly: running the same setup with 100 relaxation steps instead of 20 (same 300 training iterations either way) gave 8-15x more interval-dependent structure in the recurrent states, and also converged to a much better loss at the same iteration count. This is a real, validated direction — increasing `cfg.pc_inference_steps` measurably helps — but even at 100 steps, the interval-encoding is still 4-40x smaller than BPTT's, so it's a lead to build on, not a finished fix. Not yet tested at full training scale (3000 iterations) or checked against the actual behavioral output (only against the internal-state proxy measurement).

## Finding 2: the task's target design makes "did it cross threshold" unreliable — attempted a fix, it didn't work yet

Separately — this one isn't about PC at all — we noticed something in BPTT's own (first) results. For 15 of 20 conditions, BPTT's output peaked at 0.993-0.996, essentially a perfect match to where the target ramp topped out... but the response threshold is exactly 1.0, so those 15 conditions never technically "crossed" it, even though the network had clearly learned them well.

The reason: the target ramp was defined to reach *exactly* the threshold value and hold there — not comfortably above it. So whether a well-trained network's response counted as "successful" was decided by tiny amounts of noise sitting right at that knife's edge, not by whether it actually learned the timing.

**First fix attempt, tested overnight — didn't work.** Changed the target so it keeps rising after the original crossing point, holding at a much higher ceiling (`cfg.ramp_A = 3.0`, i.e. 3x the response threshold) instead of flattening out exactly at threshold. Retrained both PC and BPTT under this new target. Result: BPTT — the network that was previously tracking the task well — essentially failed to converge at all under the new target within the same training budget (loss barely moved, 1.28 to 0.69, versus dropping to 0.004 under the old target). The margin chosen (3x) was too big a jump in what the network has to reach, given the same amount of training time. This fix is **not working as implemented** and shouldn't be treated as resolved.

**Next things to try, not yet done:** a smaller margin (something like 1.2x threshold instead of 3x — this smaller value was actually the one referenced in earlier team notes as already-tested-and-working) and/or a longer training budget under the current 3x version to see if it's purely a convergence-time issue. Either attempt needs to be checked against BPTT specifically, since PC's own unrelated collapse (Finding 1) makes it an unreliable way to test whether this particular fix works.

## What this means together

- We can't yet compute a meaningful produced-interval / bias-slope measurement for PC, because it isn't attempting the task's timing structure at all (Finding 1) — though we now have a validated lever (relaxation depth) that measurably helps, not yet proven sufficient on its own.
- The task-level threshold-margin problem (Finding 2) is real and confirmed, but the first fix attempt overshot and needs a gentler follow-up before it can be called resolved.

These remain two separate, clearly scoped follow-ups: (1) push further on why PC takes the shortcut instead of learning timing — `pc_inference_steps` is a promising lever, not yet a proven fix — and (2) find a workable `ramp_A` margin (or training budget) that both holds above threshold *and* lets a well-trained network actually converge. Neither should be folded into the other, and neither is finished.

## Where to look

- **PC's flat-output collapse** (pre-`ramp_A`-fix): `results/figures/pc_activity/pc_v2_output_vs_target.png`.
- **BPTT genuinely tracking timing** (pre-`ramp_A`-fix): `results/figures/pc_activity/bptt_v2_output_vs_target.png`.
- **BPTT failing to converge under the `ramp_A=3.0` fix**: `results/figures/pc_activity/bptt_v3_output_vs_target.png` — the key piece of evidence that the fix as implemented doesn't work yet.
- **PC still collapsing under the new target too** (a different manifestation of Finding 1, not a new problem): `results/figures/pc_activity/pc_v3_output_vs_target.png`.
- Full numbers: `results/runs_v2/{pc,bptt}/seed_0000/metrics.json` (pre-fix) and `results/runs_v3/{pc,bptt}/seed_0000/metrics.json` (post-`ramp_A`-fix attempt).
- The `pc_inference_steps` lead: raw numbers in `.suplex/docs/discrepancy_log.md` (2026-07-22 entries); not yet turned into a saved plot.
- The tutorial notebook: `notebooks/pc_tutorial.ipynb` (still reads the `results_v2` PC run; not yet updated with any of the overnight findings).
- The task-level bug and its attempted fix: `docs/RUNBOOK.md`, "Gaps" #2, and `src/task/rsg.py::ramp()`.
- PC's current code: `src/models/pc_rnn.py` (`_rescale_updates`, `_relax`, `cfg.pc_optimizer`, `cfg.pc_clip_mode`).
- Historical context (pre-merge state, superseded): `results/figures/pc_activity_pre_merge_2026-07-21/`.
