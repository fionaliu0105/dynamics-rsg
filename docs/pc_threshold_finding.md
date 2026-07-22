# PC learns to converge, but not to tell time — and there's a separate task bug too

**TL;DR:** Two different, now-confirmed problems, not one:

1. **PC-specific:** even after fixing a real training bug, PC's network gives up on timing entirely — it outputs a flat constant per (prior, effector) combination and ignores the actual interval. We know this is about PC's learning rule specifically, not the setup, because BPTT trained on the exact same everything and did learn real timing.
2. **Task-level, affects both rules:** separately, the target the network is trained toward caps exactly at the response threshold instead of comfortably above it, which makes "did it cross the threshold" partly a coin flip even for a network that learned the task well. We saw this directly in BPTT's own results.

Both are explained below, with the evidence.

## Background: the bug that got fixed first

PC's training updates were being computed as raw, unscaled sums instead of properly-averaged values, which made them enormous and mismatched with BPTT's scale — this caused instability early in training. That's fixed now (merged from a teammate's branch, `pc_rnn_2`): updates are properly normalized, and PC now shares the same optimizer machinery (Adam) BPTT uses, which was also needed to get PC's recurrent connections to actually move during training at all (they were staying frozen otherwise). Both of these were checked directly against the reference PC implementation the project is based on, not guessed at.

With that fixed, PC trains smoothly: the loss (how far the output is from the target curve) drops from about 0.19 to 0.0003 — a very clean convergence.

## Finding 1: PC converges, but doesn't learn to tell time

Despite that clean convergence, when we look at what the network actually outputs per condition, it isn't tracking the task at all. It settles into exactly **4 flat output levels**, one for each combination of prior (short/long) and effector (eye/hand) — and within each of those 4 groups, the output is *identical* regardless of the actual sample interval (`ts`, ranging 480-1200ms depending on prior). The network isn't ramping toward the target over time; it's just picking a constant answer based on which tonic context signal is present, and ignoring the actual timed input pulses entirely.

One of those 4 constants (short prior + eye) happens to sit just above the response threshold, so it technically "produces" a response every time — but the produced time is basically the same (~85-125ms) no matter whether the true interval was 480ms or 800ms. That's not a timed response, it's noise dressed up as one.

**How we know this is about PC specifically, not the task or the setup:** we trained BPTT — the other learning rule this project compares against — on the *exact same* config, task, and random seed, changing nothing except which learning rule was used. BPTT's output visibly tracks each condition's individual target curve, rising at different times for different intervals, closely matching the target shape. Same architecture, same task, same discretization settings — BPTT learns real timing, PC doesn't. That rules out the task or the numerical setup as the explanation; the difference is coming from the learning rule itself.

This is now the real open question for PC: why does it take this shortcut instead of learning the task, now that its training mechanics are correctly matched to BPTT's? Worth digging into rather than assuming it's already understood.

## Finding 2: the task's target design makes "did it cross threshold" unreliable, even for a well-trained network

Separately — this one isn't about PC at all — we noticed something in BPTT's own results. For 15 of the 20 conditions, BPTT's output peaks at 0.993-0.996, essentially a perfect match to where the target ramp tops out... but the response threshold is exactly 1.0, so those 15 conditions never technically "cross" it, even though the network clearly learned them well. Only the 5 conditions where the peak happened to land at 1.002 or so counted as valid responses.

The reason: the target ramp is defined to reach *exactly* the threshold value and hold there — not comfortably above it. So whether a well-trained network's response counts as "successful" ends up decided by tiny amounts of noise sitting right at that knife's edge, not by whether it actually learned the timing. This would affect any network trained on this task, regardless of learning rule.

## What this means together

- We can't yet compute a meaningful produced-interval / bias-slope measurement for PC, because it isn't attempting the task's timing structure at all (Finding 1).
- Even once PC does learn timing, the produced-interval measurement itself needs a task-level fix (Finding 2) before it will reliably reflect what the network actually learned, rather than being a coin flip near threshold.

These are two separate, clearly scoped follow-ups, not one job: (1) figure out why PC takes the shortcut instead of learning timing, and (2) adjust the target ramp so it holds comfortably above threshold rather than exactly at it. Neither should be folded into the other.

## Where to look

- The plot that shows PC's flat-output collapse: `results/figures/pc_activity/pc_v2_output_vs_target.png`.
- The plot that shows BPTT genuinely tracking each condition's timing: `results/figures/pc_activity/bptt_v2_output_vs_target.png`.
- Full numbers: `results/runs_v2/pc/seed_0000/metrics.json` and `results/runs_v2/bptt/seed_0000/metrics.json`.
- The tutorial notebook: `notebooks/pc_tutorial.ipynb` (reads the PC validation run; not yet updated with this comparison).
- The task-level bug: `docs/RUNBOOK.md`, "Gaps" #2 (`cfg.ramp_A` deliberately unused; the target caps exactly at threshold instead of above it).
- PC's current code: `src/models/pc_rnn.py` (`_rescale_updates`, `cfg.pc_optimizer`, `cfg.pc_clip_mode`).
- Historical context (pre-fix state, now superseded): `results/figures/pc_activity_pre_merge_2026-07-21/`.
