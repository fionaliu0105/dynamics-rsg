# PC model: it trains fine, but never "presses go" — needs the team's eyes

> **Status (2026-07-21, later that day): partially superseded.** This note describes
> the state *before* merging `pranavpata`'s `pc_rnn_2` fix (normalize+clip, plus
> routing PC through the same Adam optimizer as BPTT — the update-scale/`J`-frozen
> problem described below). A fresh validation run against the merged fix is in
> progress; see `.suplex/docs/discrepancy_log.md` for the live status. Separately,
> `docs/RUNBOOK.md` ("Gaps" #2) documents a second, likely-contributing issue: the
> task's target ramp caps *exactly* at `cfg.threshold` rather than above it, which
> makes threshold-crossing noise-decided regardless of the learning rule. Read this
> note as historical context for the images below (archived under
> `results/figures/pc_activity_pre_merge_2026-07-21/`), not the current state.

**TL;DR:** The predictive-coding (PC) network trains cleanly on the real RSG task now (a bug that used to blow it up is fixed). But when we check whether it actually produces a timed response, it never crosses the response threshold, on any of the 20 task conditions. We don't yet know why, and it's a good problem to look at together.

## What we were checking

After training, we run the model on all 20 task conditions (2 priors x 2 effectors x 5 sample intervals each) with noise off, and look at its output over time. The model is supposed to ramp its output up and hit a threshold (`z = 1.0`) right around the moment it thinks the timed interval has elapsed — that's the "produced interval," the thing we compare to the monkey's behavior.

## What we found

Training itself looks good: the loss (how far the output is from the target curve, averaged over the whole trial) dropped to essentially zero. So on average, over the whole trial, the model's output is a very close match to what we want.

But when we zoom into just the part that matters — the response near threshold — something odd shows up. In every single condition, the output rises, then **flattens out below the threshold** and never crosses it. And the flattened value it lands on doesn't depend on the sample interval at all — it only depends on which prior (short/long) and which effector (eye/hand) is active:

| context | how close it gets (threshold = 1.0) |
|---|---|
| short prior, eye | 0.96 |
| short prior, hand | 0.93 |
| long prior, hand | 0.63 |
| long prior, eye | 0.59 |

Short-prior trials get close but don't quite make it. Long-prior trials plateau well short of threshold. And critically: within each row, that number is *identical* whether the interval was the shortest or longest in that prior's range — it looks like the network is settling into a fixed resting level per (prior, effector), not actually producing a time-locked response.

Because nothing crosses threshold, we can't compute a produced interval or the bias slope (the "regresses toward the prior mean" signature we actually care about) for any condition yet.

## What we don't know yet

Two live hypotheses, neither confirmed:

1. **The recent stability fix might be too conservative.** We found and fixed a real bug where PC's training updates could blow up to huge, broken values. The fix caps the total size of each update. It's possible that same cap is now also holding back the output weights from growing large enough to actually reach threshold.
2. **Something structural in the dynamics.** It's possible the network's recurrent activity is settling into a resting state that caps the output, independent of how long it's supposed to keep ramping.

Worth someone digging into whether the output weights are still growing at all by the end of training, and whether loosening the update cap changes this.

## Where to look

- The plot that shows this most clearly (archived, pre-merge): `results/figures/pc_activity_pre_merge_2026-07-21/pc_output_vs_target.png` (solid = model output, dashed = target, dotted = threshold).
- Full numbers (pre-merge run): `results/runs/pc/seed_0000/metrics.json`.
- The tutorial notebook walking through all of this end to end: `notebooks/pc_tutorial.ipynb` (now repointed at the post-merge validation run, `results/runs_v2/pc/seed_0000/`).
- The fix in question at the time this note was written: `src/models/pc_rnn.py`'s `_clip_updates`/normalization (since superseded by `pc_rnn_2`'s `_rescale_updates` + Adam-routing in `src/training/trainer.py`).
- The likely-contributing task-definition issue: `docs/RUNBOOK.md`, "Gaps" #2 (`cfg.ramp_A` unused, target caps exactly at threshold).
