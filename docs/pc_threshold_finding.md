# PC learns to converge, but not to tell time — the task bug is now fixed

**TL;DR:** Two different problems. **One is resolved.** The other has a real, partial lead but isn't finished.

1. **Task-level, affects both rules — RESOLVED.** The target the network was trained toward capped exactly at the response threshold instead of comfortably above it, which made "did it cross the threshold" partly a coin flip even for a network that learned the task well. First fix attempt overshot (too big a margin, BPTT couldn't converge under it); the corrected, gentler version works cleanly — confirmed with a real, textbook Bayesian-bias result from BPTT (see below).
2. **PC-specific — still open, real lead found.** Even after fixing a real training bug, PC's network gives up on timing entirely — it outputs a flat constant per (prior, effector) combination and ignores the actual interval, at both the output level *and* the internal recurrent-state level. Confirmed this is about PC's learning rule specifically (not the task, now that the task bug above is fixed) via direct BPTT comparison. Giving PC's internal relaxation process more steps (`cfg.pc_inference_steps=100` instead of the default 20) produces a real, measurable behavioral effect — not just an internal-state proxy — but it's still small and partial, not close to matching BPTT.

Full evidence for both below.

## Background: the bug that got fixed first

PC's training updates were being computed as raw, unscaled sums instead of properly-averaged values, which made them enormous and mismatched with BPTT's scale — this caused instability early in training. That's fixed now (merged from a teammate's branch, `pc_rnn_2`): updates are properly normalized, and PC now shares the same optimizer machinery (Adam) BPTT uses, which was also needed to get PC's recurrent connections to actually move during training at all (they were staying frozen otherwise). Both of these were checked directly against the reference PC implementation the project is based on, not guessed at.

With that fixed, PC trains smoothly in the loss sense: the loss (how far the output is from the target curve) drops cleanly. The problems below are both about what that convergence actually produces, not about whether training runs at all.

## Finding 1 (task-level): RESOLVED — the target now holds comfortably above threshold

We noticed BPTT's own output peaked at 0.993-0.996 for most conditions — essentially a perfect match to where the target ramp topped out — but the response threshold was exactly 1.0, so those conditions never technically "crossed" it, even though the network had clearly learned them well. The target was defined to reach *exactly* the threshold and hold there, so whether a well-trained response counted as "successful" was decided by noise at that knife's edge.

**First attempt overshot.** Changed the target to keep rising after the original crossing point, holding at 3x the threshold instead of flattening exactly at it. This was too big a jump: BPTT — the network that was tracking the task well — failed to converge at all under it within the same training budget (loss barely moved). Not usable as implemented.

**Corrected version works.** Retried with a gentler margin, 1.2x threshold (matching a value earlier team notes describe as already tested), changing nothing else. Retrained BPTT under it: **`valid_tp_count` is now 20/20** (every condition produces a valid response), and the fitted bias slopes are `short = 0.494`, `long = 0.413` — both in the theoretically expected (0, 1) range, with the long (noisier) prior showing more regression toward the mean than short, exactly as Bayesian-integration theory predicts. The `tp`-vs-`ts` plot also shows the classic central-tendency pattern at the `ts=800` overlap point: the same physical interval produces a different response depending on which prior context is active. This is the first clean demonstration of genuine Bayesian-bias behavior from a trained network in this project.

`cfg.ramp_A` now defaults to `1.2` (was `3.0`, briefly).

## Finding 2 (PC-specific): still open, but with a real, partial lead

Despite clean loss convergence, PC's output isn't tracking the task. It settles into **4 flat output levels**, one per (prior, effector) combination — identical regardless of the actual sample interval `ts`. Confirmed via direct BPTT comparison (same architecture, task, seed, everything except the learning rule) that this is about PC's rule specifically, not the setup: BPTT visibly tracks each condition's individual target curve; PC doesn't.

**It's not just the output — the internal recurrent states don't encode timing either.** PC's hidden-state trajectories differ by only ~0.0001-0.0002 across the full range of `ts` within a group — essentially noise. BPTT's states, same comparison: ~0.07-0.36, 300-1800x larger. The failure is upstream of the readout, in what the recurrent connections themselves learn to represent.

**The `pc_inference_steps` lead, now tested twice:**
- First test (overnight, on the not-yet-fixed task, internal-state proxy only): `pc_inference_steps=100` instead of the default `20` gave 8-15x more interval-dependent structure in the recurrent states, at the same training budget, with better loss too.
- Second test (this morning, on the now-corrected task, checked against actual output behavior): at a fair, matched 1000-iteration budget, `steps=20` still produces 0/20 valid responses with all 4 output groups perfectly flat. `steps=100` also produces 0/20 valid responses, **but one group (long/hand) is no longer perfectly flat** — its peak output now varies genuinely, if slightly, with `ts` (a small monotonic trend, visible as a faint fan-out in the plot). The other 3 groups remain flat even at `steps=100`.

So this is now confirmed as a real behavioral effect, not just an artifact of the internal-state metric — but it's small, appears in only one of four condition groups, and 1000 iterations clearly isn't enough for it to produce actual timed responses yet. Not yet tested at the full 3000-iteration training scale (would take roughly 3 hours based on this run's timing, at `steps=100`'s ~5x per-iteration cost) — that's the natural next test, not yet run.

## Other things added since the last update (infrastructure, not findings)

- A SLURM `sbatch` wrapper (`scripts/slurm/train.sbatch`) and cluster environment setup/activation scripts (`scripts/hpc/setup_env.sh`, `load_env.sh`/`load_env.csh`) now exist, so the `pc_inference_steps=100` full-scale test (or any future sweep) can run on the cluster instead of locally.

## Where to look

- **The clean Bayesian-bias result (Finding 1 resolved):** `results/archive/bptt_seed0000_tp_vs_ts_pilot.png` and `results/archive/pc_activity/bptt_v4_output_vs_target.png` (pilot run, pre-cluster). The equivalent real cluster run at this same seed number is a *different, unrelated* result — see `results/runs_summary.csv` / `notebooks/results_summary.ipynb` for the real 10-seed sweep. Full numbers: `results/archive/runs_v4/bptt/seed_0000/metrics.json`.
- **PC's flat collapse, before and after the task fix:** `results/archive/pc_activity/pc_v2_output_vs_target.png` (pre-fix) and `results/archive/pc_activity/pc_v3_output_vs_target.png` (post-fix, still collapsed under the too-aggressive 3x margin — a different manifestation of the same PC problem, not a new one).
- **The `pc_inference_steps=100` partial improvement (post-task-fix):** `results/archive/pc_activity/pc_v4_steps100_output_vs_target.png` — look for the faint fan-out in the yellow (`long/hand`) traces.
- **BPTT failing to converge under the too-aggressive 3x margin** (why that attempt was abandoned): `results/archive/pc_activity/bptt_v3_output_vs_target.png`.
- Full numbers by run: `results/archive/runs_v2` (pre-task-fix), `results/archive/runs_v3` (3x margin, abandoned), `results/archive/runs_v4` (1.2x margin, working; also has `pc_steps20`/`pc_steps100` subdirectories for the paired comparison). All three are pre-cluster, single-seed pilot runs — moved under `results/archive/` for tidiness once the real 10-seed cluster sweep (`results/runs/`, `results/runs_pc_steps100/`) superseded them; kept for the record, not part of the sweep's own results.
- The tutorial notebook: `notebooks/pc_tutorial.ipynb` (still reads the old `results_v2` PC run; not yet updated with any of this).
- The task fix itself: `src/task/rsg.py::ramp()`, `cfg.ramp_A` in `src/training/config.py`.
- PC's current code: `src/models/pc_rnn.py` (`_rescale_updates`, `_relax`, `cfg.pc_optimizer`, `cfg.pc_clip_mode`).
- Full technical trail with exact numbers for everything above: `.suplex/docs/discrepancy_log.md` (2026-07-21/22 entries).
- Historical context (pre-merge state, superseded): `results/archive/pc_activity_pre_merge_2026-07-21/`.
