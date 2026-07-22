"""Tests for the task track: make_batch / build_trial (plan 1.A).

These run against the standalone numpy generator (Blocker A fallback) — no neurogym
import — so they run unconditionally. If the module is ever swapped for the NeuroGym
subclass and neurogym is unavailable, gate with a skip-when-unavailable guard instead.

Run from the repo root::

    python tests/test_task.py        # plain asserts, no pytest needed
    pytest tests/test_task.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.behavior.slope import tp
from src.conditions import Condition
from src.task.rsg import build_trial, make_batch, ramp
from src.training.config import Config


def _set_step_from_mask(mask_row):
    """First index where the production mask turns on == Set step."""
    return int(np.argmax(mask_row > 0))


def test_build_trial_shapes_and_channels():
    cfg = Config.reduced()
    cond = Condition("short", 640, "eye")
    inputs, set_step = build_trial(cfg, cond)                 # jitter=False
    assert inputs.shape == (1, cfg.n_steps, 3)
    ch0, ch1, ch2 = inputs[0, :, 0], inputs[0, :, 1], inputs[0, :, 2]
    # context channels are tonic (constant across the trial)
    assert np.allclose(ch1, cfg.prior_context["short"])
    assert np.allclose(ch2, cfg.effector_context["eye"])
    # two pulses on ch0: Ready at ready_onset_step, Set at ready_onset + ts (no jitter)
    r0, pw = cfg.ready_onset_step, cfg.pulse_width_step
    ts_steps = round(cond.ts / cfg.dt)
    assert np.allclose(ch0[r0:r0 + pw], cfg.pulse_height)
    assert np.allclose(ch0[set_step:set_step + pw], cfg.pulse_height)
    assert set_step == r0 + ts_steps
    print("test_build_trial_shapes_and_channels OK")


def test_make_batch_shapes_target_mask():
    cfg = Config.reduced()
    B = 16
    batch = make_batch(cfg, B, np.random.default_rng(0))
    assert batch.inputs.shape == (B, cfg.n_steps, 3)
    assert batch.target.shape == (B, cfg.n_steps)
    assert batch.mask.shape == (B, cfg.n_steps)
    assert len(batch.conditions) == B
    for i, cond in enumerate(batch.conditions):
        ts_steps = round(cond.ts / cfg.dt)
        set_step = _set_step_from_mask(batch.mask[i])
        assert batch.mask[i, :set_step].sum() == 0            # mask 0 before Set
        assert batch.mask[i, set_step] == 1.0                 # mask 1 at Set
        cross = set_step + ts_steps
        # target crosses threshold exactly ts_steps after Set (encodes the ramp choice)
        assert batch.target[i, cross] >= cfg.threshold - 1e-5
        assert batch.target[i, cross - 1] < cfg.threshold
        assert np.allclose(batch.inputs[i, :, 1], cfg.prior_context[cond.prior])
        assert np.allclose(batch.inputs[i, :, 2], cfg.effector_context[cond.effector])
    print("test_make_batch_shapes_target_mask OK")


def test_ramp_holds_above_threshold_after_reconciling_ramp_A():
    """Regression for docs/RUNBOOK.md Gap #2.

    Before this fix, the hold segment (t > ts) plateaued at exactly cfg.threshold,
    so a well-trained network's tp validity was decided by noise at the margin --
    confirmed directly: a converged BPTT run's peak output landed at 0.993-0.996
    for 15/20 conditions, just under threshold, despite clearly having learned the
    task. The hold must now rise to and settle at cfg.ramp_A (> cfg.threshold),
    while the approach segment still crosses exactly cfg.threshold at t=ts (that
    calibration is load-bearing for tp-vs-ts and must not move).
    """
    cfg = Config.reduced()
    assert cfg.ramp_A > cfg.threshold, "fixture assumption: ramp_A configured above threshold"
    ts_steps = 100

    at_ts = ramp(np.array([ts_steps], dtype=float), ts_steps, cfg)[0]
    assert abs(float(at_ts) - cfg.threshold) < 1e-6, at_ts

    at_hold_end = ramp(np.array([ts_steps + cfg.prod_hold_step], dtype=float), ts_steps, cfg)[0]
    assert abs(float(at_hold_end) - cfg.ramp_A) < 1e-6, at_hold_end

    past_hold = ramp(np.array([ts_steps + cfg.prod_hold_step + 50], dtype=float), ts_steps, cfg)[0]
    assert abs(float(past_hold) - cfg.ramp_A) < 1e-6, past_hold

    mid_hold = ramp(np.array([ts_steps + cfg.prod_hold_step // 2], dtype=float), ts_steps, cfg)[0]
    assert cfg.threshold < float(mid_hold) < cfg.ramp_A, mid_hold
    print("test_ramp_holds_above_threshold_after_reconciling_ramp_A OK")


def test_target_tp_consistency():
    # tp() on the target ramp must recover ~ts — ties the task and behavior modules.
    cfg = Config.reduced()
    batch = make_batch(cfg, 8, np.random.default_rng(1))
    for i, cond in enumerate(batch.conditions):
        set_step = _set_step_from_mask(batch.mask[i])
        got = tp(batch.target[i], set_step, cfg)
        assert abs(got - cond.ts) < cfg.dt, f"tp(target)={got} vs ts={cond.ts}"
    print("test_target_tp_consistency OK")


def test_jitter_moves_gap_but_target_timed_to_true_ts():
    cfg = Config.reduced()
    r0 = cfg.ready_onset_step
    # no jitter: the Ready->Set gap equals the true ts
    cond = Condition("long", 1000, "hand")
    _, set_step0 = build_trial(cfg, cond, jitter=False)
    assert set_step0 - r0 == round(cond.ts / cfg.dt)
    # jitter (make_batch): input gaps move off the true ts, yet the target still crosses
    # threshold exactly true-ts steps after each trial's Set.
    batch = make_batch(cfg, 64, np.random.default_rng(7))
    gap_minus_ts = []
    for i, c in enumerate(batch.conditions):
        ts_steps = round(c.ts / cfg.dt)
        set_step = _set_step_from_mask(batch.mask[i])
        gap_minus_ts.append((set_step - r0) - ts_steps)
        assert batch.target[i, set_step + ts_steps] >= cfg.threshold - 1e-5
    assert np.any(np.array(gap_minus_ts) != 0)                # jitter actually moved Set
    print("test_jitter_moves_gap_but_target_timed_to_true_ts OK")


def test_determinism_given_rng():
    cfg = Config.reduced()
    b1 = make_batch(cfg, 8, np.random.default_rng(3))
    b2 = make_batch(cfg, 8, np.random.default_rng(3))
    assert np.array_equal(b1.inputs, b2.inputs)
    assert np.array_equal(b1.target, b2.target)
    assert np.array_equal(b1.mask, b2.mask)
    assert b1.conditions == b2.conditions
    print("test_determinism_given_rng OK")


def test_longest_condition_fits_canvas():
    # Blocker B regression: long/1200's full production epoch + hold fits n_steps.
    for maker in (Config.reduced, Config.faithful):
        cfg = maker()
        cond = Condition("long", 1200, "eye")
        _, set_step = build_trial(cfg, cond)
        prod_end = set_step + round(cond.ts / cfg.dt) + cfg.prod_hold_step
        assert prod_end <= cfg.n_steps, (
            f"{maker.__name__}: prod_end {prod_end} > n_steps {cfg.n_steps}"
        )
    print("test_longest_condition_fits_canvas OK")


def test_facade_selector_and_errors():
    # The src.task facade selects the backend from cfg.task_source (default neurogym) and
    # validates it. active_backend never needs neurogym; an invalid source raises early.
    from src.task import active_backend, make_batch as facade_make_batch
    assert active_backend(Config.reduced()) == "neurogym"           # default
    assert active_backend(Config.reduced(task_source="standalone")) == "standalone"
    try:
        facade_make_batch(Config.reduced(task_source="bogus"), 2, np.random.default_rng(0))
        raise AssertionError("expected ValueError for unknown task_source")
    except ValueError:
        pass
    print("test_facade_selector_and_errors OK")


def test_facade_dispatch_standalone():
    # cfg.task_source="standalone" -> facade routes to src.task.rsg (this module's make_batch).
    from src.task import make_batch as facade_make_batch, build_trial as facade_build_trial
    cfg = Config.reduced(task_source="standalone")
    a = make_batch(cfg, 8, np.random.default_rng(0))                 # direct (standalone)
    b = facade_make_batch(cfg, 8, np.random.default_rng(0))          # via facade
    assert a.conditions == b.conditions
    assert np.array_equal(a.inputs, b.inputs)
    assert np.array_equal(a.target, b.target) and np.array_equal(a.mask, b.mask)
    ia, sa = build_trial(cfg, Condition("short", 640, "eye"))
    ib, sb = facade_build_trial(cfg, Condition("short", 640, "eye"))
    assert sa == sb and np.array_equal(ia, ib)
    print("test_facade_dispatch_standalone OK")


def main():
    test_build_trial_shapes_and_channels()
    test_make_batch_shapes_target_mask()
    test_target_tp_consistency()
    test_jitter_moves_gap_but_target_timed_to_true_ts()
    test_determinism_given_rng()
    test_longest_condition_fits_canvas()
    test_facade_selector_and_errors()
    test_facade_dispatch_standalone()
    print("\nall task tests passed")


if __name__ == "__main__":
    main()
