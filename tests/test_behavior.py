"""Tests for the behavior track: tp (threshold crossing) and tp-vs-ts slope (plan 2.2).

Numpy-only — no torch/neurogym needed. Run from the repo root::

    python tests/test_behavior.py        # plain asserts, no pytest needed
    pytest tests/test_behavior.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.behavior.slope import slopes_by_prior, tp
from src.training.config import Config


def _ramp_trace(cfg, set_step, cross_step, n_time):
    """0 up to ``set_step``, then a linear ramp reaching ``cfg.threshold`` exactly at
    index ``cross_step`` and staying above afterwards."""
    z = np.zeros(n_time, dtype=float)
    ramp = np.linspace(0.0, cfg.threshold, cross_step - set_step + 1)
    z[set_step:cross_step + 1] = ramp
    z[cross_step + 1:] = cfg.threshold * 1.5
    return z


def test_tp_known_crossing():
    cfg = Config.reduced()                       # dt=5, threshold=1.0, pulse_width_step=17
    set_step, cross_step = 100, 180
    z = _ramp_trace(cfg, set_step, cross_step, cfg.n_steps)
    got = tp(z, set_step, cfg)
    expected = (cross_step - set_step) * cfg.dt
    assert abs(got - expected) < cfg.dt, f"tp {got} != {expected}"
    print("test_tp_known_crossing OK")


def test_tp_no_crossing():
    cfg = Config.reduced()
    z = np.zeros(cfg.n_steps, dtype=float)       # never reaches threshold
    assert np.isnan(tp(z, 100, cfg))
    print("test_tp_no_crossing OK")


def test_tp_skips_set_transient():
    cfg = Config.reduced()
    set_step, cross_step = 100, 200
    z = np.zeros(cfg.n_steps, dtype=float)
    z[set_step + 1] = cfg.threshold + 2.0        # transient inside the skip window
    z[cross_step:] = cfg.threshold + 0.5         # the real crossing
    got = tp(z, set_step, cfg)
    expected = (cross_step - set_step) * cfg.dt
    assert abs(got - expected) < cfg.dt, f"tp used the transient: {got} vs {expected}"
    print("test_tp_skips_set_transient OK")


def test_tp_batched():
    cfg = Config.reduced()
    set_step = 100
    z0 = _ramp_trace(cfg, set_step, 160, cfg.n_steps)
    z1 = _ramp_trace(cfg, set_step, 220, cfg.n_steps)
    got = tp(np.stack([z0, z1]), set_step, cfg)
    assert got.shape == (2,)
    assert abs(got[0] - (160 - 100) * cfg.dt) < cfg.dt
    assert abs(got[1] - (220 - 100) * cfg.dt) < cfg.dt
    print("test_tp_batched OK")


def test_slopes_by_prior_recovers_slope():
    # synthetic tp = 0.7 * ts + 50 per prior -> slope must come back ~0.7
    ts = [480, 560, 640, 720, 800, 800, 900, 1000, 1100, 1200]
    labels = ["short"] * 5 + ["long"] * 5
    tps = [0.7 * t + 50 for t in ts]
    out = slopes_by_prior(ts, tps, labels)
    assert abs(out["short"] - 0.7) < 1e-6 and abs(out["long"] - 0.7) < 1e-6
    print("test_slopes_by_prior_recovers_slope OK")


def test_slopes_drops_nan_and_needs_three():
    ts = [480, 560, 640, 720, 800]
    labels = ["short"] * 5
    two_valid = [np.nan, np.nan, np.nan, 600.0, 650.0]     # < 3 finite -> NaN
    assert np.isnan(slopes_by_prior(ts, two_valid, labels)["short"])
    three_valid = [np.nan, 500.0, 550.0, 600.0, np.nan]    # 3 finite -> finite slope
    assert np.isfinite(slopes_by_prior(ts, three_valid, labels)["short"])
    print("test_slopes_drops_nan_and_needs_three OK")


def main():
    test_tp_known_crossing()
    test_tp_no_crossing()
    test_tp_skips_set_transient()
    test_tp_batched()
    test_slopes_by_prior_recovers_slope()
    test_slopes_drops_nan_and_needs_three()
    print("\nall behavior tests passed")


if __name__ == "__main__":
    main()
