"""Smoke tests for the Preprocessor (plan 1.E).  [Preprocess & RSA track]

Numpy-only, plain-asserts — no torch/rsatoolbox/scipy needed (they are ABI-broken in
the base env). Mirrors tests/test_foundation.py::

    python tests/test_preprocess.py        # plain-asserts, no pytest needed
    pytest tests/test_preprocess.py         # also works if pytest is installed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.conditions import N_CONDITIONS
from src.preprocess.pipeline import PreprocessConfig, Preprocessor


def _random_system(rng, time, units):
    return rng.standard_normal((N_CONDITIONS, time, units))


def test_output_shape():
    rng = np.random.default_rng(0)
    cfg = PreprocessConfig(k=10, n_time_bins=25)
    pre = Preprocessor(cfg).fit(_random_system(rng, 37, 64))
    out = pre.transform(_random_system(rng, 37, 64))
    assert out.shape == (N_CONDITIONS, 25, 10)
    print("test_output_shape OK")


def test_identical_dims_across_differently_shaped_systems():
    # THE headline invariant: two systems with different time AND unit counts, put
    # through the SAME fitted Preprocessor, exit with identical (n_time_bins, k).
    rng = np.random.default_rng(1)
    cfg = PreprocessConfig(k=8, n_time_bins=20)
    pre = Preprocessor(cfg).fit(_random_system(rng, 30, 200))
    a = pre.transform(_random_system(rng, 30, 200))     # model-like: many units
    b = pre.transform(_random_system(rng, 45, 54))      # neural-like: 54 units
    assert a.shape == b.shape == (N_CONDITIONS, 20, 8)
    print("test_identical_dims_across_differently_shaped_systems OK")


def test_variable_length_conditions():
    # Ragged input: each condition has a different trial length (varying ts/tp).
    rng = np.random.default_rng(2)
    conds = [rng.standard_normal((10 + 3 * i, 16)) for i in range(N_CONDITIONS)]
    system = np.array(conds, dtype=object)
    cfg = PreprocessConfig(k=6, n_time_bins=25)
    pre = Preprocessor(cfg).fit(system)
    out = pre.transform(system)
    assert out.shape == (N_CONDITIONS, 25, 6)
    print("test_variable_length_conditions OK")


def test_k_greater_than_units_zero_padded():
    rng = np.random.default_rng(3)
    cfg = PreprocessConfig(k=10, n_time_bins=12)
    pre = Preprocessor(cfg).fit(_random_system(rng, 20, 4))     # only 4 units
    out = pre.transform(_random_system(rng, 20, 4))
    assert out.shape == (N_CONDITIONS, 12, 10)
    # With units=4, at most 4 PCs carry signal; columns 4.. must be exact zeros.
    assert np.allclose(out[:, :, 4:], 0.0)
    print("test_k_greater_than_units_zero_padded OK")


def test_dead_unit_no_nan():
    rng = np.random.default_rng(4)
    system = _random_system(rng, 20, 8)
    system[:, :, 3] = 1.7                              # a constant (dead) unit
    cfg = PreprocessConfig(k=5, n_time_bins=15)
    pre = Preprocessor(cfg).fit(system)
    out = pre.transform(system)
    assert np.isfinite(out).all()
    print("test_dead_unit_no_nan OK")


def test_zscore_removes_offset_and_scale():
    # A per-unit affine transform (offset + scale) must not change the geometry, since
    # z-score normalizes it away. RDMs of the two systems should match.
    from src.compare.rsa import build_rdm

    rng = np.random.default_rng(5)
    base = _random_system(rng, 30, 12)
    offset = rng.standard_normal((1, 1, 12)) * 5.0
    scale = np.abs(rng.standard_normal((1, 1, 12))) + 0.5
    shifted = base * scale + offset
    cfg = PreprocessConfig(k=6, n_time_bins=20)
    pre = Preprocessor(cfg)
    rdm_base = build_rdm(pre.fit(base).transform(base))
    rdm_shift = build_rdm(pre.fit(shifted).transform(shifted))
    assert np.allclose(rdm_base, rdm_shift, atol=1e-6)
    print("test_zscore_removes_offset_and_scale OK")


def test_inputs_warp_time_aligned():
    # transform_inputs must put inputs on the same time base as states, with the SAME
    # warp: a pulse at the same normalized position lands in the same bin.
    rng = np.random.default_rng(6)
    n_in = 3
    inputs = []
    for i in range(N_CONDITIONS):
        t = 20 + i                                    # variable length
        u = np.zeros((t, n_in))
        u[t // 2, 0] = 1.0                            # pulse at the midpoint
        inputs.append(u)
    system = np.array(inputs, dtype=object)
    cfg = PreprocessConfig(k=4, n_time_bins=25)
    pre = Preprocessor(cfg).fit(system)
    warped = pre.transform_inputs(system)
    assert warped.shape == (N_CONDITIONS, 25, n_in)
    # midpoint pulse -> peak near the middle bin for every condition
    peak_bins = warped[:, :, 0].argmax(axis=1)
    assert np.all(np.abs(peak_bins - 12) <= 1)
    print("test_inputs_warp_time_aligned OK")


def test_transform_before_fit_raises():
    rng = np.random.default_rng(7)
    pre = Preprocessor(PreprocessConfig())
    try:
        pre.transform(_random_system(rng, 10, 8))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError when transform precedes fit")
    print("test_transform_before_fit_raises OK")


def test_wrong_condition_count_raises():
    rng = np.random.default_rng(8)
    bad = rng.standard_normal((N_CONDITIONS - 1, 10, 8))
    pre = Preprocessor(PreprocessConfig())
    try:
        pre.fit(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for wrong condition count")
    print("test_wrong_condition_count_raises OK")


def main():
    test_output_shape()
    test_identical_dims_across_differently_shaped_systems()
    test_variable_length_conditions()
    test_k_greater_than_units_zero_padded()
    test_dead_unit_no_nan()
    test_zscore_removes_offset_and_scale()
    test_inputs_warp_time_aligned()
    test_transform_before_fit_raises()
    test_wrong_condition_count_raises()
    print("\nall preprocess tests passed")


if __name__ == "__main__":
    main()
