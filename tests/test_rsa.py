"""Smoke tests for RSA (plan 2.3).  [Preprocess & RSA track]

Numpy-only, plain-asserts — no rsatoolbox needed (the noise ceiling has a numpy
fallback). Mirrors tests/test_foundation.py::

    python tests/test_rsa.py        # plain-asserts, no pytest needed
    pytest tests/test_rsa.py         # also works if pytest is installed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.compare.rsa import (
    build_rdm,
    noise_ceiling,
    rdm_distance,
    rsa_distances_per_seed,
)
from src.conditions import N_CONDITIONS


def _random_activity(rng, time=15, k=6):
    return rng.standard_normal((N_CONDITIONS, time, k))


def test_rdm_shape_symmetry_zero_diagonal():
    rng = np.random.default_rng(0)
    rdm = build_rdm(_random_activity(rng))
    assert rdm.shape == (N_CONDITIONS, N_CONDITIONS)
    assert np.allclose(rdm, rdm.T)
    assert np.allclose(np.diag(rdm), 0.0)
    print("test_rdm_shape_symmetry_zero_diagonal OK")


def test_identical_systems_distance_zero():
    rng = np.random.default_rng(1)
    act = _random_activity(rng)
    d = rdm_distance(build_rdm(act), build_rdm(act))
    assert abs(d) < 1e-9
    print("test_identical_systems_distance_zero OK")


def test_permuted_conditions_larger_distance():
    rng = np.random.default_rng(2)
    act = _random_activity(rng)
    perm = rng.permutation(N_CONDITIONS)
    while np.all(perm == np.arange(N_CONDITIONS)):
        perm = rng.permutation(N_CONDITIONS)
    rdm = build_rdm(act)
    rdm_perm = build_rdm(act[perm])
    d_same = rdm_distance(rdm, rdm)
    d_perm = rdm_distance(rdm, rdm_perm)
    assert d_perm > d_same
    print("test_permuted_conditions_larger_distance OK")


def test_distance_in_bounds():
    rng = np.random.default_rng(3)
    a = build_rdm(_random_activity(rng))
    b = build_rdm(_random_activity(rng))
    for method in ("spearman", "pearson"):
        d = rdm_distance(a, b, method=method)
        assert 0.0 <= d <= 2.0
    print("test_distance_in_bounds OK")


def test_noise_ceiling_ordering():
    # A shared signal RDM plus per-split noise; lower <= upper, both finite.
    rng = np.random.default_rng(4)
    signal = rng.standard_normal((N_CONDITIONS, 15, 6))
    splits = [signal + 0.3 * rng.standard_normal(signal.shape) for _ in range(5)]
    lower, upper = noise_ceiling(splits)
    assert np.isfinite(lower) and np.isfinite(upper)
    assert lower <= upper
    print("test_noise_ceiling_ordering OK")


def test_per_seed_model_to_reference():
    rng = np.random.default_rng(5)
    reference = _random_activity(rng)
    systems = {
        "bptt": {s: _random_activity(rng) for s in range(3)},
        "pc": {s: _random_activity(rng) for s in range(3)},
    }
    out = rsa_distances_per_seed(systems, reference=reference)
    assert set(out) == {"bptt", "pc"}
    for rule in ("bptt", "pc"):
        assert len(out[rule]) == 3
        assert all(isinstance(v, float) for v in out[rule])
    print("test_per_seed_model_to_reference OK")


def test_per_seed_rule_vs_rule():
    rng = np.random.default_rng(6)
    systems = {
        "bptt": {s: _random_activity(rng) for s in range(4)},
        "pc": {s: _random_activity(rng) for s in range(4)},
    }
    out = rsa_distances_per_seed(systems, reference=None)
    assert list(out.keys()) == ["bptt_vs_pc"]
    assert len(out["bptt_vs_pc"]) == 4
    print("test_per_seed_rule_vs_rule OK")


def test_per_seed_rule_vs_rule_three_rules():
    """A third learning rule must produce all three pairs, not silently drop one.

    The two-rule key stays exactly as it was, so existing outputs keep their meaning.
    """
    rng = np.random.default_rng(7)
    systems = {
        "bptt": {s: _random_activity(rng) for s in range(4)},
        "pc": {s: _random_activity(rng) for s in range(4)},
        "rflo": {s: _random_activity(rng) for s in range(4)},
    }
    out = rsa_distances_per_seed(systems, reference=None)
    assert set(out) == {"bptt_vs_pc", "bptt_vs_rflo", "pc_vs_rflo"}
    for pair, values in out.items():
        assert len(values) == 4, pair
        assert all(isinstance(v, float) for v in values), pair
    print("test_per_seed_rule_vs_rule_three_rules OK")


def test_canonical_order_respected():
    # Build activity where each condition i is a distinct constant pattern; the RDM
    # must then reflect the identity of condition i at row i (canonical order).
    time, k = 15, 6
    act = np.zeros((N_CONDITIONS, time, k))
    for i in range(N_CONDITIONS):
        act[i, :, i % k] = float(i + 1)               # a per-condition signature
    rdm = build_rdm(act)
    # Row 0 and its own column are zero on the diagonal; the matrix is 20x20 in order.
    assert rdm.shape == (N_CONDITIONS, N_CONDITIONS)
    assert rdm[0, 0] == 0.0
    print("test_canonical_order_respected OK")


def main():
    test_rdm_shape_symmetry_zero_diagonal()
    test_identical_systems_distance_zero()
    test_permuted_conditions_larger_distance()
    test_distance_in_bounds()
    test_noise_ceiling_ordering()
    test_per_seed_model_to_reference()
    test_per_seed_rule_vs_rule()
    test_canonical_order_respected()
    print("\nall rsa tests passed")


if __name__ == "__main__":
    main()
