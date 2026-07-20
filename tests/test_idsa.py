"""Synthetic-trajectory validation for the iDSA track (plan 0.7 / 2.4).

This runs without the DSA repo, torch, neurogym, or trained checkpoints. It fits
operators on controlled linear systems whose relationships we know, and checks that
the InputDSA distances order them the way the paper predicts (arXiv 2510.25943,
Sec 3.1):

    * two identical systems              -> distance ~ 0 (smallest)
    * perturbed recurrent dynamics       -> larger than identical
    * shuffled time points               -> large distance
    * same recurrent dynamics, different input matrix
                                         -> small state distance. This is the point
                                            of iDSA; plain DSA would call these two
                                            systems far apart.

It also covers the basic contracts: finite and reproducible distances, dimension
guards, and that Subspace DMDc runs and returns finite results on partially observed
data.

Run from repo root::

    python tests/test_idsa.py       # plain asserts, no pytest needed
    pytest tests/test_idsa.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.compare.idsa import (
    InputDSAConfig,
    Operators,
    controllability_matrix,
    dsa_distance,
    fit_operators,
    input_dsa,
    subspace_dmdc,
)


# --- helpers ----------------------------------------------------------------


def _stable_matrix(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    """Random n x n matrix rescaled to spectral radius ``rho`` (paper Appendix J)."""
    A = rng.standard_normal((n, n))
    A *= rho / np.max(np.abs(np.linalg.eigvals(A)))
    return A


def _lowpass_noise(rng: np.random.Generator, T: int, m: int, alpha: float = 0.2) -> np.ndarray:
    """Low-pass filtered white noise input drive [T, m] (paper's input family)."""
    u = np.zeros((T, m))
    for t in range(1, T):
        u[t] = (1 - alpha) * u[t - 1] + alpha * rng.standard_normal(m)
    return u


def _simulate(A, B, rng, n_traj=12, T=120):
    """Roll out x_{t+1} = A x_t + B u_t for several trajectories -> [n_traj, T, n]."""
    n, m = B.shape
    states = np.empty((n_traj, T, n))
    inputs = np.empty((n_traj, T, m))
    for k in range(n_traj):
        u = _lowpass_noise(rng, T, m)
        x = rng.standard_normal(n) * 0.1
        for t in range(T):
            states[k, t] = x
            inputs[k, t] = u[t]
            x = A @ x + B @ u[t]
    return states, inputs


# --- tests ------------------------------------------------------------------


def test_identical_systems_near_zero():
    rng = np.random.default_rng(0)
    n, m = 6, 2
    A, B = _stable_matrix(rng, n, 0.9), rng.standard_normal((n, m))
    cfg = InputDSAConfig(method="dmdc", rank=n)
    s1, u1 = _simulate(A, B, rng)
    s2, u2 = _simulate(A, B, rng)          # same operators, different noise/inits
    op1, op2 = fit_operators(s1, u1, cfg), fit_operators(s2, u2, cfg)

    d_same = dsa_distance(op1, op2, cfg)
    # normalize by system scale to talk about "near zero"
    scale = np.linalg.norm(controllability_matrix(op1.A, op1.B, cfg.n_powers, True, cfg.power_norm_cap))
    assert np.isfinite(d_same)
    assert d_same / scale < 0.15, f"identical systems not close: {d_same/scale:.3f}"
    print(f"test_identical_systems_near_zero OK (rel dist {d_same/scale:.4f})")
    return d_same / scale


def test_perturbed_dynamics_larger_than_identical():
    rng = np.random.default_rng(1)
    n, m = 6, 2
    A, B = _stable_matrix(rng, n, 0.9), rng.standard_normal((n, m))
    cfg = InputDSAConfig(method="dmdc", rank=n)
    s1, u1 = _simulate(A, B, rng)
    s2, u2 = _simulate(A, B, rng)
    A_pert = A + 0.15 * rng.standard_normal((n, n))
    sp, up = _simulate(A_pert, B, rng)
    op1, op2, opp = (fit_operators(x, y, cfg) for x, y in [(s1, u1), (s2, u2), (sp, up)])

    d_same = input_dsa(op1, op2, cfg)["state_distance"]
    d_pert = input_dsa(op1, opp, cfg)["state_distance"]
    assert d_pert > d_same, f"perturbed ({d_pert:.3f}) not > identical ({d_same:.3f})"
    print(f"test_perturbed_dynamics_larger_than_identical OK ({d_same:.3f} < {d_pert:.3f})")


def test_shuffled_time_is_far():
    rng = np.random.default_rng(2)
    n, m = 6, 2
    A, B = _stable_matrix(rng, n, 0.9), rng.standard_normal((n, m))
    cfg = InputDSAConfig(method="dmdc", rank=n)
    s1, u1 = _simulate(A, B, rng)
    s2, u2 = _simulate(A, B, rng)
    # destroy the temporal order of one system's states within each trajectory
    s_shuf = s2.copy()
    for k in range(s_shuf.shape[0]):
        rng.shuffle(s_shuf[k])
    op1 = fit_operators(s1, u1, cfg)
    op_ok = fit_operators(s2, u2, cfg)
    op_shuf = fit_operators(s_shuf, u2, cfg)

    d_ok = dsa_distance(op1, op_ok, cfg)
    d_shuf = dsa_distance(op1, op_shuf, cfg)
    assert d_shuf > d_ok, f"shuffled ({d_shuf:.3f}) not > intact ({d_ok:.3f})"
    print(f"test_shuffled_time_is_far OK (intact {d_ok:.3f} < shuffled {d_shuf:.3f})")


def test_demix_same_dynamics_different_input():
    """Same A, different B should give a small state distance.

    Plain geometry or DSA would be thrown off by the different input drive. InputDSA
    separates it out, so the recurrent (state) distance stays small while the input
    distance is large.
    """
    rng = np.random.default_rng(3)
    n, m = 6, 2
    A = _stable_matrix(rng, n, 0.9)
    B1 = rng.standard_normal((n, m)) * 0.5
    B2 = rng.standard_normal((n, m)) * 2.0        # very different input mapping
    A_other = _stable_matrix(rng, n, 0.8)         # a genuinely different recurrent A
    cfg = InputDSAConfig(method="dmdc", rank=n)

    op_a = fit_operators(*_simulate(A, B1, rng), cfg)
    op_b = fit_operators(*_simulate(A, B2, rng), cfg)      # same A, different B
    op_c = fit_operators(*_simulate(A_other, B1, rng), cfg)  # different A

    same_A = input_dsa(op_a, op_b, cfg)
    diff_A = input_dsa(op_a, op_c, cfg)
    # same recurrent dynamics -> small state distance despite different inputs
    assert same_A["state_distance"] < diff_A["state_distance"], (
        f"demixing failed: same-A state dist {same_A['state_distance']:.3f} "
        f"not < diff-A {diff_A['state_distance']:.3f}"
    )
    # and the different-input pair is separated by the INPUT distance
    assert same_A["input_distance"] > 0
    print(
        "test_demix_same_dynamics_different_input OK "
        f"(same-A state {same_A['state_distance']:.3f} < diff-A {diff_A['state_distance']:.3f}; "
        f"input dist same-A {same_A['input_distance']:.3f})"
    )


def test_reproducible_and_symmetric():
    rng = np.random.default_rng(4)
    n, m = 5, 2
    A, B = _stable_matrix(rng, n, 0.9), rng.standard_normal((n, m))
    cfg = InputDSAConfig(method="dmdc", rank=n)
    op1 = fit_operators(*_simulate(A, B, rng), cfg)
    op2 = fit_operators(*_simulate(A, B, rng), cfg)
    d_ab = dsa_distance(op1, op2, cfg)
    d_ab2 = dsa_distance(op1, op2, cfg)
    d_ba = dsa_distance(op2, op1, cfg)
    assert d_ab == d_ab2, "distance not reproducible for a fixed config"
    assert abs(d_ab - d_ba) < 1e-8 * (1 + d_ab), f"distance not symmetric: {d_ab} vs {d_ba}"
    print(f"test_reproducible_and_symmetric OK (d={d_ab:.4f}, symmetric)")


def test_dimension_guards():
    a = Operators(A=np.eye(4), B=np.ones((4, 2)), rank=4, method="dmdc", delays=1)
    b = Operators(A=np.eye(5), B=np.ones((5, 2)), rank=5, method="dmdc", delays=1)
    try:
        dsa_distance(a, b)
    except ValueError:
        print("test_dimension_guards OK (rank mismatch rejected)")
    else:
        raise AssertionError("expected ValueError on mismatched rank")


def test_subspace_dmdc_runs_partial_obs():
    """Subspace DMDc returns finite operators on partially observed data (Stage 4 path)."""
    rng = np.random.default_rng(5)
    n_full, m, n_obs = 12, 2, 3
    A = _stable_matrix(rng, n_full, 0.9)
    B = rng.standard_normal((n_full, m))
    states_full, inputs = _simulate(A, B, rng, n_traj=10, T=200)
    states_obs = states_full[:, :, :n_obs]              # observe only 3 of 12 dims
    cfg = InputDSAConfig(method="subspace", rank=6, delays=4)
    op = subspace_dmdc(states_obs, inputs, cfg)
    assert np.all(np.isfinite(op.A)) and np.all(np.isfinite(op.B))
    assert op.A.shape == (op.rank, op.rank)
    # a second draw of the same system compares with a finite distance
    states_full2, inputs2 = _simulate(A, B, rng, n_traj=10, T=200)
    op2 = subspace_dmdc(states_full2[:, :, :n_obs], inputs2, cfg)
    d = dsa_distance(op, op2, cfg)
    assert np.isfinite(d)
    print(f"test_subspace_dmdc_runs_partial_obs OK (self-distance {d:.4e}, finite)")


def test_subspace_dmdc_demixes_partial_obs():
    """Subspace DMDc separates recurrent from input structure under partial observation.

    Paper Fig. 2: with 4 systems built from 2 recurrent matrices by 2 input matrices,
    observing only a few dimensions, InputDSA should judge same-A systems close in
    state distance and same-B systems close in input distance. Plain DMDc's B gets
    biased toward the intrinsic dynamics under partial observation, which is why
    Stage 4 (the neural comparison) uses Subspace DMDc instead.
    """
    rng = np.random.default_rng(11)
    n_full, m, n_obs = 15, 2, 3
    A1, A2 = _stable_matrix(rng, n_full, 0.9), _stable_matrix(rng, n_full, 0.8)
    B1 = rng.standard_normal((n_full, m)) * 0.6
    B2 = rng.standard_normal((n_full, m)) * 1.8
    cfg = InputDSAConfig(method="subspace", rank=8, delays=6)

    def op(A, B):
        s, u = _simulate(A, B, rng, n_traj=12, T=400)
        return fit_operators(s[:, :, :n_obs], u, cfg)

    o11, o12, o21 = op(A1, B1), op(A1, B2), op(A2, B1)
    same_A = input_dsa(o11, o12, cfg)     # share recurrent A1, differ in input
    diff_A = input_dsa(o11, o21, cfg)     # share input B1, differ in recurrent
    assert same_A["state_distance"] < diff_A["state_distance"], (
        f"recurrent demixing failed: same-A {same_A['state_distance']:.3f} "
        f"!< diff-A {diff_A['state_distance']:.3f}"
    )
    assert diff_A["input_distance"] < same_A["input_distance"], (
        f"input demixing failed: same-B {diff_A['input_distance']:.3f} "
        f"!< diff-B {same_A['input_distance']:.3f}"
    )
    print(
        "test_subspace_dmdc_demixes_partial_obs OK "
        f"(state: same-A {same_A['state_distance']:.3f} < diff-A {diff_A['state_distance']:.3f}; "
        f"input: same-B {diff_A['input_distance']:.3f} < diff-B {same_A['input_distance']:.3f})"
    )


if __name__ == "__main__":
    test_identical_systems_near_zero()
    test_perturbed_dynamics_larger_than_identical()
    test_shuffled_time_is_far()
    test_demix_same_dynamics_different_input()
    test_reproducible_and_symmetric()
    test_dimension_guards()
    test_subspace_dmdc_runs_partial_obs()
    test_subspace_dmdc_demixes_partial_obs()
    print("\nall iDSA synthetic-validation tests passed")
