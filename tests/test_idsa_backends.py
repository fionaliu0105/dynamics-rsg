"""Cross-check the two iDSA backends (plan 0.6 / 0.7).

The builtin numpy/scipy implementation and the official ``dsa-metric`` package should
agree on the InputDSA distances. This is the concrete "install & validate" step: it
confirms our fallback reproduces the reference, and that the reference is wired
correctly (right argument order, right component mapping).

The whole module skips when ``dsa-metric`` is not importable, so it is a no-op in the
modeling env and runs in the isolated iDSA env (requirements-idsa.txt)::

    <dsa-env>/bin/python -m pytest tests/test_idsa_backends.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

pytest.importorskip("DSA", reason="official dsa-metric package not installed in this env")

from src.compare.idsa import InputDSAConfig, fit_operators, input_dsa


def _stable_matrix(rng, n, rho):
    A = rng.standard_normal((n, n))
    return A * (rho / np.max(np.abs(np.linalg.eigvals(A))))


def _lowpass_noise(rng, T, m, alpha=0.2):
    u = np.zeros((T, m))
    for t in range(1, T):
        u[t] = (1 - alpha) * u[t - 1] + alpha * rng.standard_normal(m)
    return u


def _simulate(A, B, rng, n_traj=14, T=160):
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


def _four_systems(rng, n=6, m=2):
    A1, A2 = _stable_matrix(rng, n, 0.9), _stable_matrix(rng, n, 0.5)
    B1 = rng.standard_normal((n, m))
    B1 = 0.8 * B1 / np.linalg.norm(B1)
    B2 = rng.standard_normal((n, m))
    B2 = 1.2 * B2 / np.linalg.norm(B2)
    return [(A1, B1), (A1, B2), (A2, B1), (A2, B2)]


def _pair_distances(cfg):
    """Off-diagonal state/input distances over the 4 systems for one backend."""
    rng = np.random.default_rng(3)
    ops = [fit_operators(*_simulate(A, B, rng), cfg) for A, B in _four_systems(rng)]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    state = np.array([input_dsa(ops[i], ops[j], cfg)["state_distance"] for i, j in pairs])
    inp = np.array([input_dsa(ops[i], ops[j], cfg)["input_distance"] for i, j in pairs])
    return state, inp


def test_backends_agree_dmdc():
    """Builtin and dsa-metric produce correlated, similarly-scaled distances (dmdc)."""
    builtin = InputDSAConfig(method="dmdc", rank=6, backend="builtin")
    official = InputDSAConfig(method="dmdc", rank=6, backend="dsa-metric")

    b_state, b_inp = _pair_distances(builtin)
    o_state, o_inp = _pair_distances(official)

    # strong agreement in shape (which pairs are near/far)
    assert np.corrcoef(b_state, o_state)[0, 1] > 0.95, (b_state, o_state)
    assert np.corrcoef(b_inp, o_inp)[0, 1] > 0.95, (b_inp, o_inp)
    # and similar magnitude (the two estimators recover the same linear system)
    assert np.allclose(b_state, o_state, rtol=0.25, atol=0.2), (b_state, o_state)
    assert np.allclose(b_inp, o_inp, rtol=0.25, atol=0.2), (b_inp, o_inp)


def test_both_backends_demix():
    """Both backends group state by recurrent A and input by input B (paper Fig 2)."""
    for backend in ("builtin", "dsa-metric"):
        cfg = InputDSAConfig(method="dmdc", rank=6, backend=backend)
        state, inp = _pair_distances(cfg)
        # pairs order: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        same_A = [state[0], state[5]]                 # (0,1)=A1, (2,3)=A2
        diff_A = [state[1], state[2], state[3], state[4]]
        same_B = [inp[1], inp[4]]                      # (0,2)=B1, (1,3)=B2
        diff_B = [inp[0], inp[2], inp[3], inp[5]]
        assert max(same_A) < min(diff_A), (backend, "state", state)
        assert max(same_B) < min(diff_B), (backend, "input", inp)


if __name__ == "__main__":
    test_backends_agree_dmdc()
    test_both_backends_demix()
    print("iDSA backend cross-check passed (builtin vs dsa-metric agree)")
