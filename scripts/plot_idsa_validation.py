"""Make a shareable figure of the iDSA validation (plan 0.7 / 2.4).

This is a demo figure, not a results figure. It simulates linear systems whose
relationships we already know, fits operators with the same code the pipeline uses,
and shows two things.

  1. InputDSA separates recurrent from input structure. Four systems built from two
     recurrent matrices (A1, A2) crossed with two input matrices (B1, B2). The state
     distance groups them by recurrent matrix; the input distance groups them by
     input matrix. Plain geometry or DSA cannot make that split.
  2. The distance ordering behaves. Identical systems sit near zero, a perturbed
     recurrent matrix reads larger, and shuffling time reads large.

Runs headless (Agg backend via src.viz.figures) and writes a PNG. Nothing here
touches torch, neurogym, the DSA repo, or trained checkpoints.

    python scripts/plot_idsa_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.compare.idsa import InputDSAConfig, fit_operators, input_dsa  # noqa: E402
from src.viz.figures import plt, savefig  # noqa: E402  (Agg backend forced there)

SEED = 7
INK = "#22303f"
BAR = "#2a9d8f"
CMAP = "viridis"  # perceptually uniform and colorblind-safe; distances are magnitude


# --- small linear-system simulator (self-contained, matches the test harness) ---


def _stable_matrix(rng, n, rho):
    A = rng.standard_normal((n, n))
    return A * (rho / np.max(np.abs(np.linalg.eigvals(A))))


def _lowpass_noise(rng, T, m, alpha=0.2):
    u = np.zeros((T, m))
    for t in range(1, T):
        u[t] = (1 - alpha) * u[t - 1] + alpha * rng.standard_normal(m)
    return u


def _simulate(A, B, rng, n_traj=14, T=140):
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


# --- compute the numbers the figure shows -----------------------------------


def _four_system_distances(cfg):
    """State- and input-distance matrices over the 4 systems A{1,2} x B{1,2}."""
    rng = np.random.default_rng(3)
    n, m = 6, 2
    A1, A2 = _stable_matrix(rng, n, 0.9), _stable_matrix(rng, n, 0.5)
    B1 = rng.standard_normal((n, m))
    B1 = 0.8 * B1 / np.linalg.norm(B1)
    B2 = rng.standard_normal((n, m))
    B2 = 1.2 * B2 / np.linalg.norm(B2)
    grid = [(A1, B1), (A1, B2), (A2, B1), (A2, B2)]
    labels = ["A1·B1", "A1·B2", "A2·B1", "A2·B2"]
    ops = [fit_operators(*_simulate(A, B, rng), cfg) for A, B in grid]

    n_sys = len(ops)
    state = np.zeros((n_sys, n_sys))
    inp = np.zeros((n_sys, n_sys))
    for i in range(n_sys):
        for j in range(n_sys):
            if i == j:
                continue
            d = input_dsa(ops[i], ops[j], cfg)
            state[i, j] = d["state_distance"]
            inp[i, j] = d["input_distance"]
    # symmetrize for display (the metric is symmetric up to numerical Procrustes error)
    return labels, 0.5 * (state + state.T), 0.5 * (inp + inp.T)


def _ordering_distances(cfg):
    """Joint distance for identical / perturbed-recurrent / shuffled-time systems."""
    rng = np.random.default_rng(SEED + 1)
    n, m = 6, 2
    A, B = _stable_matrix(rng, n, 0.9), rng.standard_normal((n, m))
    ref = fit_operators(*_simulate(A, B, rng), cfg)

    twin = fit_operators(*_simulate(A, B, rng), cfg)
    A_pert = A + 0.15 * rng.standard_normal((n, n))
    pert = fit_operators(*_simulate(A_pert, B, rng), cfg)
    s_shuf, u_shuf = _simulate(A, B, rng)
    for k in range(s_shuf.shape[0]):
        rng.shuffle(s_shuf[k])
    shuf = fit_operators(s_shuf, u_shuf, cfg)

    return {
        "identical": input_dsa(ref, twin, cfg)["distance"],
        "perturbed\nrecurrent A": input_dsa(ref, pert, cfg)["distance"],
        "shuffled\ntime": input_dsa(ref, shuf, cfg)["distance"],
    }


# --- draw --------------------------------------------------------------------


def _heatmap(ax, mat, labels, title):
    im = ax.imshow(mat, cmap=CMAP, vmin=0.0)
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_title(title, fontsize=10.5, color=INK, pad=8)
    hi = mat.max()
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mat[i, j]
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if val < 0.55 * hi else INK,
            )
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)


def main() -> Path:
    # builtin backend: a portable, dependency-free demo (no torch/dsa-metric needed).
    cfg = InputDSAConfig(method="dmdc", rank=6, backend="builtin")
    labels, state, inp = _four_system_distances(cfg)
    ordering = _ordering_distances(cfg)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    fig.suptitle(
        "InputDSA (iDSA) validation: the dynamics comparison separates recurrent from input structure",
        fontsize=13, fontweight="bold", color=INK, y=1.03,
    )

    _heatmap(
        axes[0], state, labels,
        "State distance ‖C A₁Cᵀ − A₂‖ (Eq. 9)\nsmall within a shared recurrent A",
    )
    _heatmap(
        axes[1], inp, labels,
        "Input distance ‖C B₁ − B₂‖ (Eq. 10)\nsmall within a shared input B",
    )

    ax = axes[2]
    names = list(ordering.keys())
    vals = [ordering[k] for k in names]
    bars = ax.bar(names, vals, color=BAR, width=0.62, zorder=3)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9, color=INK)
    ax.set_title("Distance ordering is sane", fontsize=11, color=INK, pad=8)
    ax.set_ylabel("InputDSA distance  (Eq. 8)", fontsize=9, color=INK)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", color="0.85", linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.text(
        0.5, -0.06,
        "Four linear systems, A{1,2} crossed with B{1,2}, fully observed. Operators fit with DMDc; "
        "distances via Procrustes on the controllability matrix. Pairs that share a recurrent A "
        "(e.g. A1·B1 vs A1·B2) give the smallest state distances; pairs that share an input B "
        "(e.g. A1·B1 vs A2·B1) give the smallest input distances. Partial-observation (neural) case "
        "uses Subspace DMDc and is covered in tests/test_idsa.py.",
        ha="center", va="top", fontsize=8.5, color="0.35", wrap=True,
    )

    path = savefig(fig, "idsa_validation")
    return path


if __name__ == "__main__":
    out = main()
    print(f"wrote {out}")
