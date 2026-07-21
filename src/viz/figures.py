"""Plotting harness.  [FOUNDATION — I own the harness; tracks add their panels; plan 3.3]

Non-interactive by construction: forces the ``Agg`` backend and writes files (no
``plt.show()`` — must work headless on a compute node, AGENTS.md "Execution"). Reads
saved metrics only; never retrains or re-extracts.

Panels the summary needs:
    * behavioral tp-vs-ts with fitted slope (Fig 1E) — behavior track
    * training-loss curves per seed
    * PCA trajectories showing prior-support curvature (Fig 7C)
    * RDM heatmaps + MDS — RSA track
    * THE SUMMARY FIGURE: PC vs BPTT distance to DMFC on RSA and iDSA, seed spread
    * pairwise system-distance matrix

This file gives the harness (backend, save helper, a seed-spread summary skeleton);
each track fills in its panel. Uses matplotlib (present) + numpy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: pick the backend BEFORE importing pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS_DIR = Path("results/figures")


def savefig(fig, name: str, out_dir: Path = RESULTS_DIR) -> Path:
    """Save a figure to ``results/figures/<name>.png`` and close it. Returns the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def summary_distance_figure(
    distances: Dict[str, Dict[str, Sequence[float]]],
    out_dir: Path = RESULTS_DIR,
) -> Path:
    """THE headline figure: distance-to-DMFC per rule, per metric, with seed spread.

    Args:
        distances: ``{metric: {rule: [per-seed distances]}}``, e.g.
            ``{"RSA": {"bptt": [...], "pc": [...]}, "iDSA": {...}}``.

    Draws mean +/- spread over seeds per rule, grouped by metric. This reads saved
    metrics only. Reusable as-is; tracks feed it their per-seed distance arrays.
    """
    metrics = list(distances)
    rules = sorted({r for m in distances.values() for r in m})
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for i, rule in enumerate(rules):
            vals = np.asarray(distances[metric].get(rule, []), dtype=float)
            if vals.size:
                ax.bar(i, vals.mean(), yerr=vals.std(), capsize=5, label=rule)
                ax.scatter(np.full(vals.size, i), vals, color="k", s=12, zorder=3)
        ax.set_xticks(range(len(rules)))
        ax.set_xticklabels(rules)
        ax.set_title(f"{metric}: distance to DMFC")
        ax.set_ylabel("distance (per-seed spread)")
    return savefig(fig, "summary_distance_to_dmfc", out_dir)


# --- RSA-track panels (plan 3.3) -----------------------------------------------
# These read an already-computed RDM (20x20, canonical condition order) and write a
# file. They never build the RDM or re-extract activity — that is src.compare.rsa.

def rdm_heatmap(rdm, name: str = "rdm_heatmap", out_dir: Path = RESULTS_DIR) -> Path:
    """Heatmap of a 20x20 RDM in the canonical condition order.

    ``rdm``: [n_cond, n_cond] dissimilarity matrix from src.compare.rsa.build_rdm.
    Ticks are labelled with the canonical Condition labels so the prior/effector/ts
    structure is legible.
    """
    from src.conditions import CONDITIONS  # local import keeps this module import-light

    rdm = np.asarray(rdm, dtype=float)
    labels = [c.label for c in CONDITIONS]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(rdm, cmap="viridis", aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("RDM (dissimilarity)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dissimilarity")
    return savefig(fig, name, out_dir)


def rdm_mds(rdm, name: str = "rdm_mds", out_dir: Path = RESULTS_DIR) -> Path:
    """2-D MDS embedding of a 20x20 RDM, colored by prior and marked by effector.

    Uses classical (Torgerson) MDS via numpy eigendecomposition — no sklearn needed
    (sklearn is ABI-broken in the base env). Points are colored short/long and marked
    eye/hand so the prior geometry and effector split are visible at a glance.
    """
    from src.conditions import CONDITIONS

    rdm = np.asarray(rdm, dtype=float)
    coords = _classical_mds(rdm, n_components=2)
    fig, ax = plt.subplots(figsize=(6, 5))
    color = {"short": "tab:blue", "long": "tab:red"}
    marker = {"eye": "o", "hand": "^"}
    seen = set()
    for i, c in enumerate(CONDITIONS):
        key = (c.prior, c.effector)
        ax.scatter(
            coords[i, 0], coords[i, 1],
            c=color[c.prior], marker=marker[c.effector], s=60,
            edgecolors="k", linewidths=0.5,
            label=f"{c.prior}/{c.effector}" if key not in seen else None,
        )
        seen.add(key)
    ax.set_title("RDM — classical MDS")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.legend(fontsize=7, loc="best")
    return savefig(fig, name, out_dir)


def _classical_mds(rdm: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Torgerson classical MDS: double-center -D^2/2, take top eigenvectors."""
    d2 = np.asarray(rdm, dtype=float) ** 2
    n = d2.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    b = 0.5 * (b + b.T)                              # symmetrize
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1][:n_components]
    pos = np.clip(vals[order], 0.0, None)
    return vecs[:, order] * np.sqrt(pos)
