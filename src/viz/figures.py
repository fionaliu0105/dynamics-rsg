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
