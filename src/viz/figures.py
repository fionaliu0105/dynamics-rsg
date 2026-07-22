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
from typing import Dict, Mapping, Sequence

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


def training_loss_curve(losses: Sequence[float], rule: str, seed: int, out_dir: Path = RESULTS_DIR) -> Path:
    """Plot one run's training loss curve from saved metrics."""
    vals = np.asarray(losses, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(vals.size), vals, color="#2f6f73", linewidth=1.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("masked MSE")
    ax.set_title(f"{rule.upper()} seed {seed}: training loss")
    ax.grid(True, color="#d0d7de", linewidth=0.6, alpha=0.8)
    return savefig(fig, f"{rule}_seed{seed:04d}_training_loss", out_dir)


def behavior_panel(eval_records: Sequence[dict], rule: str, seed: int, out_dir: Path = RESULTS_DIR) -> Path:
    """Plot produced interval against sample interval, grouped by prior."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"short": "#3b6ea8", "long": "#b45f3c"}
    for prior in sorted({str(r["prior"]) for r in eval_records}):
        rows = [r for r in eval_records if r["prior"] == prior]
        ts = np.asarray([r["ts"] for r in rows], dtype=float)
        produced = np.asarray([r["tp"] for r in rows], dtype=float)
        keep = np.isfinite(produced)
        ax.scatter(ts[keep], produced[keep], label=prior, color=colors.get(prior), s=32)
        if keep.sum() >= 3:
            slope, intercept = np.polyfit(ts[keep], produced[keep], 1)
            xs = np.linspace(ts[keep].min(), ts[keep].max(), 50)
            ax.plot(xs, slope * xs + intercept, color=colors.get(prior), linewidth=1.2)
    ax.plot([450, 1250], [450, 1250], color="#6b7280", linestyle="--", linewidth=1.0)
    ax.set_xlim(450, 1250)
    ax.set_xlabel("ts (ms)")
    ax.set_ylabel("tp (ms)")
    ax.set_title(f"{rule.upper()} seed {seed}: behavior")
    ax.legend(title="prior", frameon=False)
    ax.grid(True, color="#d0d7de", linewidth=0.6, alpha=0.8)
    return savefig(fig, f"{rule}_seed{seed:04d}_tp_vs_ts", out_dir)


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


def training_loss_figure(
    losses: Sequence[float],
    name: str = "training_loss",
    out_dir: Path = RESULTS_DIR,
) -> Path:
    """Loss-vs-iteration curve for one seed's training run."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(np.arange(len(losses)), losses)
    ax.set_xlabel("iteration")
    ax.set_ylabel("masked MSE loss")
    ax.set_title("training loss")
    return savefig(fig, name, out_dir)


def unit_activity_figure(
    states: np.ndarray,
    dt: float,
    n_units: int = 8,
    name: str = "unit_activity",
    out_dir: Path = RESULTS_DIR,
    title: str = "unit activity (r = tanh(x))",
) -> Path:
    """Time traces of a handful of units' activity for ONE condition.

    Args:
        states: ``[time, units]`` (or ``[trials, time, units]`` — first trial used).
        dt: ms per step, to label the x-axis in ms.
    """
    states = np.asarray(states)
    if states.ndim == 3:
        states = states[0]
    time_ms = np.arange(states.shape[0]) * dt
    idx = np.linspace(0, states.shape[1] - 1, min(n_units, states.shape[1])).astype(int)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i in idx:
        ax.plot(time_ms, states[:, i], lw=1)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("r")
    ax.set_title(title)
    return savefig(fig, name, out_dir)


def output_vs_target_figure(
    outputs: np.ndarray,
    target: np.ndarray,
    dt: float,
    labels: Sequence[str],
    threshold: float | None = None,
    name: str = "output_vs_target",
    out_dir: Path = RESULTS_DIR,
) -> Path:
    """Overlay produced output ``z_t`` against the ramp target, one line per trial."""
    outputs = np.asarray(outputs)
    target = np.asarray(target)
    time_ms = np.arange(outputs.shape[1]) * dt

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, outputs.shape[0]))
    for i, label in enumerate(labels):
        ax.plot(time_ms, outputs[i], color=colors[i], label=f"{label} (out)")
        ax.plot(time_ms, target[i], color=colors[i], ls="--", alpha=0.5)
    if threshold is not None:
        ax.axhline(threshold, color="k", ls=":", lw=1, label="threshold")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("z")
    ax.set_title("output (solid) vs. target ramp (dashed)")
    ncol = 1 if len(labels) <= 8 else 2
    ax.legend(fontsize=6, loc="upper left", ncol=ncol)
    return savefig(fig, name, out_dir)


def pca_trajectories_figure(
    states_by_condition: Mapping[str, np.ndarray],
    n_components: int = 2,
    name: str = "pca_trajectories",
    out_dir: Path = RESULTS_DIR,
    color_by: Mapping[str, str] | None = None,
    linestyle_by: Mapping[str, str] | None = None,
) -> Path:
    """PCA trajectories per condition — Fig 7C-style prior-support curvature panel.

    Args:
        states_by_condition: ``{condition_label: states [time, units]}``, already on
            a shared time base. PCA is fit jointly (concatenated over conditions and
            time) so all trajectories share one projection.
        color_by: optional ``{condition_label: color}`` (e.g. by prior) — falls back
            to a categorical colormap over conditions if omitted.

    No sklearn dependency: PCA via numpy SVD on the mean-centered pooled activity.
    """
    labels = list(states_by_condition)
    pooled = np.concatenate([np.asarray(states_by_condition[k]) for k in labels], axis=0)
    mean = pooled.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(pooled - mean, full_matrices=False)
    components = vt[:n_components]  # [n_components, units]

    if color_by is None:
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 1)))
        color_by = {lab: cmap[i] for i, lab in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(5, 5))
    for lab in labels:
        traj = (np.asarray(states_by_condition[lab]) - mean) @ components.T  # [time, n_components]
        ls = linestyle_by[lab] if linestyle_by else "-"
        ax.plot(traj[:, 0], traj[:, 1], color=color_by[lab], ls=ls, label=lab, lw=1.5)
        ax.scatter(*traj[0, :2], color=color_by[lab], marker="o", s=25)   # start
        ax.scatter(*traj[-1, :2], color=color_by[lab], marker="s", s=25)  # end
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("condition trajectories in PCA space (o=start, sq=end)")
    ncol = 2 if len(labels) <= 10 else 3
    ax.legend(fontsize=5, loc="best", ncol=ncol)
    return savefig(fig, name, out_dir)
