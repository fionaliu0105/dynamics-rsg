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
from typing import Dict, Mapping, Optional, Sequence, Tuple

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
    ceilings: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> Path:
    """THE headline figure: distance-to-DMFC per rule, per metric, with seed spread.

    Args:
        distances: ``{metric: {rule: [per-seed distances]}}``, e.g.
            ``{"RSA": {"bptt": [...], "pc": [...]}, "iDSA": {...}}``.
        ceilings: optional ``{metric: (lower, upper)}`` neural noise-ceiling band, in
            the SAME distance units, drawn as a shaded span per metric panel (from
            ``src.compare.rsa.noise_ceiling``). Omitted metrics get no band.

    Draws mean +/- spread over seeds per rule, grouped by metric. This reads saved
    metrics only. Reusable as-is; tracks feed it their per-seed distance arrays.
    """
    metrics = list(distances)
    rules = sorted({r for m in distances.values() for r in m})
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        if ceilings and metric in ceilings:
            lo, hi = ceilings[metric]
            ax.axhspan(lo, hi, color="0.8", alpha=0.6, zorder=0, label="noise ceiling")
        for i, rule in enumerate(rules):
            vals = np.asarray(distances[metric].get(rule, []), dtype=float)
            if vals.size:
                ax.bar(i, vals.mean(), yerr=vals.std(), capsize=5, label=rule)
                ax.scatter(np.full(vals.size, i), vals, color="k", s=12, zorder=3)
        ax.set_xticks(range(len(rules)))
        ax.set_xticklabels(rules)
        ax.set_title(f"{metric}: distance to DMFC")
        ax.set_ylabel("distance (per-seed spread)")
        if ceilings and metric in ceilings:
            ax.legend(fontsize=8)
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

# --- Behavior-track panel (Fig 1E, plan 2.2 / Step D) --------------------------
# Reads already-measured per-condition behavior (ts, tp, prior) — from the store's meta
# or an aggregated metrics table — and draws the tp-vs-ts regression per prior. It never
# runs a model; tp comes from src.behavior.slope on stored outputs.

def behavior_slope_figure(
    ts,
    tp,
    prior_labels,
    *,
    title: str = "Behavior: tp vs ts (Fig 1E)",
    name: str = "behavior_tp_vs_ts",
    out_dir: Path = RESULTS_DIR,
) -> Path:
    """Fig 1E: produced interval ``tp`` vs sample interval ``ts``, per prior, with the
    fitted tp-vs-ts slope — the Bayesian-bias signature (slope in (0, 1); Long flatter).

    Args:
        ts, tp, prior_labels: equal-length saved metrics, one entry per stored
            (condition[, seed]). ``tp`` may contain ``NaN`` (no threshold crossing);
            NaN points are dropped from the fit.

    The reported slope per prior comes from :func:`src.behavior.slope.slopes_by_prior`
    (the canonical metric), so the drawn line and the legend agree. Reads metrics only —
    never retrains or re-extracts.
    """
    from src.behavior.slope import slopes_by_prior  # local: keep module import-light

    ts = np.asarray(ts, dtype=float)
    tp = np.asarray(tp, dtype=float)
    labels = np.asarray(prior_labels)
    slopes = slopes_by_prior(ts, tp, labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    color = {"short": "tab:blue", "long": "tab:red"}
    finite_ts = ts[np.isfinite(ts)]
    if finite_ts.size:
        lo, hi = finite_ts.min(), finite_ts.max()
        ax.plot([lo, hi], [lo, hi], ls=":", c="gray", lw=1, label="unity (slope 1)")
    for prior in dict.fromkeys(labels.tolist()):
        m = (labels == prior) & np.isfinite(tp)
        if not m.any():
            continue
        c = color.get(prior)
        tsg, tpg, s = ts[m], tp[m], slopes[prior]
        ax.scatter(tsg, tpg, s=28, c=c, edgecolors="k", linewidths=0.4,
                   label=f"{prior} (slope={s:.2f})")
        if np.isfinite(s):
            b = tpg.mean() - s * tsg.mean()          # OLS line through the centroid
            xs = np.array([tsg.min(), tsg.max()])
            ax.plot(xs, s * xs + b, c=c, lw=2)
    ax.set_xlabel("sample interval ts (ms)")
    ax.set_ylabel("produced interval tp (ms)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    return savefig(fig, name, out_dir)


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
