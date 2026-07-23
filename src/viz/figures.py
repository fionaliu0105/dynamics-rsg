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
    ceilings: Optional[Mapping[str, Tuple[float, float]]] = None,
    title_suffix: str = "distance to DMFC",
    name: str = "summary_distance_to_dmfc",
) -> Path:
    """THE headline figure: distance-to-DMFC per rule, per metric, with seed spread.

    Args:
        distances: ``{metric: {rule: [per-seed distances]}}``, e.g.
            ``{"RSA": {"bptt": [...], "pc": [...]}, "iDSA": {...}}``. Also reused for
            rule-vs-rule (model-to-model, no DMFC) comparisons, where the "rule" keys
            are instead comparison labels (e.g. ``{"pc_steps20": [...], "pc_steps100":
            [...]}``) — pass ``title_suffix``/``name`` to describe that case correctly,
            since "distance to DMFC" would be wrong when there's no DMFC involved.
        ceilings: optional ``{metric: (lower, upper)}`` neural noise-ceiling band, in
            the SAME distance units, drawn as a shaded span per metric panel (from
            ``src.compare.rsa.noise_ceiling``). Omitted metrics get no band.
        title_suffix: appended to each panel's title as ``f"{metric}: {title_suffix}"``.
        name: output filename (without extension).

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
        ax.set_title(f"{metric}: {title_suffix}")
        ax.set_ylabel("distance (per-seed spread)")
        if ceilings and metric in ceilings:
            ax.legend(fontsize=8)
    return savefig(fig, name, out_dir)


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

def rdm_heatmap(
    rdm, name: str = "rdm_heatmap", out_dir: Path = RESULTS_DIR, system_label: str = "",
) -> Path:
    """Heatmap of ONE system's own 20x20 RDM in the canonical condition order.

    ``rdm``: [n_cond, n_cond] dissimilarity matrix from src.compare.rsa.build_rdm, for
    a single system (e.g. one model seed, or DMFC). BOTH axes are the SAME 20
    canonical conditions, in the SAME order -- this is one system's own
    condition-by-condition geometry, not a cross-system (e.g. model-vs-DMFC) matrix.
    RSA compares two systems by comparing two SEPARATE heatmaps like this one to each
    other (their overall pattern), never by reading a cell of one against a cell of
    another directly.

    Args:
        system_label: shown in the title (e.g. "DMFC", "bptt seed 3") so which
            system's own geometry this is is never ambiguous from the image alone.
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
    title = "RDM: condition x condition dissimilarity"
    if system_label:
        title = f"{system_label} — {title}"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dissimilarity")
    return savefig(fig, name, out_dir)


def rdm_gallery_figure(
    rdms_by_label: Mapping[str, Sequence[np.ndarray]],
    name: str = "rdm_gallery",
    out_dir: Path = RESULTS_DIR,
    seeds_by_label: Optional[Mapping[str, Sequence[int]]] = None,
) -> Path:
    """Small-multiples grid of RDM heatmaps: one row per label, one panel per seed.

    Every panel is ONE system's own 20x20 condition-by-condition RDM (both axes are
    the same canonical condition order; see :func:`rdm_heatmap`'s docstring) -- this
    grid is NOT a cross-system (e.g. model-vs-DMFC) matrix. Comparing two panels'
    overall pattern to each other is what RSA does; axis ticks are omitted here for
    space (each panel is small) -- use :func:`rdm_heatmap` on a single RDM for the
    full labeled version.

    Args:
        rdms_by_label: ``{label: [rdm, rdm, ...]}`` — e.g. ``{"dmfc": [rdm], "bptt":
            [seed0_rdm, seed1_rdm, ...]}``. Every seed's RDM is shown (not one
            "representative" seed), so the geometry keeps the same seed-spread
            AGENTS.md asks similarity numbers to carry.
        seeds_by_label: optional ``{label: [seed_id, ...]}`` matching the order of
            ``rdms_by_label``, so column titles show the real seed id rather than its
            position in the list (matters whenever seeds are non-contiguous, e.g.
            ``[1, 5, 7, 9]``). Falls back to positional numbering if omitted.

    All panels share one color scale (min/max over every RDM shown) so panels are
    visually comparable, with a single shared colorbar.
    """
    labels = list(rdms_by_label)
    n_cols = max(len(rdms_by_label[lab]) for lab in labels)
    all_vals = np.concatenate([np.asarray(r).ravel() for lab in labels for r in rdms_by_label[lab]])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    fig, axes = plt.subplots(
        len(labels), n_cols, figsize=(2.1 * n_cols, 2.3 * len(labels)), squeeze=False,
    )
    im = None
    for i, lab in enumerate(labels):
        rdms = rdms_by_label[lab]
        seed_ids = seeds_by_label[lab] if seeds_by_label and lab in seeds_by_label else list(range(len(rdms)))
        # Only label columns "seed N" for rows that actually have multiple seeds --
        # a single-panel row (e.g. DMFC, which isn't a "seed" at all) gets no title,
        # rather than the misleading "seed 0" a positional fallback would suggest.
        show_seed_titles = len(rdms) > 1
        for j in range(n_cols):
            ax = axes[i][j]
            if j < len(rdms):
                im = ax.imshow(np.asarray(rdms[j]), cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(lab, fontsize=9)
                if show_seed_titles:
                    ax.set_title(f"seed {seed_ids[j]}", fontsize=8)
            else:
                ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="dissimilarity")
    fig.suptitle(
        "RDM gallery — each panel is that ONE system's own 20x20 condition-by-condition "
        "dissimilarity matrix (same condition order on both axes); it is not a\n"
        "model-vs-DMFC matrix. RSA compares two panels' overall pattern to each other, "
        "never one panel's cell against another panel's cell.",
        fontsize=9,
    )
    return savefig(fig, name, out_dir)


def _classical_mds(rdm: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Classical (metric) MDS via double-centering + eigendecomposition.

    Deliberately plain numpy (not sklearn.manifold.MDS): comparing independently-fit
    non-metric MDS solutions across systems invites a rotation/reflection-ambiguity
    trap, so each system's embedding here is shown as its own panel rather than
    overlaid with another system's.
    """
    rdm = np.asarray(rdm, dtype=float)
    n = rdm.shape[0]
    d2 = rdm ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    pos = np.clip(eigvals[:n_components], a_min=0.0, a_max=None)
    return eigvecs[:, :n_components] * np.sqrt(pos)[None, :]


def mds_embedding_figure(
    rdms_by_label: Mapping[str, np.ndarray],
    name: str = "mds_embedding",
    out_dir: Path = RESULTS_DIR,
) -> Path:
    """Classical 2D MDS of each system's RDM, condition points colored by prior,
    marker shaped by effector — one panel per label (e.g. dmfc + each arm).

    Args:
        rdms_by_label: ``{label: rdm [20, 20]}`` — one RDM per label (e.g. each arm's
            seed-averaged RDM, and DMFC's).
    """
    from src.conditions import CONDITIONS  # local import keeps this module import-light

    labels = list(rdms_by_label)
    color = {"short": "#3b6ea8", "long": "#b45f3c"}
    marker = {"eye": "o", "hand": "^"}

    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 4.2), squeeze=False)
    for ax, lab in zip(axes[0], labels):
        pts = _classical_mds(rdms_by_label[lab], n_components=2)
        for i, c in enumerate(CONDITIONS):
            ax.scatter(
                pts[i, 0], pts[i, 1], color=color[c.prior], marker=marker[c.effector],
                s=40, edgecolors="k", linewidths=0.3,
            )
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color["short"], markersize=8, label="short"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color["long"], markersize=8, label="long"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="eye"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=8, label="hand"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Classical MDS of condition geometry (color=prior, marker=effector)", fontsize=11, y=1.1)
    return savefig(fig, name, out_dir)


def rsa_temporal_figure(
    times_ms: np.ndarray,
    curves_by_arm: Mapping[str, np.ndarray],
    ceiling: Optional[Tuple[float, float]] = None,
    name: str = "rsa_temporal",
    out_dir: Path = RESULTS_DIR,
) -> Path:
    """Geometry-over-time curve: distance-to-DMFC per time bin, mean +/- seed spread.

    Args:
        times_ms: ``[n_time_bins]`` bin centers in ms (from dmfc_meta's bin_ms).
        curves_by_arm: ``{arm: [n_seeds, n_time_bins]}`` per-seed distance-to-DMFC
            curves (each row from build_rdms_over_time + rdm_distance per bin).
        ceiling: optional ``(lower, upper)`` neural noise-ceiling band (same distance
            units as the RSA summary bars), drawn as a shaded span.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if ceiling is not None:
        lo, hi = ceiling
        ax.axhspan(lo, hi, color="0.8", alpha=0.6, zorder=0, label="noise ceiling")
    for arm, curves in curves_by_arm.items():
        curves = np.asarray(curves, dtype=float)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        ax.plot(times_ms, mean, lw=1.8, label=arm)
        ax.fill_between(times_ms, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("RSA distance to DMFC")
    ax.set_title("Time-resolved RSA: geometry-to-DMFC distance across the trial")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, color="#d0d7de", linewidth=0.6, alpha=0.8)
    return savefig(fig, name, out_dir)


def eigenvalue_spectrum_figure(
    eigs_by_arm: Mapping[str, Sequence[np.ndarray]],
    dmfc_eigs: np.ndarray,
    name: str = "eigenvalue_spectrum",
    out_dir: Path = RESULTS_DIR,
) -> Path:
    """Complex-plane eigenvalue spectra of the fitted recurrent operator A: one panel
    per arm, every seed's eigenvalues plus DMFC's, with the unit circle for reference.

    Args:
        eigs_by_arm: ``{arm: [eigs_seed0, eigs_seed1, ...]}``, each ``eigs_seedN`` a
            complex array (``np.linalg.eigvals`` of that seed's fitted ``A``).
        dmfc_eigs: DMFC's own fitted ``A``'s eigenvalues, overlaid in every panel.

    Points at/beyond the unit circle indicate marginally-stable or unstable modes;
    the angle encodes oscillation frequency (relative to the state's own time step).
    """
    arms = list(eigs_by_arm)
    theta = np.linspace(0, 2 * np.pi, 200)
    fig, axes = plt.subplots(1, len(arms), figsize=(4.2 * len(arms), 4.2), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        ax.plot(np.cos(theta), np.sin(theta), color="0.6", lw=1, ls="--", zorder=1)
        for seed_eigs in eigs_by_arm[arm]:
            seed_eigs = np.asarray(seed_eigs)
            ax.scatter(seed_eigs.real, seed_eigs.imag, color="#3b6ea8", s=18, alpha=0.6, zorder=2)
        dmfc_eigs = np.asarray(dmfc_eigs)
        ax.scatter(dmfc_eigs.real, dmfc_eigs.imag, color="#b45f3c", s=28, marker="x", zorder=3, label="DMFC")
        ax.set_title(arm, fontsize=10)
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Eigenvalue spectra of the fitted recurrent operator A (blue=model, orange=DMFC)", fontsize=10.5)
    return savefig(fig, name, out_dir)


def unit_activity_figure(
    states: np.ndarray,
    dt: float,
    n_units: int = 8,
    name: str = "unit_activity",
    out_dir: Path = RESULTS_DIR,
    condition_label: str | None = None,
) -> Path:
    """Time traces of a handful of individual recurrent units, for ONE condition only.

    This is a raw single-unit sanity check, not a similarity/geometry figure: it lets
    you eyeball whether individual units are doing something sensible (ramping,
    oscillating, saturating) rather than pooling across units the way RSA/iDSA or the
    PCA trajectory figure do. It only ever shows one condition (out of the 20 the
    model was trained on) because with 20 conditions x many units, plotting more than
    one at a time would be unreadable -- see the caller for which single condition it
    picked (typically just the first canonical one, not a specially chosen one) and
    read it next to the PCA/RSA figures, which DO show every condition, for the fuller
    picture.

    Args:
        states: ``[time, units]`` (or ``[trials, time, units]`` — first trial used).
            ``units`` is the model's full recurrent layer (e.g. N=160 in the reduced
            regime) -- NOT a fixed "8 channels"; only ``n_units`` of them are
            subsampled (evenly spaced by index, an arbitrary but reproducible pick,
            not the "most interesting" units) so the plot stays readable. The
            legend identifies exactly which unit index each line is.
        dt: ms per step, to label the x-axis in ms.
        condition_label: which single condition's trial this is (e.g.
            ``"short/480ms/eye"``), shown in the title so it's never ambiguous which
            of the 20 trained conditions this trace belongs to.

    ``r`` (the y-axis) is the unit's activation ``r_t = tanh(x_t)``, where ``x_t`` is
    the recurrent layer's pre-activation state at time ``t`` -- the same ``r`` used
    everywhere else in this codebase (the RNN's hidden state fed to the readout).
    """
    states = np.asarray(states)
    if states.ndim == 3:
        states = states[0]
    time_ms = np.arange(states.shape[0]) * dt
    total_units = states.shape[1]
    idx = np.linspace(0, total_units - 1, min(n_units, total_units)).astype(int)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i in idx:
        ax.plot(time_ms, states[:, i], lw=1, label=f"unit {i}")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"$r = \tanh(x)$")
    title = f"unit activity: {len(idx)} of {total_units} units, one condition's trial"
    if condition_label:
        title += f"\ncondition shown: {condition_label}"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
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

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, outputs.shape[0]))
    for i, label in enumerate(labels):
        ax.plot(time_ms, outputs[i], color=colors[i], label=f"{label} (out)")
        ax.plot(time_ms, target[i], color=colors[i], ls="--", alpha=0.5)
    if threshold is not None:
        ax.axhline(threshold, color="k", ls=":", lw=1, label="threshold")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("z")
    ax.set_title("output (solid) vs. target ramp (dashed)")
    ncol = 1 if len(labels) <= 20 else 2
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, ncol=ncol)
    return savefig(fig, name, out_dir)


def pca_trajectories_figure(
    states_by_condition: Mapping[str, np.ndarray],
    n_components: int = 3,
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
        n_components: 2 draws a 2D panel (PC1/PC2); 3 (the default) draws a 3D panel
            (PC1/PC2/PC3) via matplotlib's built-in 3D toolkit.
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

    ncol = 2 if len(labels) <= 10 else 3
    if n_components >= 3:
        fig = plt.figure(figsize=(7.5, 6))
        ax = fig.add_subplot(111, projection="3d")
        for lab in labels:
            traj = (np.asarray(states_by_condition[lab]) - mean) @ components.T  # [time, n_components]
            ls = linestyle_by[lab] if linestyle_by else "-"
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color_by[lab], ls=ls, label=lab, lw=1.5)
            ax.scatter(*traj[0, :3], color=color_by[lab], marker="o", s=25)   # start
            ax.scatter(*traj[-1, :3], color=color_by[lab], marker="s", s=25)  # end
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3", labelpad=18)
        ax.set_title("condition trajectories in PCA space (o=start, sq=end)")
        ax.legend(fontsize=5, loc="best", ncol=ncol)
        # bbox_inches="tight" (the shared savefig()'s default) systematically
        # mis-measures a rotated 3D z-axis label and crops it; use a fixed, generous
        # pad instead for this branch only.
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches=None, pad_inches=0.3)
        plt.close(fig)
        return path

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
    ax.legend(fontsize=5, loc="best", ncol=ncol)
    return savefig(fig, name, out_dir)
