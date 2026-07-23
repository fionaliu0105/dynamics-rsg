"""Aggregate results-summary figure: valid-response behavior across learning rules.

Reads the per-seed aggregate table written by ``scripts/summarize_runs.py``
(``results/runs_summary.csv``) and produces a two-panel summary figure:

- **Panel A** — per-seed valid-response counts by variant. Each point is one
  seed's ``valid_tp_count`` (out of ``valid_tp_total`` = 20 test conditions);
  the horizontal marker is the per-variant mean. Shows which learning
  rules / inference-step regimes actually produce valid RSG timed responses.
- **Panel B** — best training loss (log axis) versus valid-response count,
  colored by variant. Makes the dissociation explicit: low training loss does
  not imply valid timing behavior (e.g. ``pc_steps20`` reaches low loss yet
  yields zero valid responses).

Consistent with AGENTS.md: this reads saved metrics only and never retrains or
re-extracts. Non-interactive backend; writes files under ``results/figures/``.

Usage::

    python scripts/plot_results_summary.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a compute node (AGENTS.md execution contract)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUMMARY_CSV = Path("results/runs_summary.csv")
OUT_DIR = Path("results/figures")
OUT_NAME = "results_summary"

# Ordered along the locality axis of the learning rule (AGENTS.md, "third arm"):
# BPTT (nonlocal) -> RFLO (local in space+time) -> PC (local in space, relaxed
# over the trajectory) at two inference-step budgets.
VARIANT_ORDER = ["bptt", "rflo", "pc_steps20", "pc_steps100"]
VARIANT_LABELS = {
    "bptt": "BPTT",
    "rflo": "RFLO",
    "pc_steps20": "PC\n(20 steps)",
    "pc_steps100": "PC\n(100 steps)",
}
# One restrained palette: a neutral/cool family for the working rules and a
# muted red reserved for the regime that fails to produce valid responses.
VARIANT_COLORS = {
    "bptt": "#3B6EA5",       # blue
    "rflo": "#E08214",       # amber
    "pc_steps20": "#B2182B",  # red — the failure regime
    "pc_steps100": "#4D9221",  # green
}


def _style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",   # editable text in SVG
        "pdf.fonttype": 42,        # editable TrueType text in PDF
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })


def _save(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")


def _panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Per-seed valid-response counts by variant (jittered strip + mean bar)."""
    rng = np.random.default_rng(0)  # reproducible jitter
    total = int(df["valid_tp_total"].dropna().max()) if df["valid_tp_total"].notna().any() else 20

    for x, variant in enumerate(VARIANT_ORDER):
        sub = df[df["variant"] == variant]
        counts = sub["valid_tp_count"].dropna().to_numpy(dtype=float)
        color = VARIANT_COLORS[variant]
        if counts.size:
            jitter = rng.uniform(-0.16, 0.16, size=counts.size)
            ax.scatter(
                np.full(counts.size, x) + jitter, counts,
                s=26, color=color, alpha=0.75,
                edgecolors="white", linewidths=0.4, zorder=3,
            )
            mean = counts.mean()
            ax.hlines(mean, x - 0.28, x + 0.28, color=color, linewidth=2.2, zorder=4)

    ax.set_xticks(range(len(VARIANT_ORDER)))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER])
    ax.set_ylim(-1, total + 1)
    ax.set_ylabel(f"Valid responses per seed (of {total})")
    ax.set_xlabel("Learning rule / inference regime")
    ax.set_title("A  Valid-response counts by variant", loc="left", fontweight="bold")
    ax.margins(x=0.08)


def _panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Best training loss (log) vs valid-response count, colored by variant."""
    valid = df.dropna(subset=["best_loss", "valid_tp_count"])
    for variant in VARIANT_ORDER:
        sub = valid[valid["variant"] == variant]
        if sub.empty:
            continue
        ax.scatter(
            sub["best_loss"], sub["valid_tp_count"],
            s=30, color=VARIANT_COLORS[variant],
            alpha=0.8, edgecolors="white", linewidths=0.4,
            label=VARIANT_LABELS[variant].replace("\n", " "), zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Best training loss (log scale)")
    ax.set_ylabel("Valid responses per seed")
    ax.set_title("B  Loss does not predict valid behavior", loc="left", fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, handletextpad=0.3, borderaxespad=0.2)


def main() -> int:
    if not SUMMARY_CSV.exists():
        raise SystemExit(f"[plot_results_summary] missing {SUMMARY_CSV}; run scripts/summarize_runs.py first")

    _style()
    df = pd.read_csv(SUMMARY_CSV)

    unknown = sorted(set(df["variant"].unique()) - set(VARIANT_ORDER))
    if unknown:
        print(f"[plot_results_summary] note: variants not plotted (unknown order): {unknown}")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    _panel_a(ax_a, df)
    _panel_b(ax_b, df)
    fig.tight_layout(w_pad=2.5, rect=(0, 0.06, 1, 1))

    caption = (
        "Valid Ready-Set-Go responses per seed across learning rules. "
        "(A) BPTT and RFLO produce valid responses reliably; PC fails at 20 inference "
        "steps and partially recovers at 100. (B) Best training loss (log scale) does not "
        "predict valid behavior. Points are individual seeds; bars in A are per-variant means."
    )
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=6.5,
             wrap=True, color="#333333")

    stem = OUT_DIR / OUT_NAME
    _save(fig, stem)
    plt.close(fig)
    print(f"[plot_results_summary] wrote {stem}.png / .pdf / .svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
