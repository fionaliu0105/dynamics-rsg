"""Re-render the RQ1-RQ3 slide figures from cached metrics, in the shared palette.

The signature/geometry/summary drivers recompute distances from the activation store
and the processed DMFC neural data. When only the saved metrics are on hand (this is
the AGENTS.md "plotting reads saved metrics ... never retrains" path), this script
redraws whatever those metrics can support, using the shared colors in
``src.viz.palette`` so the figures stay consistent.

It reads only ``results/signature/signature.json`` (plus the fixed condition schema)
and writes:

* ``results/signature/figures/rq1_rsa_within_between.png`` (RQ1 geometry)
* ``results/signature/figures/rq1_idsa_within_between.png`` (RQ1 dynamics)
* ``results/signature/figures/rq2_per_ts_distance.png`` (RQ2)
* ``results/signature/figures/rq2_overlap_800.png`` (RQ2)
* ``results/signature/figures/rq3_rsa_paired_vs_bptt.png`` (RQ3 paired)
* ``results/signature/figures/rq3_idsa_paired_vs_bptt.png`` (RQ3 paired)
* ``results/rsa/summary_dmfc_comparison.png`` (RQ3 headline, RSA)
* ``results/idsa/summary_dmfc_comparison.png`` (RQ3 headline, iDSA)

It does not redraw the Setup RDM heatmap or galleries. Those need the raw 20x20
condition RDMs (the processed neural data), which are not in signature.json. Once that
data is present, ``scripts/run_rsa_geometry.py`` redraws them in the same colors,
because the colormap now lives in ``src.viz.palette``.

Usage::

    python scripts/plot_slide_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conditions import TS_BY_PRIOR
from src.viz.figures import (
    overlap_separation_figure,
    paired_contrast_figure,
    per_ts_curve_figure,
    summary_distance_figure,
    within_between_matrix_figure,
)

SIGNATURE_JSON = Path("results/signature/signature.json")
SIG_FIG_DIR = Path("results/signature/figures")
RSA_DIR = Path("results/rsa")
IDSA_DIR = Path("results/idsa")

LABELS = {"untrained": "Untrained", "bptt": "BPTT", "rflo": "RFLO",
          "pc_steps100": "PC (100 steps)", "pc_steps20": "PC (20 steps)",
          "DMFC": "DMFC (split halves)"}


def _seed_dict_to_list(by_seed: dict) -> list:
    """{'0': v, '1': v, ...} -> [v, v, ...] in numeric seed order."""
    return [by_seed[k] for k in sorted(by_seed, key=int)]


def main() -> int:
    if not SIGNATURE_JSON.exists():
        raise SystemExit(f"[plot_slide_figures] missing {SIGNATURE_JSON}; run scripts/run_rule_signature.py first")

    sig = json.loads(SIGNATURE_JSON.read_text())
    SIG_FIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # --- RQ1: is there a signature at all? within vs between, per metric -----------
    written.append(within_between_matrix_figure(
        sig["rsa_matrix"], "RSA", SIG_FIG_DIR, "rq1_rsa_within_between", labels=LABELS))
    written.append(within_between_matrix_figure(
        sig["idsa_matrix"], "iDSA", SIG_FIG_DIR, "rq1_idsa_within_between", labels=LABELS))

    # --- RQ2: does interval length / prior modulate the fit? -----------------------
    written.append(per_ts_curve_figure(
        sig["per_ts_curves"], TS_BY_PRIOR, "RSA", SIG_FIG_DIR, "rq2_per_ts_distance",
        ceiling=tuple(sig["noise_ceiling"]), labels=LABELS))
    overlap = sig["overlap_800_separation"]
    written.append(overlap_separation_figure(
        overlap["arms"], overlap.get("dmfc"), SIG_FIG_DIR, "rq2_overlap_800", labels=LABELS))

    # --- RQ3 paired: seed-for-seed, since seed N starts bit-identical across arms ---
    for metric, key in (("RSA", "rsa_paired_vs_bptt"), ("iDSA", "idsa_paired_vs_bptt")):
        name = f"rq3_{metric.lower()}_paired_vs_bptt"
        written.append(paired_contrast_figure(
            sig[key], "BPTT", metric, SIG_FIG_DIR, name, labels=LABELS))

    # --- RQ3 headline: distance-to-DMFC per arm, seeds as points, CIs, ceiling -----
    rsa_dist = {"RSA": {arm: _seed_dict_to_list(v) for arm, v in sig["rsa_to_dmfc"].items()}}
    written.append(summary_distance_figure(
        rsa_dist, out_dir=RSA_DIR, ceilings={"RSA": tuple(sig["noise_ceiling"])},
        title_suffix="distance to DMFC, per seed", name="summary_dmfc_comparison",
        labels=LABELS))
    idsa_dist = {"iDSA": {arm: _seed_dict_to_list(v) for arm, v in sig["idsa_to_dmfc"].items()}}
    written.append(summary_distance_figure(
        idsa_dist, out_dir=IDSA_DIR,
        title_suffix="distance to DMFC, per seed (no ceiling: iDSA noise ceiling not computed)",
        name="summary_dmfc_comparison", labels=LABELS))

    print("[plot_slide_figures] wrote:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
