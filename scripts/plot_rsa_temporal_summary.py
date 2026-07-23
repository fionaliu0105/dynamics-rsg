"""Combined time-resolved RSA figure: all four arms' distance-to-DMFC curves overlaid.

Reads the per-arm artifacts scripts/run_rsa_geometry.py already wrote (no retraining,
no re-extraction) and calls src.viz.figures.rsa_temporal_figure once with all four
arms together, instead of the four separate single-arm figures that script produces.

Usage::

    python scripts/plot_rsa_temporal_summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.viz.figures import rsa_temporal_figure

RSA_DIR = Path("results/rsa")
# Untrained control first, then the locality axis BPTT -> PC -> RFLO. Without the
# control a distance-to-DMFC bar has no floor: a random RNN with this architecture
# and this input drive already sits at some finite distance from DMFC.
ARMS = ["untrained", "bptt", "pc_steps20", "pc_steps100", "rflo"]


def main() -> int:
    curves = {}
    n_time_bins = None
    total_time_ms = None
    for arm in ARMS:
        arm_dir = RSA_DIR / f"{arm}_vs_dmfc"
        curves[arm] = np.load(arm_dir / "temporal_distance.npy")
        meta = json.loads((arm_dir / "geometry_meta.json").read_text())
        n_time_bins = meta["n_time_bins"]
        total_time_ms = meta["total_time_ms"]

    payload = json.loads((RSA_DIR / "bptt_vs_dmfc" / "rsa_distances.json").read_text())
    ceiling = payload["noise_ceiling"]["RSA"] if payload.get("noise_ceiling") else None

    times_ms = (np.arange(n_time_bins) + 0.5) * (total_time_ms / n_time_bins)
    path = rsa_temporal_figure(
        times_ms, curves, ceiling=ceiling, out_dir=RSA_DIR, name="temporal_all_arms",
    )
    print(f"[plot_rsa_temporal_summary] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
