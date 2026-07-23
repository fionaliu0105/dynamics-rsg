"""Combined RSA/iDSA comparison figure: pc_steps20 vs pc_steps100, side by side.

Reads the per-variant JSON outputs already written by scripts/run_rsa.py and
scripts/run_idsa.py (AGENTS.md, "Plotting reads saved metrics ... never
retrains") and merges them into one bar per variant per metric, rather than
the two separate single-bar figures each run produces on its own.

Usage::

    python scripts/plot_rsa_idsa_summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.viz.figures import summary_distance_figure

RSA_DIR = Path("results/rsa")
IDSA_DIR = Path("results/idsa")
VARIANTS = ["bptt_vs_pc_steps20", "bptt_vs_pc_steps100"]


def main() -> int:
    rsa_distances = {"RSA": {}}
    for variant in VARIANTS:
        payload = json.loads((RSA_DIR / variant / "rsa_distances.json").read_text())
        label = variant.replace("bptt_vs_", "")
        rsa_distances["RSA"][label] = payload["distances"]["RSA"]["bptt_vs_pc"]

    idsa_distances = {"iDSA": {}}
    for variant in VARIANTS:
        payload = json.loads((IDSA_DIR / variant / "idsa_distances.json").read_text())
        label = variant.replace("bptt_vs_", "")
        idsa_distances["iDSA"][label] = [d["distance"] for d in payload["per_seed"].values()]

    rsa_path = summary_distance_figure(
        rsa_distances, out_dir=RSA_DIR,
        title_suffix="BPTT vs PC, per seed (model-to-model, no DMFC)",
        name="summary_pc_steps_comparison",
    )
    idsa_path = summary_distance_figure(
        idsa_distances, out_dir=IDSA_DIR,
        title_suffix="BPTT vs PC, per seed (model-to-model, no DMFC)",
        name="summary_pc_steps_comparison",
    )
    print(f"[plot_rsa_idsa_summary] wrote {rsa_path} and {idsa_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
