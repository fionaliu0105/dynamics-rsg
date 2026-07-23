"""Combined RSA/iDSA distance-to-DMFC figure: bptt, rflo, pc_steps20, pc_steps100.

Reads the per-arm JSON outputs already written by scripts/run_rsa.py --neural and
scripts/run_idsa.py --neural (AGENTS.md, "Plotting reads saved metrics ... never
retrains") and merges them into one bar per arm per metric, mirroring
scripts/plot_rsa_idsa_summary.py's model-to-model pattern but against DMFC, with the
RSA noise ceiling band included.

Usage::

    python scripts/plot_rsa_idsa_dmfc_summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.viz.figures import summary_distance_figure

RSA_DIR = Path("results/rsa")
IDSA_DIR = Path("results/idsa")
# Untrained control first, then the locality axis BPTT -> PC -> RFLO. Without the
# control a distance-to-DMFC bar has no floor: a random RNN with this architecture
# and this input drive already sits at some finite distance from DMFC.
ARMS = ["untrained", "bptt", "pc_steps20", "pc_steps100", "rflo"]


def main() -> int:
    rsa_distances = {"RSA": {}}
    ceilings = None
    for arm in ARMS:
        payload = json.loads((RSA_DIR / f"{arm}_vs_dmfc" / "rsa_distances.json").read_text())
        rule_key = "pc" if arm.startswith("pc_steps") else arm
        rsa_distances["RSA"][arm] = payload["distances"]["RSA"][rule_key]
        if ceilings is None and payload.get("noise_ceiling"):
            ceilings = payload["noise_ceiling"]

    idsa_distances = {"iDSA": {}}
    for arm in ARMS:
        payload = json.loads((IDSA_DIR / f"{arm}_vs_dmfc" / "idsa_distances.json").read_text())
        rule_key = "pc" if arm.startswith("pc_steps") else arm
        idsa_distances["iDSA"][arm] = payload["distances"][rule_key]

    rsa_path = summary_distance_figure(
        rsa_distances, out_dir=RSA_DIR, ceilings=ceilings,
        title_suffix="distance to DMFC, per seed",
        name="summary_dmfc_comparison",
    )
    idsa_path = summary_distance_figure(
        idsa_distances, out_dir=IDSA_DIR,
        title_suffix="distance to DMFC, per seed (no ceiling: iDSA noise ceiling not computed)",
        name="summary_dmfc_comparison",
    )
    print(f"[plot_rsa_idsa_dmfc_summary] wrote {rsa_path} and {idsa_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
