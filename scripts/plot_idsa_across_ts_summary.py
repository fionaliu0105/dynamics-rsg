"""Combined ts-band-resolved iDSA figure: short vs long prior distance-to-DMFC, all arms.

Reads the per-arm across_ts.json artifacts scripts/run_idsa_operators.py already wrote
(no recomputation) and reuses src.viz.figures.summary_distance_figure (the same
function the plain distance-to-DMFC summaries use) with "short"/"long" as the two
metric panels and each arm as a bar within them.

Usage::

    python scripts/plot_idsa_across_ts_summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.viz.figures import summary_distance_figure

IDSA_DIR = Path("results/idsa")
# Untrained control first, then the locality axis BPTT -> PC -> RFLO. Without the
# control a distance-to-DMFC bar has no floor: a random RNN with this architecture
# and this input drive already sits at some finite distance from DMFC.
ARMS = ["untrained", "bptt", "pc_steps20", "pc_steps100", "rflo"]


def main() -> int:
    distances = {"short": {}, "long": {}}
    for arm in ARMS:
        payload = json.loads((IDSA_DIR / f"{arm}_vs_dmfc" / "across_ts.json").read_text())
        for band in ("short", "long"):
            distances[band][arm] = [v["distance"] for v in payload[band].values()]

    path = summary_distance_figure(
        distances, out_dir=IDSA_DIR,
        title_suffix="distance to DMFC, per seed", name="across_ts_all_arms",
    )
    print(f"[plot_idsa_across_ts_summary] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
