"""Smoke test for the Fig 1E behavior panel (plan Step D). Renders to a temp dir.

Needs matplotlib (present); the viz module forces the headless Agg backend. Run from
the repo root::

    python tests/test_viz_behavior.py
    pytest tests/test_viz_behavior.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.viz.figures import behavior_slope_figure


def test_behavior_slope_figure_renders():
    ts = [480, 560, 640, 720, 800, 800, 900, 1000, 1100, 1200]
    labels = ["short"] * 5 + ["long"] * 5
    # synthetic biased behavior: slope ~0.7 (short), ~0.5 (long, flatter), one NaN drop
    tp = [0.7 * t + 120 for t in ts[:5]] + [0.5 * t + 300 for t in ts[5:]]
    tp[2] = np.nan
    with tempfile.TemporaryDirectory() as d:
        path = behavior_slope_figure(ts, tp, labels, out_dir=Path(d))
        assert path.exists() and path.stat().st_size > 0
    print("test_behavior_slope_figure_renders OK")


def main():
    test_behavior_slope_figure_renders()
    print("\nbehavior viz smoke test passed")


if __name__ == "__main__":
    main()
