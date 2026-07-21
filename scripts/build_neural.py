"""Thin entry point for DMFC neural ingestion (plan 1.D).

Runs in the ISOLATED ingestion env (dandi/pynwb/nlb_tools). Same code path
interactively or under SLURM -- it only sets up ``sys.path`` and delegates to
``src.data.build_neural.main`` (no cluster values baked in).

    python scripts/build_neural.py --out-dir data/processed
    python -m src.data.build_neural --out-dir data/processed   # equivalent
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build_neural import main

if __name__ == "__main__":
    raise SystemExit(main())
