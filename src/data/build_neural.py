"""Build DMFC neural tensors from DANDI 000130.  [FOUNDATION — I own this; plan 1.D]

Runs in the ISOLATED ingestion env (dandi/pynwb/nlb_tools); writes plain numpy
tensors to ``data/processed/`` that the modeling env reads. Modeling code never
imports pynwb/dandi. `nlb_tools`/`dandi` are imported lazily INSIDE the functions so
this module imports fine in the contracts env too.

Verified dataset facts (NLB paper): 1 monkey (Haydn/"H"), 1 session, 54 sorted
units (40 held-in / 14 held-out), ~1,289 trials, events Ready/Set/Go, behavioral
tp. 40 real conditions = prior 2 x ts 5 x effector 2 x direction 2.

PIPELINE
    1. Download dandiset 000130 (dandi) / load NWB (pynwb) — or via nlb_tools.
    2. Align spikes to Ready/Set/Go; bin to rates.
    3. **Average over the 2 target directions** -> the 20-condition comparison set
       (prior x ts x effector), matching src.conditions.
    4. Write [condition, time, unit] + behavioral tp into the store / data/processed,
       tagged with the SAME (ts, prior, effector) metadata as the model side.

VERIFY ON LOAD (do not assume): ts values and effector labels match
src.conditions; both effectors present; per-cell trial counts logged (the prior is
sampled, so cells may be unbalanced) — pool or flag thin cells.

TODO(me): implement against the actual nlb_tools DMFC_RSG API once the ingestion
env is stood up. Keep the direction-averaging explicit and logged.
"""

from __future__ import annotations

from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def build_neural(out_dir: Path = PROCESSED_DIR) -> None:
    """Ingest DMFC_RSG -> data/processed/. Lazy-imports the ingestion stack."""
    try:
        import nlb_tools  # noqa: F401  (real import lands in the ingestion env)
    except ImportError as e:
        raise ImportError(
            "build_neural needs the ingestion env (nlb_tools/dandi/pynwb). "
            "See requirements-ingestion.txt."
        ) from e
    raise NotImplementedError("Foundation: implement DMFC_RSG ingestion (plan 1.D)")
