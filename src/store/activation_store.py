"""Activation store, keyed by (model, seed, condition). Numpy-only backend.

On-disk layout (one ``.npz`` per record, under a store directory)::

    {root}/{model}/seed_{seed:04d}/{condition_key}.npz
        states : [time, units]     (condition-averaged, or per-trial-mean)
        inputs : [time, n_in]      the aligned external drive
        meta   : JSON string        prior, ts, effector, tp, ...

**Why ``.npz`` and not HDF5/zarr:** the public API here (``write`` / ``read`` /
``has`` / ``keys``) hides the backend, so the on-disk format is an implementation
detail. We default to ``.npz`` because it needs only numpy — the h5py build in the
shared anaconda env has a numpy-ABI mismatch, and per AGENTS.md we do not
force-upgrade a shared pin to satisfy one package. Swapping in an h5py single-file
or zarr backend later touches ONLY this file. (See ``docs/implementation_plan.md``
0.4, "zarr or hdf5"; the store contract is what matters, not the container.)

Design choices that outlive the backend:

- **Idempotent writes.** Re-writing an existing (model, seed, condition) overwrites
  it in place — re-running a completed seed is a cheap no-op, which is what makes
  requeuing a SLURM array safe (AGENTS.md, "Assume the process can die").
- Keys come from :class:`src.conditions.Condition`, never ad-hoc strings.
- ``states`` and ``inputs`` are stored together and share a time axis, because iDSA
  needs the input series aligned to the states.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import numpy as np

from src.conditions import Condition, condition_by_key


@dataclass
class Record:
    """One (model, seed, condition) entry: states, aligned inputs, and metadata."""

    model: str
    seed: int
    condition: Condition
    states: np.ndarray                      # [time, units]
    inputs: np.ndarray                      # [time, n_in]
    meta: Dict[str, Any] = field(default_factory=dict)


class ActivationStore:
    """Read/write interface over the on-disk store. ``root`` is a directory.

    Example::

        store = ActivationStore("results/activations")
        store.write(Record("bptt", 0, cond, states, inputs, {"tp": 812.0}))
        rec = store.read("bptt", 0, cond)
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    # --- write ------------------------------------------------------------------
    def write(self, record: Record) -> None:
        states = np.asarray(record.states)
        inputs = np.asarray(record.inputs)
        if states.ndim != 2:
            raise ValueError(f"states must be [time, units], got shape {states.shape}")
        if inputs.ndim != 2:
            raise ValueError(f"inputs must be [time, n_in], got shape {inputs.shape}")
        if states.shape[0] != inputs.shape[0]:
            raise ValueError(
                f"states and inputs must share the time axis: "
                f"{states.shape[0]} != {inputs.shape[0]}"
            )
        meta = {
            "prior": record.condition.prior,
            "ts": record.condition.ts,
            "effector": record.condition.effector,
            "condition_key": record.condition.key,
            **record.meta,
        }
        path = self._record_path(record.model, record.seed, record.condition)
        path.parent.mkdir(parents=True, exist_ok=True)
        # savez to a temp name then replace -> atomic-ish, survives interruption.
        # NB: np.savez appends ".npz" unless the name already ends in it, so the
        # temp name must end in ".npz" or the replace target goes missing.
        tmp = path.parent / f"{path.stem}.tmp.npz"
        np.savez(tmp, states=states, inputs=inputs, meta=json.dumps(meta))
        tmp.replace(path)

    # --- read -------------------------------------------------------------------
    def read(self, model: str, seed: int, condition: Condition) -> Record:
        path = self._record_path(model, seed, condition)
        if not path.exists():
            raise KeyError(f"no record at {path}")
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            return Record(
                model=model,
                seed=seed,
                condition=condition,
                states=z["states"],
                inputs=z["inputs"],
                meta=meta,
            )

    def has(self, model: str, seed: int, condition: Condition) -> bool:
        return self._record_path(model, seed, condition).exists()

    def keys(self) -> Iterator[Tuple[str, int, Condition]]:
        """Yield every (model, seed, condition) present in the store."""
        if not self.root.exists():
            return
        for model_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for seed_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                seed = int(seed_dir.name.split("_")[1])
                for rec in sorted(seed_dir.glob("*.npz")):
                    yield model_dir.name, seed, condition_by_key(rec.stem)

    # --- helpers ----------------------------------------------------------------
    def _record_path(self, model: str, seed: int, condition: Condition) -> Path:
        return self.root / model / f"seed_{seed:04d}" / f"{condition.key}.npz"
