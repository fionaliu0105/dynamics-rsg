"""Activation store: one indexable on-disk home for states + inputs + metadata.

Keyed by ``(model, seed, condition)``. Holds the network ``states`` AND the input
drive ``inputs`` (iDSA needs inputs aligned to states — reconstructing them later
is error-prone) plus per-condition metadata (ts, prior, effector, tp).

The neural data is written into the SAME store with the same condition metadata, so
model and brain sit in one comparable structure. See AGENTS.md ("store/ keeps the
input time series") and ``docs/implementation_plan.md`` 0.4 / 7.

Backend: numpy ``.npz`` per record (h5py in the shared env has a numpy-ABI break).
The public API hides this, so an h5py/zarr single-file backend can drop in later
without touching callers.
"""

from src.store.activation_store import ActivationStore, Record

__all__ = ["ActivationStore", "Record"]
