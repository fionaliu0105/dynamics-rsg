"""Identical preprocessing for model & neural data.  [MEMBER TRACK: Preprocess & RSA — plan 1.E]

THE LINCHPIN INVARIANT. RSA and iDSA compare systems only after BOTH pass through
this one pipeline. If it differs across model and brain, the comparison is invalid.

STEPS (apply the SAME way to every system)
    1. per-unit normalization   (z-score each unit over time x conditions)
    2. project to a shared latent dimensionality k   (PCA to a common k)
    3. warp/bin to matched time bins aligned on task events (Ready/Set/Go)

INTERFACE
    Preprocessor(PreprocessConfig).fit(reference) then .transform(system) so the
    SAME k / time base is applied everywhere. Input/out: [condition, time, units].
    Guarantee: after transform, every system has identical (k, n_time_bins).

DEFINITION OF DONE (plan 1.E check)
    Output dimensionality and time bins match across model and neural data.

DEPENDENCY NOTE: this feeds BOTH RSA (2.3) and iDSA (2.4/2.5). The iDSA member
develops against this stub until the real one lands, so keep the signature stable.

Numpy-only; no torch needed to import this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PreprocessConfig:
    """Knobs shared across all systems so preprocessing is provably identical."""

    k: int = 10                 # shared latent dimensionality (PCA target)
    n_time_bins: int = 25       # matched time bins after warping/binning
    normalize: str = "zscore"   # per-unit normalization scheme


class Preprocessor:
    """Fit shared parameters once, then apply identically to model & neural data."""

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self._fitted = False

    def fit(self, reference: np.ndarray) -> "Preprocessor":
        """Learn normalization + PCA basis + time base. ``reference``: [cond, time, units].

        TODO(preprocess-track): implement. Decide and DOCUMENT whether PCA is fit
        per-system to k dims (RSA-friendly) — the shared quantity is the target k
        and the matched time base, not a shared basis.
        """
        raise NotImplementedError("Preprocess track: implement fit (plan 1.E)")

    def transform(self, system: np.ndarray) -> np.ndarray:
        """Apply the fitted steps. Returns [cond, n_time_bins, k]. TODO(preprocess-track)."""
        raise NotImplementedError("Preprocess track: implement transform (plan 1.E)")
