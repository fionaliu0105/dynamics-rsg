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

WHY PCA IS FIT PER-SYSTEM (not a shared basis) — a deliberate, documented choice.
    A shared PCA basis needs a common feature (unit) space. Model units (N=160/200
    artificial units) and DMFC units (54 sorted neurons) live in different,
    non-corresponding spaces — there is no single rotation that applies to both.
    RSA compares representational *geometry* (condition-by-condition dissimilarity),
    which is invariant to the orthonormal basis chosen WITHIN a system. So each
    system gets its own top-k principal subspace; the quantities that must MATCH
    across systems are the target ``k`` and the ``n_time_bins`` time base, not the
    basis. Consequently ``fit`` learns only the time base + config (and arms the
    standardization gate); it does NOT produce a projection reused by ``transform``.

DEFINITION OF DONE (plan 1.E check)
    Output dimensionality and time bins match across model and neural data.

DEPENDENCY NOTE: this feeds BOTH RSA (2.3) and iDSA (2.4/2.5). The iDSA member
develops against this stub until the real one lands, so the documented
``fit(reference).transform(system)`` signature stays stable; ``transform_inputs`` /
``transform_with_inputs`` are additive (iDSA needs the input drive warped onto the
same time base as the states, so it stays aligned for DMDc).

Numpy-only; no torch needed to import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import numpy as np

from src.conditions import N_CONDITIONS

#: A "system" is either a uniform [cond, time, units] array or a list of per-condition
#: [time, units] arrays (time length varies with ts/tp, so ragged input is allowed).
System = Union[np.ndarray, Sequence[np.ndarray]]


@dataclass
class PreprocessConfig:
    """Knobs shared across all systems so preprocessing is provably identical."""

    k: int = 10                 # shared latent dimensionality (PCA target)
    n_time_bins: int = 25       # matched time bins after warping/binning
    normalize: str = "zscore"   # per-unit normalization scheme: "zscore" | "none"
    warp: str = "resample"      # time-warp mode: "resample" now; "event_anchored" reserved


class Preprocessor:
    """Fit shared parameters once, then apply identically to model & neural data."""

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self._fitted = False

    # --- fit --------------------------------------------------------------------
    def fit(self, reference: System) -> "Preprocessor":
        """Lock the shared time base + config and arm the standardization gate.

        ``reference``: a system, [cond, time, units] or a list of per-condition
        [time, units]. Because PCA is fit per-system (see the module docstring),
        ``fit`` does NOT learn a projection to reuse — it validates the reference,
        records the invariant contract (k, n_time_bins, expected 20 conditions), and
        sets ``_fitted`` so downstream code cannot feed unstandardized data into RSA
        by accident. The honest shared quantities are ``k`` and ``n_time_bins``.
        """
        conds = _as_condition_list(reference)
        if len(conds) != N_CONDITIONS:
            raise ValueError(
                f"reference must have {N_CONDITIONS} conditions, got {len(conds)}"
            )
        if self.cfg.warp not in ("resample", "event_anchored"):
            raise ValueError(f"unknown warp mode {self.cfg.warp!r}")
        if self.cfg.warp == "event_anchored":
            raise NotImplementedError(
                "event_anchored warp is reserved: it needs Ready/Set/Go indices "
                "threaded into BOTH the store meta and neural data/processed (a "
                "cross-track change). Use warp='resample' for now."
            )
        if self.cfg.normalize not in ("zscore", "none"):
            raise ValueError(f"unknown normalize scheme {self.cfg.normalize!r}")
        self._fitted = True
        return self

    # --- transform (states) -----------------------------------------------------
    def transform(self, system: System) -> np.ndarray:
        """Apply z-score -> per-system PCA(k) -> time warp. Returns [cond, n_time_bins, k].

        Steps (identical for every system):
            1. per-unit z-score, stats pooled over (condition x time);
            2. PCA to k via numpy SVD on this system's own pooled matrix, zero-padded
               to exactly k columns when units < k;
            3. linear resample of each condition to ``n_time_bins`` over [0, 1] time.
        """
        if not self._fitted:
            raise RuntimeError(
                "call fit(reference) before transform: preprocessing must be "
                "standardized before RSA/iDSA (AGENTS.md, 'Identical preprocessing')"
            )
        conds = _as_condition_list(system)
        if len(conds) != N_CONDITIONS:
            raise ValueError(
                f"system must have {N_CONDITIONS} conditions, got {len(conds)}"
            )

        # 1. per-unit z-score, stats pooled over condition x time.
        if self.cfg.normalize == "zscore":
            conds = _zscore_per_unit(conds)

        # 2. per-system PCA to k via SVD (fit on this system's own data).
        conds = _pca_to_k(conds, self.cfg.k)

        # 3. warp each condition to the matched time base.
        warped = [_resample_time(c, self.cfg.n_time_bins) for c in conds]

        out = np.stack(warped, axis=0)
        assert out.shape == (N_CONDITIONS, self.cfg.n_time_bins, self.cfg.k), out.shape
        return out

    # --- transform (inputs, for iDSA) -------------------------------------------
    def transform_inputs(self, inputs: System) -> np.ndarray:
        """Warp the input drive onto the matched time base. Returns [cond, n_time_bins, n_in].

        Warp ONLY — no z-score, no PCA. The input drive is the physical Ready/Set
        stimulus; iDSA/DMDc needs it sample-aligned to the warped states, so it goes
        through the SAME per-condition resample map and nothing else.
        """
        if not self._fitted:
            raise RuntimeError("call fit(reference) before transform_inputs")
        conds = _as_condition_list(inputs)
        if len(conds) != N_CONDITIONS:
            raise ValueError(
                f"inputs must have {N_CONDITIONS} conditions, got {len(conds)}"
            )
        warped = [_resample_time(c, self.cfg.n_time_bins) for c in conds]
        return np.stack(warped, axis=0)

    def transform_with_inputs(
        self, states: System, inputs: System
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience for iDSA: warped states [cond, n_time_bins, k] AND warped inputs
        [cond, n_time_bins, n_in], guaranteed on the same time base."""
        return self.transform(states), self.transform_inputs(inputs)

    def fit_transform(self, system: System) -> np.ndarray:
        return self.fit(system).transform(system)


# --- helpers (module-level, numpy-only) ----------------------------------------

def _as_condition_list(system: System) -> List[np.ndarray]:
    """Coerce a system to a list of per-condition [time, features] float64 arrays.

    Accepts a uniform 3D [cond, time, feat] array or a ragged sequence of 2D arrays,
    so conditions with different trial lengths (varying ts/tp) are both supported.
    """
    if isinstance(system, np.ndarray) and system.ndim == 3:
        conds = [np.asarray(system[i], dtype=np.float64) for i in range(system.shape[0])]
    else:
        conds = [np.asarray(c, dtype=np.float64) for c in system]
    for i, c in enumerate(conds):
        if c.ndim != 2:
            raise ValueError(f"condition {i} must be [time, features], got shape {c.shape}")
        if not np.isfinite(c).all():
            raise ValueError(f"condition {i} contains non-finite values")
    n_feat = conds[0].shape[1]
    for i, c in enumerate(conds):
        if c.shape[1] != n_feat:
            raise ValueError(
                f"all conditions must share the feature axis: condition {i} has "
                f"{c.shape[1]} features, expected {n_feat}"
            )
    return conds


def _zscore_per_unit(conds: List[np.ndarray]) -> List[np.ndarray]:
    """Z-score each unit using mean/std pooled over ALL conditions and time.

    A unit's scale/offset is a property of the unit across the whole dataset, not of
    any one condition; pooling over condition x time removes activation-scale
    differences symmetrically for model and neural data. Dead units (std == 0) map to
    zeros (a constant unit carries no geometry) rather than NaN.
    """
    pooled = np.concatenate(conds, axis=0)          # [sum_t, units]
    mean = pooled.mean(axis=0, keepdims=True)
    std = pooled.std(axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    return [(c - mean) / std for c in conds]


def _pca_to_k(conds: List[np.ndarray], k: int) -> List[np.ndarray]:
    """Project each condition onto this system's own top-k PCs (numpy SVD).

    Fit on the pooled [sum_t, units] matrix, project every condition, and zero-pad to
    exactly ``k`` columns when units < k so every system exits with the same k.
    """
    pooled = np.concatenate(conds, axis=0)          # [sum_t, units]
    units = pooled.shape[1]
    k_eff = min(k, units)
    # Center before SVD (post-zscore mean is ~0, but center defensively for "none").
    mean = pooled.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(pooled - mean, full_matrices=False)
    components = vt[:k_eff]                          # [k_eff, units]
    out = []
    for c in conds:
        proj = (c - mean) @ components.T            # [time, k_eff]
        if k_eff < k:
            pad = np.zeros((proj.shape[0], k - k_eff), dtype=proj.dtype)
            proj = np.concatenate([proj, pad], axis=1)
        out.append(proj)
    return out


def _resample_time(series: np.ndarray, n_time_bins: int) -> np.ndarray:
    """Linearly resample a [time, feat] series to [n_time_bins, feat] over [0, 1] time.

    Handles variable-length inputs uniformly. A single time sample is broadcast across
    all bins (a degenerate but well-defined case).
    """
    t_in = series.shape[0]
    if t_in == n_time_bins:
        return series.copy()
    if t_in == 1:
        return np.repeat(series, n_time_bins, axis=0)
    t_old = np.linspace(0.0, 1.0, t_in)
    t_new = np.linspace(0.0, 1.0, n_time_bins)
    out = np.empty((n_time_bins, series.shape[1]), dtype=series.dtype)
    for j in range(series.shape[1]):
        out[:, j] = np.interp(t_new, t_old, series[:, j])
    return out
