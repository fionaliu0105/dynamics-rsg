"""RSA — representational geometry.  [MEMBER TRACK: Preprocess & RSA — plan 2.3]

Build condition-averaged RDMs over the 20 conditions (prior x ts x effector) and
compare geometries. TWO comparisons, same shape of output:
    * rule-vs-rule  : BPTT RDM vs PC RDM               (available in the prototype)
    * model-to-DMFC : each model's RDM vs the DMFC RDM (needs neural ingestion)
with a noise ceiling estimated from neural split-halves. Return PER-SEED distances
(never a point estimate — AGENTS.md, "Seeds are the unit of evidence").

INPUT: preprocessed [condition, time, k] tensors (from src.preprocess), so both
systems already share k and the time base, and axis 0 is the CANONICAL condition
order (src.conditions.CONDITIONS). These functions are pure geometry — no store or
DANDI I/O (that lives in scripts/run_rsa.py).

METRICS (confirmed with the track owner)
    RDM cell        = 1 - Pearson between condition patterns (correlation distance)
    RDM-vs-RDM      = 1 - Spearman over the upper triangle (rank-based, robust to
                      monotonic scale differences between a model and the brain)

DEFINITION OF DONE (plan 2.3 check)
    Runs on neural data and on a surrogate/other-RNN; dtypes and standardized inputs
    align in time; RDMs are 20x20 over the canonical condition order.

REFERENCE: rsatoolbox — https://rsatoolbox.readthedocs.io/  (used only optionally,
behind a lazy import, for the noise ceiling; a numpy fallback keeps this testable
without it in the ABI-broken base env.)
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.conditions import N_CONDITIONS


def build_rdm(activity: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """Condition x condition RDM from preprocessed [cond, time, k] activity.

    Each condition's [time, k] pattern is flattened to a vector, then the 20x20
    dissimilarity is computed in the canonical condition order (axis 0 of ``activity``
    is already that order — the caller stacks via src.conditions.CONDITIONS).

    Args:
        activity: [N_CONDITIONS, time, k] preprocessed tensor.
        metric: "correlation" (1 - Pearson) or "euclidean".

    Returns:
        [20, 20] symmetric matrix, zero diagonal.
    """
    activity = np.asarray(activity, dtype=np.float64)
    if activity.ndim != 3 or activity.shape[0] != N_CONDITIONS:
        raise ValueError(
            f"activity must be [{N_CONDITIONS}, time, k], got shape {activity.shape}"
        )
    patterns = activity.reshape(N_CONDITIONS, -1)   # [20, time*k]

    if metric == "correlation":
        # 1 - Pearson correlation between condition pattern vectors.
        c = np.corrcoef(patterns)                   # [20, 20]
        rdm = 1.0 - c
    elif metric == "euclidean":
        diff = patterns[:, None, :] - patterns[None, :, :]
        rdm = np.sqrt((diff ** 2).sum(axis=-1))
    else:
        raise ValueError(f"unknown metric {metric!r}; use 'correlation' or 'euclidean'")

    # Kill float asymmetry / diagonal drift.
    rdm = 0.5 * (rdm + rdm.T)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def rdm_distance(
    model_rdm: np.ndarray, ref_rdm: np.ndarray, method: str = "spearman"
) -> float:
    """Second-order distance between two RDMs over their upper triangles.

    Args:
        method: "spearman" (1 - rank correlation, default) or "pearson".

    Returns:
        Distance in [0, 2]; identical RDMs -> 0. NaN if a triangle has zero variance.
    """
    model_rdm = np.asarray(model_rdm, dtype=np.float64)
    ref_rdm = np.asarray(ref_rdm, dtype=np.float64)
    if model_rdm.shape != ref_rdm.shape or model_rdm.shape[0] != model_rdm.shape[1]:
        raise ValueError(
            f"RDMs must be square and equal-shaped, got {model_rdm.shape} vs {ref_rdm.shape}"
        )
    iu = np.triu_indices(model_rdm.shape[0], k=1)
    a = model_rdm[iu]
    b = ref_rdm[iu]
    if method == "spearman":
        a = _rankdata(a)
        b = _rankdata(b)
    elif method != "pearson":
        raise ValueError(f"unknown method {method!r}; use 'spearman' or 'pearson'")
    return float(1.0 - _pearson(a, b))


def noise_ceiling(
    neural_splits: Sequence[np.ndarray],
    metric: str = "correlation",
    method: str = "spearman",
    use_rsatoolbox: bool = False,
) -> Tuple[float, float]:
    """Upper/lower noise ceiling from neural split RDMs, in distance units.

    ``neural_splits``: a sequence of preprocessed [cond, time, k] tensors, one per
    trial split (e.g. split-halves or trial subgroups). We build one RDM per split,
    then:
        upper = mean distance of each split RDM to the grand-mean RDM (includes it);
        lower = mean distance of each split RDM to the leave-one-out mean RDM.
    Returned as distances (1 - correlation) so they overlay directly on the
    distance-to-DMFC summary figure. Guaranteed ``lower <= upper``.

    ``use_rsatoolbox`` opts into rsatoolbox's ceiling if it is importable; otherwise
    (and by default) the numpy path above runs, so this is testable without it.
    """
    rdms = [build_rdm(s, metric=metric) for s in neural_splits]
    n = len(rdms)
    if n < 2:
        raise ValueError("noise ceiling needs at least 2 neural splits")

    if use_rsatoolbox:
        ceil = _noise_ceiling_rsatoolbox(rdms, method=method)
        if ceil is not None:
            return ceil  # else fall through to numpy

    stack = np.stack(rdms, axis=0)                  # [n, 20, 20]
    grand = stack.mean(axis=0)
    upper = float(np.mean([1.0 - rdm_distance(r, grand, method=method) for r in rdms]))
    loo_dists = []
    for i, r in enumerate(rdms):
        loo = (stack.sum(axis=0) - r) / (n - 1)
        loo_dists.append(1.0 - rdm_distance(r, loo, method=method))
    lower = float(np.mean(loo_dists))
    # Report in distance units; keep lower <= upper.
    return (1.0 - upper, 1.0 - lower)


def rsa_distances_per_seed(
    systems_by_rule: Dict[str, Dict[int, np.ndarray]],
    reference: Optional[np.ndarray] = None,
    method: str = "spearman",
    metric: str = "correlation",
) -> Dict[str, List[float]]:
    """Per-seed RDM distances, as ``{rule: [distance per seed]}``.

    Args:
        systems_by_rule: ``{rule: {seed: preprocessed [cond, time, k]}}``.
        reference: a preprocessed [cond, time, k] tensor (e.g. DMFC). If given, each
            model seed is compared to the reference RDM (model-to-DMFC). If None, the
            rules are compared to each other seed-by-seed (rule-vs-rule): seeds are
            paired by their shared seed id.
        method, metric: passed to rdm_distance / build_rdm.

    Returns:
        ``{rule: [per-seed distances]}`` — the array src.viz.figures.summary_distance
        _figure consumes. For rule-vs-rule the series are keyed by rule PAIR, one entry
        per unordered pair in sorted order: two rules give ``{"bptt_vs_pc": [...]}``,
        three give ``bptt_vs_pc``, ``bptt_vs_rflo`` and ``pc_vs_rflo``. Enumerating all
        pairs (rather than requiring exactly two rules) is what lets a third learning
        rule enter this diagnostic without callers having to invoke it pairwise.
    """
    if reference is not None:
        ref_rdm = build_rdm(reference, metric=metric)
        out: Dict[str, List[float]] = {}
        for rule, by_seed in systems_by_rule.items():
            dists = []
            for seed in sorted(by_seed):
                rdm = build_rdm(by_seed[seed], metric=metric)
                dists.append(rdm_distance(rdm, ref_rdm, method=method))
            out[rule] = dists
        return out

    # rule-vs-rule: every unordered pair of rules, paired by shared seed id.
    rules = sorted(systems_by_rule)
    if len(rules) < 2:
        raise ValueError(
            f"rule-vs-rule needs at least 2 rules, got {rules}; pass a reference for "
            f"model-to-reference instead"
        )
    # Cache each seed's RDM so a rule appearing in several pairs is only built once.
    rdms = {
        rule: {seed: build_rdm(system, metric=metric) for seed, system in by_seed.items()}
        for rule, by_seed in systems_by_rule.items()
    }
    out: Dict[str, List[float]] = {}
    for ra, rb in itertools.combinations(rules, 2):
        shared = sorted(set(rdms[ra]) & set(rdms[rb]))
        out[f"{ra}_vs_{rb}"] = [
            rdm_distance(rdms[ra][seed], rdms[rb][seed], method=method) for seed in shared
        ]
    return out


# --- numpy stats helpers -------------------------------------------------------

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; NaN if either vector has zero variance."""
    if a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), matching scipy.stats.rankdata's default.

    Needed so Spearman = Pearson-of-ranks is correct in the presence of tied
    dissimilarities (common when RDM cells coincide).
    """
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # Average ranks within tie groups.
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def _noise_ceiling_rsatoolbox(rdms, method: str):
    """Optional rsatoolbox ceiling; returns (lower_dist, upper_dist) or None if unavailable."""
    try:
        import rsatoolbox  # noqa: F401
    except Exception:
        return None
    try:
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.inference import boot_noise_ceiling
        n = rdms[0].shape[0]
        iu = np.triu_indices(n, k=1)
        vecs = np.stack([r[iu] for r in rdms], axis=0)
        rdm_obj = RDMs(dissimilarities=vecs)
        corr = "spearman" if method == "spearman" else "pearson"
        lower, upper = boot_noise_ceiling(rdm_obj, method=corr)
        # rsatoolbox returns correlations; convert to distances.
        return (float(1.0 - upper), float(1.0 - lower))
    except Exception:
        return None
