"""Within-rule seed baseline: is a between-rule difference bigger than seed noise?

Every other comparison in ``src/compare`` answers "how far is system A from system
B". None of them answer the prior question: **how far apart are two seeds of the
SAME rule?** Without that number, a between-rule distance is uninterpretable. If two
BPTT seeds are as far apart from each other as BPTT is from PC, then there is no
learning-rule signature and every downstream claim is noise (AGENTS.md, "Seeds are
the unit of evidence").

So this module builds one square matrix per metric whose

* **diagonal** is the mean within-arm, seed-to-seed distance (the null), and
* **off-diagonal** is the mean between-arm distance (the signal).

The claim "the learning rule leaves a signature" is exactly the claim that the
off-diagonal exceeds the diagonal. Reading that off one picture is the point.

Two conventions that keep the comparison honest:

- Within and between cells are both **all unordered pairs**, so the two are computed
  the same way and land in the same units. Using seed-matched pairs between arms but
  all-pairs within would make the diagonal artificially large.
- Systems arrive already routed through ONE shared ``Preprocessor`` instance, same
  as ``scripts/run_rsa.py`` (AGENTS.md, "Identical preprocessing").

No filtering on behavior happens here or anywhere downstream (AGENTS.md, "Behavior is
measured, never a filter").
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


def _pair_stats(values: Sequence[float]) -> Dict[str, float]:
    """Mean / SD / n over a bag of pairwise distances, NaN-safe and empty-safe."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
            "n": int(arr.size)}


def within_between_matrix(
    reps_by_arm: Dict[str, Dict[int, object]],
    distance: Callable[[object, object], float],
) -> Dict[str, object]:
    """Square within/between distance matrix over arms.

    Args:
        reps_by_arm: ``{arm: {seed: representation}}``. A "representation" is
            whatever ``distance`` consumes -- an RDM for RSA, a fitted ``Operators``
            for iDSA. This function is deliberately agnostic so the same null applies
            to geometry and dynamics without duplicating the pairing logic.
        distance: symmetric two-argument distance.

    Returns:
        ``{"arms": [...], "mean": [[...]], "sd": [[...]], "n": [[...]],
           "pairs": {"armA|armB": [raw distances]}}``. Cell (i, i) is the within-arm
        seed-to-seed null for arm i; cell (i, j) is the between-arm distance.
    """
    arms = list(reps_by_arm)
    n_arms = len(arms)
    mean = np.full((n_arms, n_arms), np.nan)
    sd = np.full((n_arms, n_arms), np.nan)
    counts = np.zeros((n_arms, n_arms), dtype=int)
    pairs: Dict[str, List[float]] = {}

    for i, arm_i in enumerate(arms):
        for j, arm_j in enumerate(arms):
            if j < i:
                continue
            reps_i, reps_j = reps_by_arm[arm_i], reps_by_arm[arm_j]
            if i == j:
                # Within-arm null: every unordered pair of DISTINCT seeds.
                combos = [(reps_i[a], reps_i[b]) for a, b in combinations(sorted(reps_i), 2)]
            else:
                # Between-arm: every cross pair, so it is computed like the diagonal.
                combos = [(reps_i[a], reps_j[b]) for a in sorted(reps_i) for b in sorted(reps_j)]
            dists = [distance(x, y) for x, y in combos]
            stats = _pair_stats(dists)
            mean[i, j] = mean[j, i] = stats["mean"]
            sd[i, j] = sd[j, i] = stats["sd"]
            counts[i, j] = counts[j, i] = stats["n"]
            pairs[f"{arm_i}|{arm_j}"] = [float(d) for d in dists]

    return {
        "arms": arms,
        "mean": mean.tolist(),
        "sd": sd.tolist(),
        "n": counts.tolist(),
        "pairs": pairs,
    }


def signature_margin(matrix: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    """Per arm-pair: how far the between-arm distance sits above the two within nulls.

    The learning-rule signature exists for a pair only if the between distance clears
    BOTH arms' seed-to-seed nulls, so we report the margin against the LARGER (more
    conservative) of the two diagonals, in units of the pooled seed-to-seed SD.

    A ``margin_sd`` at or below 0 means: these two rules are no more different from
    each other than two seeds of the same rule are -- i.e. no signature for that pair.
    """
    arms = matrix["arms"]
    mean = np.asarray(matrix["mean"], dtype=float)
    sd = np.asarray(matrix["sd"], dtype=float)
    out: Dict[str, Dict[str, float]] = {}
    for i, j in combinations(range(len(arms)), 2):
        within_max = float(np.nanmax([mean[i, i], mean[j, j]]))
        pooled_sd = float(np.nanmean([sd[i, i], sd[j, j]]))
        between = float(mean[i, j])
        margin = between - within_max
        out[f"{arms[i]}_vs_{arms[j]}"] = {
            "between": between,
            "within_max": within_max,
            "margin": margin,
            "margin_sd": float(margin / pooled_sd) if pooled_sd > 0 else float("nan"),
        }
    return out


def paired_seed_contrast(
    distances_by_arm: Dict[str, Dict[int, float]],
    reference_arm: str,
) -> Dict[str, Dict[str, object]]:
    """Per-seed PAIRED differences in distance-to-DMFC, against a reference arm.

    Seed *N* of every arm starts from bit-identical weights (AGENTS.md, "third arm":
    RFLO's feedback matrix is drawn from a SEPARATE RNG stream precisely so this
    holds). That makes ``arm - reference`` a paired quantity per seed, which is a
    strictly stronger test than comparing two independent clouds of seeds -- the
    shared initialization cancels.

    Only seeds present in BOTH arms are used; the pairing is the whole point, so a
    seed missing on one side is dropped rather than silently unpaired.

    Returns ``{arm: {"seeds": [...], "deltas": [...], "mean": float,
                     "n_favor_reference": int}}`` where a NEGATIVE delta means the arm
    is CLOSER to DMFC than the reference.
    """
    ref = distances_by_arm[reference_arm]
    out: Dict[str, Dict[str, object]] = {}
    for arm, by_seed in distances_by_arm.items():
        if arm == reference_arm:
            continue
        seeds = sorted(set(by_seed) & set(ref))
        deltas = [float(by_seed[s] - ref[s]) for s in seeds]
        out[arm] = {
            "seeds": seeds,
            "deltas": deltas,
            "mean": float(np.mean(deltas)) if deltas else float("nan"),
            "n_favor_reference": int(sum(d > 0 for d in deltas)),
            "n_pairs": len(deltas),
        }
    return out
