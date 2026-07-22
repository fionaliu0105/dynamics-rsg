"""tp-vs-ts regression slope per prior.  [MEMBER TRACK: Task & behavior — plan 2.2]

The behavioral signature of Bayesian integration: reproductions regress toward the
prior mean, so the slope of tp on ts is in (0, 1), and the noisier Long prior is
MORE biased (flatter) than Short. We COMPUTE and REPORT this for every seed and
carry it beside every similarity value — it is **never** used to exclude a seed
(AGENTS.md, "Behavior is measured, never a filter"; plan decision 4).

WHAT TO BUILD
    tp(outputs, set_step, cfg) -> produced interval = first threshold crossing after
        Set (skip the Set-triggered transient, ~one pulse width).
    slopes_by_prior(ts, tp, prior_labels) -> {prior: slope} via a linear fit.

DEFINITION OF DONE (plan 2.2 check)
    Slope matches the reconstruction's ``bias_slopes`` on a trained net.

Numpy-only; no torch needed to import this module.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from src.training.config import Config


def tp(outputs: np.ndarray, set_step: int, cfg: Config) -> float:
    """Produced interval in ms: first threshold crossing after Set.

    The search skips one pulse width after Set to avoid scoring a Set-triggered
    transient as the produced response. Returns ``NaN`` if the output never
    reaches ``cfg.threshold`` before trial end.
    """
    y = np.asarray(outputs, dtype=float).reshape(-1)
    start = min(max(set_step + cfg.pulse_width_step, 0), y.shape[0])
    crossed = np.flatnonzero(y[start:] >= cfg.threshold)
    if crossed.size == 0:
        return float("nan")
    crossing_step = start + int(crossed[0])
    return float((crossing_step - set_step) * cfg.dt)


def slopes_by_prior(
    ts: Sequence[float],
    tp: Sequence[float],
    prior_labels: Sequence[str],
) -> Dict[str, float]:
    """Least-squares slope of tp on ts, computed separately within each prior.

    NaN produced intervals are dropped. Priors with fewer than three valid points
    return NaN rather than forcing an unstable regression.
    """
    ts_arr = np.asarray(ts, dtype=float)
    tp_arr = np.asarray(tp, dtype=float)
    priors = np.asarray(prior_labels)
    if ts_arr.shape != tp_arr.shape or ts_arr.shape != priors.shape:
        raise ValueError("ts, tp, and prior_labels must have matching one-dimensional lengths")

    slopes: Dict[str, float] = {}
    for prior in sorted(set(priors.tolist())):
        keep = (priors == prior) & np.isfinite(ts_arr) & np.isfinite(tp_arr)
        if int(keep.sum()) < 3:
            slopes[str(prior)] = float("nan")
        else:
            slopes[str(prior)] = float(np.polyfit(ts_arr[keep], tp_arr[keep], 1)[0])
    return slopes
