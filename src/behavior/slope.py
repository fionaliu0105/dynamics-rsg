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


def slopes_by_prior(
    ts: Sequence[float],
    tp: Sequence[float],
    prior_labels: Sequence[str],
) -> Dict[str, float]:
    """Least-squares slope of tp on ts, computed separately within each prior.

    TODO(behavior-track): group by prior, drop NaN tp, fit ``np.polyfit(ts, tp, 1)``,
    return the slope per prior. Require >= 3 valid points per prior or return NaN.
    """
    raise NotImplementedError("Behavior track: implement slopes_by_prior (plan 2.2)")
