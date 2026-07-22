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


def tp(outputs, set_step, cfg):
    """Produced interval(s): time from Set to the first threshold crossing, in ms.

    The network signals "Go" when its readout ``outputs`` (the ramp ``z``) first
    crosses ``cfg.threshold``; the produced interval ``tp`` is the Set→Go duration.
    We ignore the first ``cfg.pulse_width_step`` samples after Set so the Set-triggered
    input transient cannot be mistaken for the crossing. Sub-step precision comes from
    linearly interpolating between the two samples that straddle the threshold.

    Args:
        outputs: readout ``z``, shape ``[time]`` (one trial) or ``[trials, time]``.
        set_step: index of the Set event — a scalar (applied to every trial) or a
            per-trial 1-D array of length ``trials``.
        cfg: run config; uses ``pulse_width_step``, ``threshold``, ``dt``.

    Returns:
        Produced interval in ms — a float for 1-D input, else a ``[trials]`` array.
        ``NaN`` wherever the readout never crosses threshold after Set.
    """
    z = np.asarray(outputs, dtype=float)
    squeeze = z.ndim == 1
    if squeeze:
        z = z[None, :]
    if z.ndim != 2:
        raise ValueError(f"outputs must be [time] or [trials, time], got {z.shape}")
    n_trials, n_time = z.shape

    set_steps = np.broadcast_to(np.asarray(set_step), (n_trials,)).astype(int)
    skip = int(cfg.pulse_width_step)
    thr = float(cfg.threshold)

    out = np.full(n_trials, np.nan, dtype=float)
    for i in range(n_trials):
        s = int(set_steps[i])
        start = s + skip
        if start >= n_time:
            continue
        crossed = np.flatnonzero(z[i, start:] >= thr)
        if crossed.size == 0:
            continue
        idx = start + int(crossed[0])
        # Sub-step precision: interpolate between the straddling samples. Guard on
        # ``z[idx-1] < thr`` so a Set transient that is already above threshold at the
        # first checked sample is not "interpolated" against.
        pos = float(idx)
        if idx > 0 and z[i, idx - 1] < thr:
            denom = z[i, idx] - z[i, idx - 1]
            if denom > 0:
                pos = (idx - 1) + (thr - z[i, idx - 1]) / denom
        out[i] = (pos - s) * cfg.dt

    return float(out[0]) if squeeze else out


def slopes_by_prior(
    ts: Sequence[float],
    tp: Sequence[float],
    prior_labels: Sequence[str],
) -> Dict[str, float]:
    """Least-squares slope of tp on ts, computed separately within each prior.

    Groups by prior, drops trials whose ``tp`` is ``NaN`` (no threshold crossing),
    and fits ``tp = slope * ts + b`` per prior via ``np.polyfit``. A prior needs
    >= 3 valid points with some spread in ts, otherwise its slope is ``NaN``.

    This is a REPORTED covariate, never a filter (AGENTS.md; plan decision 4): the
    caller carries the slope beside each seed's similarity — it does not gate seeds.
    """
    ts = np.asarray(ts, dtype=float)
    tp = np.asarray(tp, dtype=float)
    labels = np.asarray(prior_labels)
    if not (ts.shape == tp.shape == labels.shape):
        raise ValueError(
            f"ts, tp, prior_labels must share shape; got {ts.shape}, {tp.shape}, "
            f"{labels.shape}"
        )

    slopes: Dict[str, float] = {}
    for prior in dict.fromkeys(labels.tolist()):     # unique labels, first-seen order
        m = (labels == prior) & np.isfinite(tp)
        ts_g, tp_g = ts[m], tp[m]
        if ts_g.size >= 3 and np.ptp(ts_g) > 0:
            slopes[prior] = float(np.polyfit(ts_g, tp_g, 1)[0])
        else:
            slopes[prior] = float("nan")
    return slopes
