"""Two-prior Ready-Set-Go task generator.  [MEMBER TRACK: Task & behavior — plan 1.A]

A trial's input has THREE channels (``Model.N_IN == 3``):
    0. Ready/Set pulse channel  (two ``pulse_height`` pulses ``t_m`` apart)
    1. prior-context channel    (tonic ``cfg.prior_context[prior]``)
    2. effector-context channel (tonic ``cfg.effector_context[effector]``)
and a target ramp that reaches ``cfg.threshold`` at ``ts`` after Set (Eq. 9).

Conditions are drawn from :data:`src.conditions.CONDITIONS`. During TRAINING the
Ready-Set separation is jittered with scalar noise ``t_m ~ N(ts, ts * cfg.w_m)`` while
the target stays timed to the TRUE ``ts`` — that averaging is what bends the learned
mapping toward the prior mean (the Bayesian bias).

IMPLEMENTATION NOTE — standalone generator (Blocker A fallback, NOT the NeuroGym subclass)
    AGENTS/plan Decision 1 want ``TwoPriorRSG(neurogym.envs.ReadySetGo)`` ("NeuroGym is
    the task source of truth"). neurogym cannot be installed/tested in this env (py3.9;
    neurogym 2.3.1 needs py>=3.10 + numpy 2, which fights torch/the analysis stack — see
    PLAN_TRACK1.md Blocker A). This module is the documented, reversible fallback: a
    self-contained numpy generator that mirrors ReadySetGo's
    ``fixation->ready->measure->set->production`` timing behind the SAME
    ``make_batch``/``build_trial`` interface. Once the env spike resolves, the NeuroGym
    subclass swaps in here with no change to any caller.

CORRESPONDENCE TO NEUROGYM ``ReadySetGo`` (what we preserve vs. extend)
    The period skeleton is mirrored 1:1 — ``fixation -> ready -> measure -> set ->
    production`` — with matched timing: the pre-Ready dead time is neurogym's 100 ms
    ``fixation`` (``cfg.ready_onset``); ``cfg.pulse_width`` (83 ms) is its
    ``ready``/``set`` period duration; and the Ready->Set *onset* gap is its
    ``measure`` (= ts) — exactly how neurogym stacks ``measure`` from Ready onset.
    Deliberate two-prior extensions (AGENTS.md "extends or wraps"): the 3 obs
    channels are re-mapped from ``{fixation, ready, set}`` to ``{Ready/Set pulse,
    prior-context, effector-context}``; ``measure`` is a discrete two-prior ``ts``
    (passed in via ``make_batch`` with Bayesian jitter), not ``U(800, 1500)``; and
    the target is a continuous ramp-to-threshold, not neurogym's single go-impulse.
    ``tp`` is timed from Set *onset* (ramp crosses at ``set_step + ts``); neurogym
    places its go-impulse ts after Set *end* — a constant one-pulse offset, kept at
    onset here to match the RSG paper and ``src/behavior/slope.py``.

INTERFACE (what the trainer and store rely on)
    make_batch(cfg, batch, rng) -> Batch(inputs[B,T,3], target[B,T], mask[B,T], conditions)
    build_trial(cfg, condition, jitter=False) -> (inputs[1,T,3], set_step:int)

Numpy at the boundary — no torch import here; the trainer moves arrays to torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.conditions import CONDITIONS, N_CONDITIONS, Condition
from src.training.config import Config


@dataclass
class Batch:
    """One training batch. Arrays are numpy; the trainer moves them to torch."""

    inputs: np.ndarray          # [batch, time, 3]
    target: np.ndarray          # [batch, time]
    mask: np.ndarray            # [batch, time]
    conditions: List[Condition]


def ramp(t_rel_steps, ts_steps: int, cfg: Config) -> np.ndarray:
    """Target ramp value(s) at ``t_rel_steps`` steps after Set.

    Two segments:

    * **Approach** (``0 <= t_rel_steps <= ts_steps``): monotone power-law rise,
      **crossing ``cfg.threshold`` at exactly ``t_rel_steps == ts_steps``**. This
      calibration is load-bearing and unchanged from before — it fixes the
      tp-vs-ts baseline slope at 1, so the measured Bayesian bias is not
      confounded by ramp geometry.
    * **Hold** (``t_rel_steps > ts_steps``, up to ``cfg.prod_hold_step`` further):
      rises further, linearly, from ``cfg.threshold`` up to ``cfg.ramp_A`` by the
      end of the hold window, then stays at ``cfg.ramp_A``.

    Reconciles ``docs/RUNBOOK.md`` Gap #2: previously the hold plateaued at
    exactly ``cfg.threshold``, so whether a well-trained network's output
    technically "crossed" threshold was decided by noise at the margin
    (confirmed directly: a converged BPTT run's peak output landed at
    0.993-0.996 for 15/20 conditions — a near-perfect match to the old target,
    just under threshold). Holding at ``cfg.ramp_A`` instead gives real margin
    above threshold, without touching the approach segment's crossing-at-ts
    calibration.

    ``cfg.ramp_A`` default is ``1.2`` (1.2x threshold) — **not** the
    reconstruction's literal quoted ramp amplitude (~3.0/2.85, itself one of the
    "UNVALIDATED reconstruction constants" per AGENTS.md). That literal value was
    tried first (2026-07-21 night) and was too big a jump in target dynamic range
    for the current training budget: BPTT, which otherwise tracks this task well,
    failed to converge at all under it (n_iter=3000, loss 1.28->0.69 instead of
    ->0.004). 1.2x matches the margin separately described in team notes as
    already tested and working.

    This is one defensible reconciliation of ``cfg.ramp_A`` (linear hold rise),
    not a verified match to the original paper's Eq. 9 — flag it as such if a
    closer reading of the reconstruction surfaces a different intended shape.
    """
    t_rel_steps = np.asarray(t_rel_steps, dtype=float)
    approach_frac = np.clip(t_rel_steps / float(ts_steps), 0.0, 1.0)
    approach = cfg.threshold * approach_frac ** cfg.ramp_a

    hold_frac = np.clip((t_rel_steps - ts_steps) / max(cfg.prod_hold_step, 1), 0.0, 1.0)
    hold = cfg.threshold + (cfg.ramp_A - cfg.threshold) * hold_frac

    value = np.where(t_rel_steps <= ts_steps, approach, hold)
    return value.astype(np.float32)


def _measurement_steps(cfg: Config, ts: int, jitter: bool, rng) -> int:
    """Ready->Set gap in steps. Jitter draws ``t_m ~ N(ts, ts*cfg.w_m)`` (the noisy
    measurement); no-jitter uses the true ``ts``. Clipped so the whole production epoch
    still fits the fixed ``cfg.n_steps`` canvas (PLAN_TRACK1.md Blocker B expects the
    generator to clip the rare far-tail draw)."""
    t_m = rng.normal(ts, ts * cfg.w_m) if jitter else float(ts)
    step = int(round(t_m / cfg.dt))
    ts_steps = int(round(ts / cfg.dt))
    max_step = cfg.n_steps - cfg.ready_onset_step - ts_steps - cfg.prod_hold_step - 1
    return int(np.clip(step, 1, max(1, max_step)))


def _build(cfg: Config, condition: Condition, jitter: bool, rng=None):
    """Render one trial onto the fixed canvas. Returns (inputs[T,3], target[T], mask[T],
    set_step). Ready is fixed at ``ready_onset_step``; Set is at ``+t_m`` (jittered);
    the target ramp is always timed to the TRUE ts from Set."""
    T = cfg.n_steps
    pw = cfg.pulse_width_step
    r0 = cfg.ready_onset_step
    ts_steps = int(round(condition.ts / cfg.dt))

    inputs = np.zeros((T, 3), dtype=np.float32)
    inputs[:, 1] = cfg.prior_context[condition.prior]        # tonic prior context
    inputs[:, 2] = cfg.effector_context[condition.effector]  # tonic effector context

    inputs[r0:r0 + pw, 0] = cfg.pulse_height                 # Ready pulse (fixed onset)
    m_steps = _measurement_steps(cfg, condition.ts, jitter, rng)
    set_step = r0 + m_steps
    inputs[set_step:set_step + pw, 0] = cfg.pulse_height     # Set pulse (jittered gap)

    target = np.zeros(T, dtype=np.float32)
    mask = np.zeros(T, dtype=np.float32)
    prod_end = min(set_step + ts_steps + cfg.prod_hold_step, T)
    rel = np.arange(prod_end - set_step)
    target[set_step:prod_end] = ramp(rel, ts_steps, cfg)     # ramp-to-threshold + hold
    mask[set_step:prod_end] = 1.0                            # supervise the production epoch
    return inputs, target, mask, set_step


def make_batch(cfg: Config, batch: int, rng: np.random.Generator) -> Batch:
    """Sample a training batch (jittered Ready-Set gap, target timed to the true ts).

    ``make_batch`` is the SOLE condition sampler: it draws conditions from ``CONDITIONS``
    via ``rng`` and renders each with jitter. Deterministic given ``rng``.
    """
    idxs = rng.integers(0, N_CONDITIONS, size=batch)
    conditions = [CONDITIONS[int(k)] for k in idxs]

    inputs = np.empty((batch, cfg.n_steps, 3), dtype=np.float32)
    target = np.empty((batch, cfg.n_steps), dtype=np.float32)
    mask = np.empty((batch, cfg.n_steps), dtype=np.float32)
    for i, cond in enumerate(conditions):
        inputs[i], target[i], mask[i], _ = _build(cfg, cond, jitter=True, rng=rng)
    return Batch(inputs=inputs, target=target, mask=mask, conditions=conditions)


def build_trial(cfg: Config, condition: Condition, jitter: bool = False):
    """Build a single-condition input (no jitter by default) for eval/storage.

    Returns ``(inputs [1, time, 3], set_step int)``. ``jitter=False`` places Set at the
    true ``ts`` after Ready (the deterministic eval/store path the trainer uses with noise
    off); ``jitter=True`` uses a fresh, non-reproducible RNG (the reproducible jittered
    path is ``make_batch``, which threads the caller's ``rng``).
    """
    rng = np.random.default_rng() if jitter else None
    inputs, _, _, set_step = _build(cfg, condition, jitter=jitter, rng=rng)
    return inputs[None, ...], int(set_step)
