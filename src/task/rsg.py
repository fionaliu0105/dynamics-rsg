"""Two-prior Ready-Set-Go task generator.  [MEMBER TRACK: Task & behavior — plan 1.A]

WHAT TO BUILD
    Extend NeuroGym's ``ReadySetGo-v0`` (single-prior, no context input) into the
    two-prior task the animal actually did. A trial's input has THREE channels
    (``Model.N_IN == 3``):
        0. Ready/Set pulse channel  (two ``pulse_height`` pulses ``ts`` apart)
        1. prior-context channel    (tonic ``cfg.prior_context[prior]``)
        2. effector-context channel (tonic ``cfg.effector_context[effector]``)
    and a target ramp that reaches ``cfg.threshold`` at ``ts`` after Set (Eq. 9).

    Draw conditions from :data:`src.conditions.CONDITIONS`. During TRAINING, jitter
    the Ready-Set separation with scalar noise ``t_m ~ N(ts, ts * cfg.w_m)`` while
    timing the target to the TRUE ``ts`` — that averaging is what bends the learned
    mapping toward the prior mean (the Bayesian bias). See the reconstructed code in
    the planning doc's Details tab for a concrete (UNVALIDATED) starting point.

INTERFACE (what the trainer and store rely on)
    make_batch(cfg, batch, rng) -> Batch with:
        inputs  : [batch, time, 3]
        target  : [batch, time]      the ramp (BPTT loss target)
        mask    : [batch, time]      1.0 inside the production epoch, else 0.0
        conditions : list[Condition] length `batch`
    build_trial(cfg, condition, jitter=False) -> single-condition inputs (+ Set step)
        for evaluation / storing per-condition activations.

DEFINITION OF DONE (plan 1.A check)
    Input carries all three channels; every trial carries ts/prior/effector; the
    dt you get matches cfg.dt; a batch's shapes are [batch, cfg.n_steps, 3].

REFERENCE
    NeuroGym: https://github.com/neurogym/neurogym  ·  Sohn et al. 2019 STAR Methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.conditions import CONDITIONS, Condition
from src.training.config import Config


@dataclass
class Batch:
    """One training batch. Arrays are numpy; the trainer moves them to torch."""

    inputs: np.ndarray          # [batch, time, 3]
    target: np.ndarray          # [batch, time]
    mask: np.ndarray            # [batch, time]
    conditions: List[Condition]


def make_batch(cfg: Config, batch: int, rng: np.random.Generator) -> Batch:
    """Sample a training batch with scalar measurement noise on the Ready-Set gap.

    The Ready/Set input gap is jittered as ``t_m ~ N(ts, ts * cfg.w_m)`` while
    the production target remains timed to the true ``ts`` after the jittered Set.
    Keeping the jitter in the input, not the target, is the reconstruction's
    proposed Bayesian-bias mechanism and is deliberately limited to training.
    """
    choices = rng.integers(0, len(CONDITIONS), size=batch)
    inputs = np.zeros((batch, cfg.n_steps, 3), dtype=np.float32)
    target = np.zeros((batch, cfg.n_steps), dtype=np.float32)
    mask = np.zeros((batch, cfg.n_steps), dtype=np.float32)
    conditions: List[Condition] = []

    for row, idx in enumerate(choices):
        condition = CONDITIONS[int(idx)]
        measured_ts = _jittered_ts_ms(cfg, condition.ts, rng)
        set_step = _fill_trial_arrays(cfg, condition, measured_ts, inputs[row], target[row], mask[row])
        if not 0 <= set_step < cfg.n_steps:
            raise ValueError(f"set_step={set_step} outside trial for {condition}")
        conditions.append(condition)

    return Batch(inputs=inputs, target=target, mask=mask, conditions=conditions)


def build_trial(
    cfg: Config,
    condition: Condition,
    jitter: bool = False,
    rng: np.random.Generator | None = None,
):
    """Build a single-condition input (no jitter by default) for eval/storage.

    Returns ``(inputs [1, time, 3], set_step int)``. Evaluation/storage callers
    should keep the default ``jitter=False`` so activations are on the canonical
    condition time base.
    """
    inputs = np.zeros((1, cfg.n_steps, 3), dtype=np.float32)
    target = np.zeros((1, cfg.n_steps), dtype=np.float32)
    mask = np.zeros((1, cfg.n_steps), dtype=np.float32)
    if jitter:
        if rng is None:
            raise ValueError("jitter=True requires an rng for reproducibility")
        measured_ts = _jittered_ts_ms(cfg, condition.ts, rng)
    else:
        measured_ts = float(condition.ts)
    set_step = _fill_trial_arrays(cfg, condition, measured_ts, inputs[0], target[0], mask[0])
    return inputs, set_step


def trial_target_and_mask(cfg: Config, condition: Condition, set_step: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic target and mask for an already-built trial.

    This is used by tests and diagnostics; training uses the same private helper
    through :func:`make_batch`.
    """
    target = np.zeros((cfg.n_steps,), dtype=np.float32)
    mask = np.zeros((cfg.n_steps,), dtype=np.float32)
    _fill_target_and_mask(cfg, condition.ts, set_step, target, mask)
    return target, mask


def _jittered_ts_ms(cfg: Config, ts_ms: float, rng: np.random.Generator) -> float:
    measured = float(rng.normal(ts_ms, ts_ms * cfg.w_m))
    min_gap = max(cfg.dt, cfg.pulse_width)
    max_gap = cfg.total_time - cfg.ready_onset - cfg.prod_hold - ts_ms
    return float(np.clip(measured, min_gap, max(min_gap, max_gap)))


def _fill_trial_arrays(
    cfg: Config,
    condition: Condition,
    measured_ts_ms: float,
    inputs: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> int:
    ready_step = cfg.ready_onset_step
    set_step = ready_step + cfg.to_step(measured_ts_ms)
    _fill_pulse(inputs, ready_step, cfg.pulse_width_step, cfg.pulse_height)
    _fill_pulse(inputs, set_step, cfg.pulse_width_step, cfg.pulse_height)
    inputs[:, 1] = cfg.prior_context[condition.prior]
    inputs[:, 2] = cfg.effector_context[condition.effector]
    _fill_target_and_mask(cfg, condition.ts, set_step, target, mask)
    return set_step


def _fill_pulse(inputs: np.ndarray, start: int, width: int, height: float) -> None:
    stop = min(start + width, inputs.shape[0])
    if start < inputs.shape[0] and stop > start:
        inputs[start:stop, 0] = height


def _fill_target_and_mask(
    cfg: Config,
    true_ts_ms: float,
    set_step: int,
    target: np.ndarray,
    mask: np.ndarray,
) -> None:
    go_step = set_step + cfg.to_step(true_ts_ms)
    hold_end = min(go_step + cfg.prod_hold_step, cfg.n_steps)
    if set_step >= cfg.n_steps or go_step <= set_step:
        return

    ramp_end = min(go_step, cfg.n_steps)
    ramp_steps = max(ramp_end - set_step, 1)
    elapsed_ms = np.arange(ramp_steps, dtype=np.float32) * float(cfg.dt)
    x = np.clip(elapsed_ms / float(true_ts_ms), 0.0, 1.0)
    shaped = np.power(x, float(cfg.ramp_a))
    if abs(cfg.ramp_A) > 1e-12:
        shaped = (1.0 - np.exp(-float(cfg.ramp_A) * shaped)) / (1.0 - np.exp(-float(cfg.ramp_A)))
    target[set_step:ramp_end] = cfg.threshold * shaped.astype(np.float32)
    if go_step < cfg.n_steps:
        target[go_step:hold_end] = cfg.threshold
    mask[set_step:hold_end] = 1.0
