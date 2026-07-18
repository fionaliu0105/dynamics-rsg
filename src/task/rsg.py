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

from src.conditions import Condition
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

    TODO(task-track): implement per the module docstring. Keep it deterministic
    given ``rng`` so runs are reproducible.
    """
    raise NotImplementedError("Task & behavior track: implement make_batch (plan 1.A)")


def build_trial(cfg: Config, condition: Condition, jitter: bool = False):
    """Build a single-condition input (no jitter by default) for eval/storage.

    Returns ``(inputs [1, time, 3], set_step int)``. TODO(task-track).
    """
    raise NotImplementedError("Task & behavior track: implement build_trial (plan 1.A)")
