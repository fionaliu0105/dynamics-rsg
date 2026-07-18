"""Continuous-time leaky-tanh RNN, trained by BPTT.  [MEMBER TRACK: BPTT — plan 1.B]

This is the SHARED forward model — the PC net (1.C) reuses this exact ``forward``,
so that the only difference between the two arms is the learning rule (the whole
point of the study). Get the dynamics right here and PC inherits them.

DYNAMICS (Sohn et al. 2019, Eqs. 5-7; continuous-time, Euler-discretized)::

    x_{t+1} = x_t + alpha * (-x_t + J @ r_t + B @ u_t + c_x) + noise
    r_t     = tanh(x_t)                              # the returned `states`
    z_t     = w_o @ r_t + c_z                        # the returned `outputs`

    alpha = dt / tau ;  J ~ N(0, g^2 / N) ;  B, c_x, x0 ~ U[-1, 1] ;  w_o, c_z = 0
    noise = sqrt(2 * alpha) * noise_sd * N(0, I)     # disable for deterministic checks

READOUT: effector-gated. Two output channels (one per effector); the effector-
context input selects which channel the loss scores (plan decision 7). Keep BOTH
channels in `outputs` if you can, or gate to the active one — document your choice.

TRAINING: masked MSE between ``z`` and the ramp target over the production epoch
(``Batch.mask``). Adam over BPTT (the paper used Hessian-free; we substitute Adam).

DEFINITION OF DONE (plan 1.B check)
    Trains on trivial data, loss drops, tp finite and ordered, and the correct
    effector channel is the one that crosses threshold.

Requires torch (not importable in the contracts-only smoke env — that's expected).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import Model
from src.training.config import Config


class BPTTRNN(nn.Module, Model):
    """Shared continuous-time RNN; BPTT arm. See module docstring for the equations."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # TODO(bptt-track): init parameters J, B, c_x, x0, and the effector-gated
        # readout (w_o, c_z) per the docstring. Seed via cfg.seed for reproducibility.
        raise NotImplementedError("BPTT track: build parameters (plan 1.B)")

    def forward(self, inputs, *, noise: bool = True, return_states: bool = True):
        """Roll the leaky-tanh dynamics. Returns (outputs [B,T], states [B,T,N]).

        TODO(bptt-track): implement the Euler update loop from the docstring.
        This method IS the shared forward the PC net will reuse.
        """
        raise NotImplementedError("BPTT track: implement forward (plan 1.B)")
