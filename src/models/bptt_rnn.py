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

READOUT: effector-gated. Two channels, ordered ``EFFECTOR_ORDER`` (== the single
source of truth ``src.conditions.EFFECTORS``). Both channels are computed every
step and kept on ``model._last_outputs_both`` ``[B, T, 2]`` for inspection/tests;
``outputs`` gates to the channel matching each trial's tonic effector-context input
(``inputs[:, :, 2]``), since the loss only scores the active effector's channel.

TRAINING: masked MSE between ``z`` and the ramp target over the production epoch
(``Batch.mask``). Adam over BPTT (the paper used Hessian-free; we substitute Adam).

TRAINING STABILITY (investigated, not a bug — read before "fixing" a loss spike)
    Training shows periodic loss spikes under vanilla Adam + full BPTT. Root cause,
    confirmed analytically: with ``g=1`` the linearized dynamics around x=0 have
    max eigenvalue magnitude of ``(1-alpha) + alpha*J`` equal to ~1.0 REGARDLESS of
    alpha (``= dt/tau``) — g=1 puts this network at marginal stability / the edge
    of chaos by construction (Sompolinsky et al. 1988; the paper's deliberate
    choice), not a code defect. The ``reduced`` regime's coarse Euler step
    (dt=5, alpha=0.5) amplifies this into visible optimization roughness; the same
    setup at dt=1 (closer to ``faithful``) converges ~170x lower median loss for
    ~5x the compute — ``reduced`` is documented (AGENTS.md) as under-training
    on purpose, so don't read its roughness as evidence of a bug either.
    Practical consequences for anyone training this model:
      * ``cfg.grad_clip``'s default (1.0) global-norms across ~26k params
        dominated by ``J`` (N x N), starving the ~2N-param readout (``w_o``,
        ``c_z``) of gradient signal. Tune it up (5.0 worked for the reduced-regime
        smoke checks) rather than assuming the readout is broken.
      * Evaluate/report from the BEST-loss checkpoint seen during training, not
        the final iterate — a late unlucky step can transiently spike loss even
        after a good solution was found. This is what periodic checkpointing
        (trainer.py, plan 2.1) gives you for free; it isn't specific to this model.

DEFINITION OF DONE (plan 1.B check)
    Trains on trivial data, loss drops, tp finite and ordered, and the correct
    effector channel is the one that crosses threshold.

Requires torch (not importable in the contracts-only smoke env — that's expected).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.conditions import EFFECTORS as EFFECTOR_ORDER
from src.models.base import Model
from src.training.config import Config

__all__ = ["EFFECTOR_ORDER", "BPTTRNN"]


class BPTTRNN(nn.Module, Model):
    """Shared continuous-time RNN; BPTT arm. See module docstring for the equations."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        n, n_in = cfg.N, self.N_IN
        n_eff = len(EFFECTOR_ORDER)

        gen = torch.Generator().manual_seed(cfg.seed)
        self.J = nn.Parameter(torch.randn(n, n, generator=gen) * (cfg.g / math.sqrt(n)))
        self.B = nn.Parameter(torch.rand(n, n_in, generator=gen) * 2 - 1)
        self.c_x = nn.Parameter(torch.rand(n, generator=gen) * 2 - 1)
        self.x0 = nn.Parameter(torch.rand(n, generator=gen) * 2 - 1)
        # effector-gated readout: one row per channel, EFFECTOR_ORDER order; init 0
        self.w_o = nn.Parameter(torch.zeros(n_eff, n))
        self.c_z = nn.Parameter(torch.zeros(n_eff))

        effector_values = [cfg.effector_context[e] for e in EFFECTOR_ORDER]
        self.register_buffer("_effector_values", torch.tensor(effector_values, dtype=torch.float32))

        self._last_outputs_both: torch.Tensor | None = None

    def forward(self, inputs, *, noise: bool = True, return_states: bool = True):
        """Roll the leaky-tanh dynamics. Returns (outputs [B,T], states [B,T,N])."""
        batch, T, _ = inputs.shape
        device, dtype = inputs.device, inputs.dtype

        x = self.x0.to(dtype).unsqueeze(0).expand(batch, -1)
        alpha = self.cfg.alpha
        noise_scale = math.sqrt(2 * alpha) * self.cfg.noise_sd

        states_list = []
        both_list = []
        for t in range(T):
            r = torch.tanh(x)
            states_list.append(r)
            both_list.append(r @ self.w_o.t() + self.c_z)

            u = inputs[:, t, :]
            dx = -x + r @ self.J.t() + u @ self.B.t() + self.c_x
            x = x + alpha * dx
            if noise:
                x = x + noise_scale * torch.randn(x.shape, device=device, dtype=dtype)

        states = torch.stack(states_list, dim=1)          # [B, T, N]
        both = torch.stack(both_list, dim=1)               # [B, T, n_eff]
        self._last_outputs_both = both

        ctx = inputs[:, 0, 2].unsqueeze(-1)                # [B, 1] tonic effector context
        effector_values = self._effector_values.to(device=device, dtype=dtype)
        idx = torch.argmin((ctx - effector_values.unsqueeze(0)).abs(), dim=-1)  # [B]
        idx_exp = idx.view(batch, 1, 1).expand(batch, T, 1)
        outputs = both.gather(-1, idx_exp).squeeze(-1)     # [B, T]

        if not return_states:
            return outputs, None
        return outputs, states
