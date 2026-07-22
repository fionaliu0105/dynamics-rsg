"""Predictive-coding learning for the shared continuous-time RNN.

This module implements PC-A only: deterministic predictive-coding inference and
local parameter updates for the same leaky-tanh architecture used by the BPTT
arm.  Architecture parity is intentional and load-bearing for the project: the
forward dynamics, parameter shapes, initialization scheme, effector-gated readout,
and public tensor contracts match the BPTT model, while the learning rule differs.

``PCRNN.forward()`` rolls the shared RNN equations directly in this module rather
than delegating through ``BPTTRNN``.  It returns effector-gated outputs with shape
``[batch, time]`` and, when requested, rate-like states with shape
``[batch, time, units]``.  Predictive-coding-specific work lives in
``infer_and_update()``: raw pre-activation values are initialized by a
deterministic forward sweep, relaxed against temporal Euler prediction errors and
masked output errors, and then used to compute local updates for the model
parameters.

Expected tensor contracts are:

* ``inputs``: ``[batch, time, 3]``; the third channel is the tonic effector
  context used to select the readout row.
* ``outputs``: ``[batch, time]`` after effector gating.
* rate-like ``states``: ``[batch, time, units]`` from ``tanh(raw_values)``.
* raw inferred ``values``: ``[batch, time, units]`` in the diagnostics returned
  by ``infer_and_update()``.
* ``target`` and ``mask``: ``[batch, time]`` for the scored output trajectory.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from src.conditions import EFFECTORS as EFFECTOR_ORDER
from src.models.base import Model
from src.training.config import Config


class PCRNN(nn.Module, Model):
    """PC learning rule over the same continuous-time RNN architecture as BPTT.

    The module owns the RNN parameter tensors directly:

    * ``J`` recurrent weights, ``[units, units]``.
    * ``B`` input weights, ``[units, 3]``.
    * ``c_x`` recurrent bias, ``[units]``.
    * ``x0`` initial raw state, ``[units]``.
    * ``w_o`` effector-specific readout weights, ``[n_effectors, units]``.
    * ``c_z`` effector-specific readout biases, ``[n_effectors]``.

    PC-A uses those tensors in two modes.  ``forward()`` is the normal shared
    RNN rollout and exposes rate-like states.  ``infer_and_update()`` treats raw
    pre-activation trajectories as value nodes, relaxes those values for
    ``cfg.pc_inference_steps``, and computes local parameter updates from the
    resulting prediction errors.
    """

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
        self.w_o = nn.Parameter(torch.zeros(n_eff, n))
        self.c_z = nn.Parameter(torch.zeros(n_eff))

        effector_values = [cfg.effector_context[e] for e in EFFECTOR_ORDER]
        self.register_buffer("_effector_values", torch.tensor(effector_values, dtype=torch.float32))

        self._last_outputs_both: torch.Tensor | None = None
        self.last_energy_trace: list[float] = []
        self.last_update_finite: Dict[str, bool] = {}

    @property
    def dynamics(self):
        """Compatibility alias for parity checks; returns this direct RNN module.

        Earlier PC-A validation code loaded ``pc.dynamics.state_dict()`` into a
        separate BPTT model to compare forward rollouts.  ``PCRNN`` no longer
        contains or imports ``BPTTRNN``, but the direct implementation owns the
        same parameter and buffer names, so returning ``self`` preserves that
        read-only diagnostic pattern without introducing a second architecture.
        """
        return self

    def forward(self, inputs, *, noise: bool = True, return_states: bool = True):
        """Roll the architecture-parity RNN and return gated outputs and states.

        Parameters
        ----------
        inputs:
            Tensor with shape ``[batch, time, 3]``.  The first two channels are
            task inputs; the third channel is a tonic effector context.  The
            effector context in ``inputs[:, 0, 2]`` selects which readout row is
            returned for each trial.
        noise:
            If true, add private process noise after each Euler state update.
            Deterministic validation calls should pass ``noise=False``.
        return_states:
            If true, return rate-like hidden states ``tanh(x_t)`` with shape
            ``[batch, time, units]``.  If false, return ``None`` for states.

        Returns
        -------
        tuple
            ``(outputs, states)`` where ``outputs`` has shape ``[batch, time]``.
            ``states`` has shape ``[batch, time, units]`` when requested.  The
            ungated readout for every effector is also stored on
            ``self._last_outputs_both`` with shape ``[batch, time, n_effectors]``
            for diagnostics and parity checks.
        """
        batch, time, _ = inputs.shape
        device, dtype = inputs.device, inputs.dtype

        x = self.x0.to(dtype).unsqueeze(0).expand(batch, -1)
        alpha = self.cfg.alpha
        noise_scale = math.sqrt(2 * alpha) * self.cfg.noise_sd

        states_list = []
        both_list = []
        for t in range(time):
            r = torch.tanh(x)
            states_list.append(r)
            both_list.append(r @ self.w_o.t() + self.c_z)

            u = inputs[:, t, :]
            dx = -x + r @ self.J.t() + u @ self.B.t() + self.c_x
            x = x + alpha * dx
            if noise:
                x = x + noise_scale * torch.randn(x.shape, device=device, dtype=dtype)

        states = torch.stack(states_list, dim=1)
        both = torch.stack(both_list, dim=1)
        self._last_outputs_both = both

        idx = self._effector_index(inputs)
        idx_exp = idx.view(batch, 1, 1).expand(batch, time, 1)
        outputs = both.gather(-1, idx_exp).squeeze(-1)

        if not return_states:
            return outputs, None
        return outputs, states

    def _raw_forward_values(self, inputs: torch.Tensor) -> torch.Tensor:
        """Initialize raw PC value nodes by a deterministic Euler forward sweep.

        The returned tensor has shape ``[batch, time, units]`` and contains raw
        pre-activation values ``x_t``, not rates.  Parameters are detached because
        PC-A computes explicit local updates rather than using autograd for
        ``infer_and_update()``.  Process noise is deliberately omitted so the
        initialization is repeatable for a fixed input and parameter state.
        """
        batch, time, _ = inputs.shape
        x = self.x0.detach().to(dtype=inputs.dtype).unsqueeze(0).expand(batch, -1).clone()
        values = []
        alpha = self.cfg.alpha
        for t in range(time):
            values.append(x)
            r = torch.tanh(x)
            x = x + alpha * (-x + r @ self.J.detach().t() + inputs[:, t] @ self.B.detach().t() + self.c_x.detach())
        return torch.stack(values, dim=1)

    def _effector_index(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the readout row selected by each trial's tonic effector input."""
        values = self._effector_values.to(device=inputs.device, dtype=inputs.dtype)
        return torch.argmin((inputs[:, 0, 2:3] - values.unsqueeze(0)).abs(), dim=-1)

    def _relax(self, values, inputs, target, mask):
        """Relax value nodes by gradient descent on the PC energy, all timesteps at once.

        Millidge's ``PC_RNN.infer`` visits timesteps in reverse order (Gauss-Seidel) so
        that readout error propagates backward across the sequence.  That ordering is
        **not** needed here, and we deliberately do not copy it: his char-prediction
        setup scores only the sequence end, whereas RSG supervises the whole production
        epoch, so every value node already receives direct output-error pressure and
        there is no long-range credit to assign.  Measured on the reduced regime the two
        schemes are equivalent in both update direction (cosine to the BPTT gradient
        0.71 vs 0.72) and outcome after 100 iterations (``|dJ|`` 2.90 vs 2.83, loss 0.067
        vs 0.066), while the reverse sweep costs 4.7x more wall-clock because it is a
        Python loop over ``T`` steps.  Keep the vectorized form.

        Backtracking halves the step until energy is non-increasing, which is what makes
        the returned trace a usable PC-A energy-descent diagnostic.  ``values[:, 0]`` is
        held at the ``x0`` parameter -- it is the model's initial-value node, not a free
        latent.

        Returns ``(values, energy, temporal_error, output_error, r, output, trace)``.
        """
        energy, temporal_error, output_error, r, output = self._energy_and_errors(
            values, inputs, target, mask
        )
        trace = [float(energy.item())]
        for _ in range(self.cfg.pc_inference_steps):
            grad = self._value_gradient(values, inputs, temporal_error, output_error, mask)
            step = self.cfg.pc_inference_lr
            while True:
                candidate = values - step * grad
                candidate[:, 0] = self.x0.detach().to(candidate.dtype)
                cand = self._energy_and_errors(candidate, inputs, target, mask)
                candidate_energy = cand[0]
                if candidate_energy <= energy + 1e-7 or step < 1e-8:
                    values, energy = candidate, candidate_energy
                    _, temporal_error, output_error, r, output = cand
                    break
                step *= 0.5
            trace.append(float(energy.item()))
        return values, energy, temporal_error, output_error, r, output, trace

    def _energy_and_errors(self, values, inputs, target, mask):
        """Return PC energy and prediction errors for the current value nodes.

        The temporal error compares each inferred raw value ``x_{t+1}`` against
        the one-step Euler prediction from ``x_t``:

        ``x_t + alpha * (-x_t + J tanh(x_t) + B u_t + c_x)``.

        The output error compares the effector-gated readout to ``target`` and is
        multiplied by ``mask`` so only scored time points contribute.  The scalar
        energy is half the sum of squared temporal errors plus half the sum of
        squared masked output errors.
        """
        alpha = self.cfg.alpha
        r = torch.tanh(values)
        pred_next = values[:, :-1] + alpha * (
            -values[:, :-1]
            + r[:, :-1] @ self.J.detach().t()
            + inputs[:, :-1] @ self.B.detach().t()
            + self.c_x.detach()
        )
        temporal_error = values[:, 1:] - pred_next
        both = r @ self.w_o.detach().t() + self.c_z.detach()
        idx = self._effector_index(inputs).view(-1, 1, 1).expand(-1, values.shape[1], 1)
        output = both.gather(-1, idx).squeeze(-1)
        output_error = (output - target) * mask
        energy = 0.5 * (temporal_error.square().sum() + output_error.square().sum())
        return energy, temporal_error, output_error, r, output

    def _value_gradient(self, values, inputs, temporal_error, output_error, mask):
        """Compute the local gradient used to relax raw PC value nodes.

        Each value receives direct temporal-error pressure from its own mismatch,
        output-error pressure through the selected readout row and ``tanh``
        derivative, and propagated pressure because changing ``x_t`` changes the
        Euler prediction for ``x_{t+1}``.
        """
        alpha = self.cfg.alpha
        derivative = 1.0 - torch.tanh(values).square()
        grad = torch.zeros_like(values)
        grad[:, 1:] += temporal_error

        idx = self._effector_index(inputs)
        selected_w = self.w_o.detach()[idx]  # [B, N]
        grad += (output_error.unsqueeze(-1) * selected_w.unsqueeze(1)) * derivative

        # Each value also changes the one-step prediction that follows it.
        propagated = (1.0 - alpha) * temporal_error + alpha * (temporal_error @ self.J.detach()) * derivative[:, :-1]
        grad[:, :-1] -= propagated
        return grad

    def _local_updates(self, values, inputs, temporal_error, output_error, r):
        """Compute local parameter gradients with inferred values held fixed.

        Updates are returned for all PC-A parameter groups: ``J``, ``B``,
        ``c_x``, ``x0``, ``w_o``, and ``c_z``.  Recurrent/input/bias updates use
        temporal Euler errors and the previous-step rates/inputs.  Readout
        updates use masked, effector-gated output errors and only accumulate into
        the readout row selected for each trial.
        """
        alpha = self.cfg.alpha
        eps = temporal_error
        r_prev = r[:, :-1]
        u_prev = inputs[:, :-1]
        updates = {
            "J": -alpha * torch.einsum("bti,btj->ij", eps, r_prev),
            "B": -alpha * torch.einsum("bti,btk->ik", eps, u_prev),
            "c_x": -alpha * eps.sum(dim=(0, 1)),
            "x0": self._value_gradient(values, inputs, temporal_error, output_error, torch.ones_like(output_error))[:, 0].sum(dim=0),
            "w_o": torch.zeros_like(self.w_o),
            "c_z": torch.zeros_like(self.c_z),
        }
        idx = self._effector_index(inputs)
        for group in range(self.w_o.shape[0]):
            chosen = idx == group
            if chosen.any():
                err = output_error[chosen]
                rates = r[chosen]
                updates["w_o"][group] = torch.einsum("bt,bti->i", err, rates)
                updates["c_z"][group] = err.sum()
        return updates

    def _rescale_updates(self, updates, mask):
        """Put PC updates on the same footing as the BPTT arm's gradients.

        Two corrections, both matching what ``training/trainer.py`` already does for
        BPTT and neither of which the raw local rule supplies:

        1. **Normalize.** ``_local_updates`` accumulates raw *sums* over batch and
           time, while the BPTT loss is a masked *mean* (``masked_mse`` divides by
           ``mask.sum()``).  Left unnormalized the two arms differ by ~1e4 at the same
           ``cfg.lr``, which is what drove the old divergence at iteration ~9.
           Dividing the whole energy by ``mask.sum()`` is a scalar rescale of the
           objective, so it changes step size without touching update direction.
        2. **Clip.** ``cfg.grad_clip`` is applied to the BPTT arm but was never applied
           here.  Millidge clamps elementwise (``clamp_val=50``); we use a global-norm
           clip so the two arms share one clipping semantics.

        Both keep architecture *and* optimizer handling parity, so a PC-vs-BPTT
        difference remains attributable to the learning rule.
        """
        scale = float(mask.sum().clamp_min(1.0).item())
        updates = {name: value / scale for name, value in updates.items()}

        clip = float(self.cfg.grad_clip)
        if clip > 0:
            total = torch.sqrt(sum(value.square().sum() for value in updates.values()))
            if bool(torch.isfinite(total)) and float(total) > clip:
                updates = {name: value * (clip / (total + 1e-12)) for name, value in updates.items()}
        return updates

    @staticmethod
    def _metrics(pc_update: torch.Tensor, bptt_grad: torch.Tensor) -> Dict[str, float]:
        """Summarize one PC update against the matching autograd gradient.

        ``updates`` are *gradients*, not descent directions: ``infer_and_update``
        applies them as ``parameter.add_(-lr * update)``.  The reference is therefore
        ``+bptt_grad``, so a correct port scores ``cosine -> +1``.  This previously
        negated the reference, which reported an exactly-correct readout update as
        ``cosine = -1.000`` and made the PC-A alignment observable read backwards.
        """
        reference = bptt_grad
        denom = pc_update.norm() * reference.norm()
        cosine = float((pc_update.flatten() @ reference.flatten() / denom).item()) if denom > 0 else float("nan")
        relative_error = float((pc_update - reference).norm().div(reference.norm().clamp_min(1e-12)).item())
        return {"cosine": cosine, "relative_error": relative_error}

    def infer_and_update(self, inputs, target, mask, *, apply_update: bool = True):
        """Infer raw values and optionally apply one local PC-A parameter update.

        ``inputs`` must have shape ``[batch, time, 3]``.  ``target`` and ``mask``
        must both have shape ``[batch, time]``.  The computation proceeds as:

        1. Initialize raw inferred values ``[batch, time, units]`` by a
           deterministic forward sweep with noise disabled.
        2. Compute temporal Euler prediction errors between adjacent raw values.
        3. Compute masked output errors after selecting the active effector
           readout row from the tonic context input.
        4. Define energy as ``0.5 * (sum(temporal_error**2) +
           sum(masked_output_error**2))``.
        5. Relax values for ``cfg.pc_inference_steps`` using the local value
           gradient (see ``_relax``).  Backtracking halves the inference step size
           until energy is non-increasing, preserving a deterministic energy-descent
           diagnostic.
        6. Compute local updates for ``J``, ``B``, ``c_x``, ``x0``, ``w_o``, and
           ``c_z``, rescale them onto the BPTT arm's footing (``_rescale_updates``),
           and, if ``apply_update`` is true, subtract ``cfg.lr * update`` from each
           parameter.

        Note that ``updates`` are *gradients*: they are applied as
        ``parameter.add_(-lr * update)``, and ``_metrics`` scores them against
        ``+bptt_grad``.

        Returns a diagnostic dictionary with:

        * ``energy_trace``: scalar energy before inference and after each
          relaxation sweep.
        * ``loss``: masked output MSE-style loss on the final inferred values.
        * ``updates``: detached update tensors keyed by parameter name.
        * ``values``: detached final raw inferred values ``[batch, time, units]``.
        * ``outputs``: detached final effector-gated outputs ``[batch, time]``.
        * ``finite``: per-update boolean finite flags.
        """
        if inputs.ndim != 3 or target.shape != inputs.shape[:2] or mask.shape != target.shape:
            raise ValueError("expected inputs [B,T,3], target and mask [B,T]")
        if self.cfg.pc_inference_steps < 1:
            raise ValueError("pc_inference_steps must be at least one")

        with torch.no_grad():
            values = self._raw_forward_values(inputs)
            values, energy, temporal_error, output_error, r, output, trace = self._relax(
                values, inputs, target, mask
            )

            updates = self._local_updates(values, inputs, temporal_error, output_error, r)
            updates = self._rescale_updates(updates, mask)
            parameters = {"J": self.J, "B": self.B, "c_x": self.c_x, "x0": self.x0, "w_o": self.w_o, "c_z": self.c_z}
            finite = {name: bool(torch.isfinite(value).all()) for name, value in updates.items()}
            if not all(finite.values()) or not torch.isfinite(values).all() or not torch.isfinite(energy):
                raise FloatingPointError("non-finite PC inference value, energy, or update")
            if apply_update:
                for name, parameter in parameters.items():
                    parameter.add_(-self.cfg.lr * updates[name])
                if not all(torch.isfinite(parameter).all() for parameter in parameters.values()):
                    raise FloatingPointError("non-finite parameter after PC update")

        self.last_energy_trace = trace
        self.last_update_finite = finite
        return {
            "energy_trace": trace,
            "loss": float((0.5 * (output - target).square() * mask).sum().item() / mask.sum().clamp_min(1).item()),
            "updates": {name: value.detach().clone() for name, value in updates.items()},
            "values": values.detach().clone(),
            "outputs": output.detach().clone(),
            "finite": finite,
        }

    def bptt_update_alignment(self, inputs, target, mask) -> Dict[str, Dict[str, float]]:
        """Compare PC local updates with autograd/BPTT descent directions.

        This method is diagnostic and non-gating: it reports whether the local
        PC-A update vectors align with gradients from a deterministic
        ``noise=False`` forward pass through the same RNN equations, but it does
        not decide whether a seed, model, or implementation passes.  The returned
        dictionary is keyed by parameter group:

        ``{"J": {"cosine": float, "relative_error": float}, ...}``

        and includes entries for ``J``, ``B``, ``c_x``, ``x0``, ``w_o``, and
        ``c_z``.  ``cosine`` compares the PC update to the negative autograd
        gradient; ``relative_error`` is the normed difference from that descent
        direction.
        """
        result = self.infer_and_update(inputs, target, mask, apply_update=False)
        self.zero_grad(set_to_none=True)
        outputs, _ = self.forward(inputs, noise=False)
        loss = 0.5 * (((outputs - target).square()) * mask).sum()
        loss.backward()
        gradients = {"J": self.J.grad, "B": self.B.grad, "c_x": self.c_x.grad, "x0": self.x0.grad, "w_o": self.w_o.grad, "c_z": self.c_z.grad}
        return {name: self._metrics(result["updates"][name], gradient) for name, gradient in gradients.items()}
