"""RFLO learning for the shared continuous-time RNN.  [third arm: local in time AND space]

Random Feedback Local Online learning (Murray 2019, *Local online learning in recurrent
networks with random feedback*, eLife 8:e43299), adapted to this project's dynamics.

This is the third learning rule on the SAME architecture as BPTT and PC.  ``RFLORNN``
subclasses :class:`~src.models.bptt_rnn.BPTTRNN` rather than re-implementing the rollout,
so architecture parity is structural instead of hand-maintained: ``__init__``,
``forward()``, the parameter shapes, and the effector-gated readout are literally the
BPTT arm's.  Only :meth:`infer_and_update` differs.

WHERE IT SITS ON THE LOCALITY AXIS
    BPTT is nonlocal in time and space.  PC is local in space but relaxes value nodes
    over the whole stored trajectory (offline, iterative).  RFLO is local in BOTH and
    single-pass online: one forward sweep carrying decaying eligibility traces, with a
    top-down error delivered through fixed random feedback.  That makes the three arms a
    graded axis rather than three arbitrary points.

THE RULE
    RFLO is RTRL (the exact forward-mode gradient, ``O(N^3)`` memory) with two stacked
    approximations:

    1. TEMPORAL -- truncate the Jacobian ``dx(t+1)/dx(t)`` to its leak term ``(1-alpha)``,
       dropping the recurrent ``alpha * J diag(1-r^2)`` coupling.  This collapses RTRL's
       rank-3 sensitivity tensor into the decaying traces below.
    2. SPATIAL -- replace ``w_o^T`` in the learning signal with a fixed random feedback
       matrix (feedback alignment), removing weight transport.

    With ``alpha = dt/tau``, traces are advanced AFTER each step's accumulation::

        P_j(t+1) = (1-alpha) P_j(t) + alpha * r_j(t)     [B, N]   presynaptic rate
        Q_k(t+1) = (1-alpha) Q_k(t) + alpha * u_k(t)     [B, 3]   presynaptic input
        S(t+1)   = (1-alpha) S(t)   + alpha              scalar   bias
        D(t+1)   = (1-alpha) D(t),   D(0) = 1            scalar   x0 sensitivity

    and with ``delta_t = mask_t * (z_t - y_t)`` the three-factor update is::

        e_i(t)   = delta_t * feedback[eff, i] * (1 - r_i(t)^2)      [B, N]
        dJ      += einsum("bi,bj->ij", e_t, P_t)
        dB      += einsum("bi,bk->ik", e_t, Q_t)
        dc_x    += e_t.sum(0) * S_t
        dx0     += e_t.sum(0) * D_t
        dw_o[g] += einsum("b,bi->i", delta_t[g], r_t[g])   # exact, per effector group
        dc_z[g] += delta_t[g].sum()                        # exact

NO ``phi'`` IN THE TRACE -- this DIVERGES from Murray's published equation, on purpose.
    Murray's network applies the nonlinearity to the pre-activation sum
    (``h = phi(W h + ...)``), so ``W`` sits inside ``phi`` and the trace carries a
    ``phi'(u_i)`` factor, making it postsynaptically indexed and ``[B, N, N]``-shaped.
    This project uses the Sohn et al. voltage form (``x`` is current, ``r = tanh(x)`` is
    rate, ``J`` multiplies RATES), so ``dx_i(t+1)/dJ_ij = alpha * r_j(t)`` carries no
    ``phi'`` at all -- it lands in ``e_i(t)`` instead, via ``dr_i/dx_i``.  The trace
    therefore loses its postsynaptic index entirely and collapses to ``[B, N]``.  That is
    a genuine efficiency win (``O(B*N)`` trace memory, one outer product per step), but it
    is also a correctness claim about the derivation: :meth:`bptt_update_alignment` is
    what checks it, and the readout groups (whose updates are EXACT, not approximate)
    should score ``cosine ~ 1.0``.

MEASURED ALIGNMENT (toy batch, reduced regime, 20 iterations; see tests/test_rflo_rnn.py)
    Against autograd on the same deterministic rollout, per parameter group::

        group   symmetric feedback      random feedback
        w_o     cosine +1.0000          cosine +1.0000     <- exact, rel_err 0.0
        c_z     cosine +1.0000          cosine +1.0000     <- exact, rel_err 0.0
        J       cosine +0.868           cosine +0.076
        B       cosine +0.859           cosine +0.050
        c_x     cosine +0.865           cosine +0.075
        x0      cosine +0.196           cosine -0.233

    Read this as three separate facts.  The readout rows confirm the gating/masking
    bookkeeping is exactly right.  The symmetric column isolates the TEMPORAL truncation
    and shows it costs ~13% of direction on the recurrent groups -- healthy for truncated
    RTRL, and the main evidence that the ``phi'``-free trace above is derived correctly.
    The random column is low only because feedback alignment has not developed after 20
    iterations; that is the rule working as designed, not an error.

    ``x0`` is the weak group, and predictably so: its sensitivity is truncated all the
    way to ``(1-alpha)^t``, which at the reduced regime's ``alpha=0.5`` decays to nothing
    within a few steps while the true sensitivity persists through recurrence. The
    faithful regime (``alpha=0.1``) decays ~7x slower. ``x0`` is N of ~N^2 parameters, so
    this is not worth fixing with a more expensive trace -- but do not read its cosine as
    a bug report.

COST (measured, reduced regime: N=160, T=600, batch=48, CPU)
    ::

        forward only (the floor)   0.131 s/iter
        rflo                       0.214 s/iter
        bptt                       0.347 s/iter
        pc (20 inference steps)    0.580 s/iter

    RFLO is the cheap arm: ~1.6x faster than BPTT and ~2.7x faster than PC, and it sits
    close to the forward-only floor because it IS one forward pass -- an ``O(B*N^2)``
    outer product per step (same order as the forward ``r @ J.T`` itself), no backward
    sweep, and no relaxation loop.  It is also lighter in memory: chunked eligibility
    traces instead of BPTT's ``O(B*T*N)`` autograd tape.

    Getting there required batching the accumulation (see ``_ACCUM_CHUNK``); the literal
    per-timestep online form measured 0.480 s/iter, i.e. SLOWER than BPTT, entirely from
    kernel-launch overhead rather than arithmetic.

EXPECTED WEAKNESS -- measure it, do not design around it
    The eligibility trace decays with ``tau`` (10 ms by default) while Ready->Set spans up
    to 1200 ms, and long-range temporal credit assignment is RFLO's documented weak point.
    The mitigating argument is that RSG supervises the ENTIRE production epoch, so error
    pressure is dense at every timestep rather than sparse at the sequence end -- the same
    reasoning ``pc_rnn.py``'s ``_relax`` gives for skipping Millidge's reverse sweep.  If
    it does bite, it should show up as a flatter tp-vs-ts slope on the long prior.  Report
    that slope next to the similarity; per AGENTS.md it must NEVER gate seed inclusion.

REPORTED LOSS
    :meth:`infer_and_update` reports plain masked MSE, matching ``trainer.masked_mse`` and
    the BPTT arm, so ``summarize_runs.py``'s ``best_loss`` column is comparable between
    those two.  Note PC reports a ``0.5 *`` variant of the same quantity, so PC's numbers
    sit at half scale; that is pre-existing and not something this module changes.  The
    update math here is the gradient of ``0.5 * SSE`` (same convention as PC's energy),
    a constant factor that Adam absorbs.
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from src.models.bptt_rnn import BPTTRNN
from src.models.local_update import rescale_updates, update_alignment
from src.training.config import Config

__all__ = ["RFLORNN"]

#: Offset for the feedback matrix's RNG stream. ``BPTTRNN.__init__`` draws J, B, c_x, x0
#: from a generator seeded on ``cfg.seed`` in a fixed order; drawing the feedback matrix
#: from that same stream would shift it and silently break init parity with the other two
#: arms. A separate, offset stream keeps an RFLO seed's starting weights bit-identical to
#: the BPTT seed of the same number.
_FEEDBACK_SEED_OFFSET = 10_000

#: How many timesteps of per-step quantities to buffer before folding them into the
#: update accumulators with one einsum each.
#:
#: RFLO is an ONLINE rule and the reference formulation folds each step in as it happens.
#: Doing that literally means ~6 tiny kernel launches per timestep, and at T=600 that
#: overhead -- not arithmetic -- dominates: measured 0.480 s/iter against a 0.131 s/iter
#: forward-only floor in the reduced regime. Buffering a chunk and folding it in with one
#: batched einsum is NUMERICALLY the same update (the same sum, reassociated) while
#: cutting launches by this factor. It is the same trade ``PCRNN._relax`` documents for
#: vectorizing its relaxation over T.
#:
#: Chunking rather than buffering the whole trajectory is what keeps the memory story
#: honest: buffers stay ``O(B * chunk * N)`` (~10 MB at the faithful regime) instead of
#: growing to ``O(B * T * N)`` (~460 MB), so RFLO keeps a real memory advantage over
#: BPTT's autograd tape. Purely a performance knob -- changing it must not change results.
_ACCUM_CHUNK = 64


class RFLORNN(BPTTRNN):
    """RFLO learning rule over the BPTT arm's architecture. See the module docstring."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        gen = torch.Generator().manual_seed(cfg.seed + _FEEDBACK_SEED_OFFSET)
        scale = cfg.rflo_feedback_scale or (1.0 / math.sqrt(cfg.N))
        # A BUFFER, not a Parameter, and load-bearing that it stays one: the trainer
        # assigns `parameter.grad` for every entry of `model.named_parameters()` from the
        # returned `updates` dict, so a seventh parameter with no matching update key
        # raises KeyError -- and it would also collect a meaningless Adam slot. As a
        # buffer it still rides along in state_dict(), so a resumed run keeps the same
        # feedback (fresh random feedback mid-run would be a different learning rule).
        self.register_buffer(
            "feedback", torch.randn(self.w_o.shape, generator=gen) * scale
        )
        self.last_update_finite: Dict[str, bool] = {}

    # --- internals ---------------------------------------------------------------

    def _parameters_by_name(self) -> Dict[str, torch.Tensor]:
        """The six trainable tensors, keyed as ``named_parameters()`` keys them."""
        return {
            "J": self.J, "B": self.B, "c_x": self.c_x,
            "x0": self.x0, "w_o": self.w_o, "c_z": self.c_z,
        }

    def _effector_index(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the readout row selected by each trial's tonic effector input."""
        values = self._effector_values.to(device=inputs.device, dtype=inputs.dtype)
        return torch.argmin((inputs[:, 0, 2:3] - values.unsqueeze(0)).abs(), dim=-1)

    def _feedback_matrix(self) -> torch.Tensor:
        """The matrix that carries output error back into the population, ``[n_eff, N]``.

        ``"random"`` is RFLO proper -- fixed random feedback, no weight transport, the
        setting this arm is defined by.  ``"symmetric"`` substitutes ``w_o`` itself; it
        exists so :meth:`bptt_update_alignment` can isolate the TEMPORAL approximation
        (the trace truncation) from the SPATIAL one, and is a validation tool rather than
        a study condition.
        """
        mode = self.cfg.rflo_feedback
        if mode == "random":
            return self.feedback
        if mode == "symmetric":
            return self.w_o.detach()
        raise ValueError(f"unknown rflo_feedback {mode!r}")

    # --- the learning rule -------------------------------------------------------

    def infer_and_update(self, inputs, target, mask, *, apply_update: bool = True, noise: bool = True):
        """Roll forward once carrying eligibility traces, and emit local RFLO updates.

        Named to match ``PCRNN.infer_and_update`` because that is the contract the
        trainer's local-rule branch already speaks -- but the name is a mild misnomer
        here: RFLO performs no inference and no relaxation.  It is a single forward
        sweep, and the "updates" fall out of it online.

        Args:
            inputs: ``[batch, time, 3]``; the third channel is the tonic effector context.
            target, mask: both ``[batch, time]``, over the scored output trajectory.
            apply_update: if true, apply ``p -= cfg.lr * update`` in place (the pure local
                rule). The trainer normally leaves this false and hands the updates to
                Adam instead, so the step-size policy matches the other two arms.
            noise: inject process noise into the rollout. True during training -- an
                online rule sees the trajectory the network actually takes, and this
                matches the BPTT arm's ``noise=True`` training forward. Pass false for
                deterministic checks.

        Returns:
            Dict with ``loss`` (masked MSE), ``updates`` (detached, keyed by parameter
            name -- one entry per named parameter, which the trainer relies on),
            ``outputs`` ``[batch, time]``, and ``finite`` per-update flags.

        Raises:
            ValueError: on shape violations.
            FloatingPointError: on a non-finite update, loss, or post-update parameter.
        """
        if inputs.ndim != 3 or target.shape != inputs.shape[:2] or mask.shape != target.shape:
            raise ValueError("expected inputs [B,T,3], target and mask [B,T]")

        batch, time, _ = inputs.shape
        device, dtype = inputs.device, inputs.dtype
        alpha = self.cfg.alpha
        noise_scale = math.sqrt(2 * alpha) * self.cfg.noise_sd

        idx = self._effector_index(inputs)                       # [B]
        feedback = self._feedback_matrix()[idx]                  # [B, N]

        with torch.no_grad():
            x = self.x0.to(dtype).unsqueeze(0).expand(batch, -1)
            trace_rate = torch.zeros(batch, self.cfg.N, device=device, dtype=dtype)
            trace_input = torch.zeros(batch, self.N_IN, device=device, dtype=dtype)
            trace_bias = 0.0
            trace_x0 = 1.0

            updates = {
                name: torch.zeros_like(p) for name, p in self._parameters_by_name().items()
            }
            outputs = []
            squared_error = torch.zeros((), device=device, dtype=dtype)

            # Per-step quantities awaiting a batched fold-in; see _ACCUM_CHUNK.
            buf_e, buf_rate, buf_input, buf_delta, buf_r = [], [], [], [], []
            buf_bias, buf_x0 = [], []

            def fold_chunk() -> None:
                """Fold the buffered timesteps into the update accumulators."""
                if not buf_e:
                    return
                e_c = torch.stack(buf_e, dim=1)                    # [B, C, N]
                r_c = torch.stack(buf_r, dim=1)                    # [B, C, N]
                delta_c = torch.stack(buf_delta, dim=1)            # [B, C]
                bias_c = torch.tensor(buf_bias, device=device, dtype=dtype)
                x0_c = torch.tensor(buf_x0, device=device, dtype=dtype)

                updates["J"] += torch.einsum("bti,btj->ij", e_c, torch.stack(buf_rate, 1))
                updates["B"] += torch.einsum("bti,btk->ik", e_c, torch.stack(buf_input, 1))
                updates["c_x"] += torch.einsum("bti,t->i", e_c, bias_c)
                updates["x0"] += torch.einsum("bti,t->i", e_c, x0_c)
                # Readout updates need no temporal credit assignment -- they are exact.
                updates["w_o"].index_add_(0, idx, torch.einsum("bt,bti->bi", delta_c, r_c))
                updates["c_z"].index_add_(0, idx, delta_c.sum(dim=1))

                for buf in (buf_e, buf_rate, buf_input, buf_delta, buf_r, buf_bias, buf_x0):
                    buf.clear()

            for t in range(time):
                r = torch.tanh(x)
                both = r @ self.w_o.t() + self.c_z
                z = both.gather(-1, idx.view(batch, 1)).squeeze(-1)      # [B]
                outputs.append(z)

                delta = (z - target[:, t]) * mask[:, t]                  # [B]
                squared_error = squared_error + delta.square().sum()
                # Three factors: top-down error, feedback projection, local excitability.
                e = delta.unsqueeze(-1) * feedback * (1.0 - r.square())  # [B, N]

                u = inputs[:, t, :]
                buf_e.append(e)
                buf_r.append(r)
                buf_delta.append(delta)
                buf_rate.append(trace_rate)
                buf_input.append(trace_input)
                buf_bias.append(trace_bias)
                buf_x0.append(trace_x0)
                if len(buf_e) == _ACCUM_CHUNK:
                    fold_chunk()

                trace_rate = (1.0 - alpha) * trace_rate + alpha * r
                trace_input = (1.0 - alpha) * trace_input + alpha * u
                trace_bias = (1.0 - alpha) * trace_bias + alpha
                trace_x0 = (1.0 - alpha) * trace_x0

                x = x + alpha * (-x + r @ self.J.t() + u @ self.B.t() + self.c_x)
                if noise:
                    x = x + noise_scale * torch.randn(x.shape, device=device, dtype=dtype)

            fold_chunk()

            updates = rescale_updates(
                updates, mask, self.cfg.grad_clip, self.cfg.rflo_clip_mode
            )
            finite = {name: bool(torch.isfinite(v).all()) for name, v in updates.items()}
            loss = float(squared_error.item() / max(float(mask.sum().item()), 1.0))
            if not all(finite.values()) or not math.isfinite(loss):
                raise FloatingPointError("non-finite RFLO update or loss")

            if apply_update:
                parameters = self._parameters_by_name()
                for name, parameter in parameters.items():
                    parameter.add_(-self.cfg.lr * updates[name])
                if not all(torch.isfinite(p).all() for p in parameters.values()):
                    raise FloatingPointError("non-finite parameter after RFLO update")

        self.last_update_finite = finite
        return {
            "loss": loss,
            "updates": {name: v.detach().clone() for name, v in updates.items()},
            "outputs": torch.stack(outputs, dim=1).detach().clone(),
            "finite": finite,
        }

    # --- diagnostics -------------------------------------------------------------

    def bptt_update_alignment(self, inputs, target, mask) -> Dict[str, Dict[str, float]]:
        """Compare RFLO's local updates with the autograd gradients they approximate.

        Diagnostic and NON-GATING: it reports whether the truncated-RTRL derivation points
        the same way as BPTT, but it does not decide whether a seed or an implementation
        passes.  Both sides run ``noise=False`` so the comparison is deterministic.

        Expectations, useful when reading the numbers:

        * ``w_o`` and ``c_z`` are computed EXACTLY, so they should score ``cosine ~ 1.0``.
          Anything else means the readout/effector-gating bookkeeping is wrong.
        * ``J``, ``B``, ``c_x``, ``x0`` are approximate. Under
          ``rflo_feedback="symmetric"`` they isolate the temporal truncation and should be
          clearly positive; under ``"random"`` they also carry the feedback-alignment
          error and can start near zero before alignment develops.
        * At initialization ``w_o`` is all zeros, so no error can reach the recurrent
          parameters and their autograd gradient is EXACTLY zero -- cosine is ``nan``
          there by construction, not by failure. Train a few steps (or perturb ``w_o``)
          before reading those entries.

        Returns:
            ``{group: {"cosine": float, "relative_error": float}}`` for all six groups.
        """
        result = self.infer_and_update(inputs, target, mask, apply_update=False, noise=False)
        self.zero_grad(set_to_none=True)
        outputs, _ = self.forward(inputs, noise=False)
        loss = 0.5 * (((outputs - target).square()) * mask).sum() / mask.sum().clamp_min(1)
        loss.backward()
        gradients = {name: p.grad for name, p in self._parameters_by_name().items()}
        return {
            name: update_alignment(result["updates"][name], gradient)
            for name, gradient in gradients.items()
        }
