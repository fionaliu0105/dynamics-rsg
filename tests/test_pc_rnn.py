"""Deterministic PC-A checks for the shared continuous-time RNN."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.models.base import check_interface
from src.models.bptt_rnn import BPTTRNN
from src.models.pc_rnn import PCRNN
from src.training.config import Config


def _toy_batch():
    cfg = Config.reduced(
        rule="pc", seed=17, N=12, total_time=100.0, ready_onset=10.0,
        pulse_width=5.0, prod_hold=10.0, lr=2e-3, pc_inference_lr=0.05,
        pc_inference_steps=8,
    )
    inputs = torch.zeros(4, 20, 3)
    inputs[:, :, 1] = 0.3
    inputs[:2, :, 2] = cfg.effector_context["eye"]
    inputs[2:, :, 2] = cfg.effector_context["hand"]
    inputs[:, 2:4, 0] = cfg.pulse_height
    inputs[:, 8:10, 0] = cfg.pulse_height
    target = torch.zeros(4, 20)
    target[:, 8:] = torch.linspace(0.0, 0.8, 12)
    mask = torch.zeros_like(target)
    mask[:, 8:] = 1.0
    return cfg, inputs, target, mask


def test_forward_is_shared_bptt_computation():
    cfg, inputs, _, _ = _toy_batch()
    pc = PCRNN(cfg)
    direct = BPTTRNN(cfg)
    direct.load_state_dict(pc.dynamics.state_dict())
    pc_outputs, pc_states = pc(inputs, noise=False)
    bptt_outputs, bptt_states = direct(inputs, noise=False)
    assert torch.equal(pc_outputs, bptt_outputs)
    assert torch.equal(pc_states, bptt_states)
    check_interface(pc_outputs, pc_states, inputs.shape[0], inputs.shape[1])


def test_energy_descends_and_updates_are_finite():
    cfg, inputs, target, mask = _toy_batch()
    result = PCRNN(cfg).infer_and_update(inputs, target, mask, apply_update=False)
    trace = torch.tensor(result["energy_trace"])
    assert torch.isfinite(trace).all()
    assert torch.all(trace[1:] <= trace[:-1] + 1e-6), trace.tolist()
    assert all(result["finite"].values())
    assert torch.isfinite(result["values"]).all()
    assert torch.isfinite(result["outputs"]).all()


def test_toy_loss_decreases_and_alignment_is_reported():
    cfg, inputs, target, mask = _toy_batch()
    model = PCRNN(cfg)
    initial = None
    final = None
    for _ in range(160):
        result = model.infer_and_update(inputs, target, mask)
        if initial is None:
            initial = result["loss"]
        final = result["loss"]
    assert final < initial * 0.7, (initial, final)

    # PC-A records this across inference-step values; weak alignment is not failure.
    for steps in (1, 4, 8):
        model.cfg.pc_inference_steps = steps
        alignment = model.bptt_update_alignment(inputs, target, mask)
        assert alignment.keys() == {"J", "B", "c_x", "x0", "w_o", "c_z"}
        for values in alignment.values():
            assert set(values) == {"cosine", "relative_error"}
            assert torch.isfinite(torch.tensor(values["relative_error"]))


def _scale_batch(batch: int = 32, N: int = 64, T: int = 80):
    """A batch with a realistic batch*time so the summed-update scale bites.

    The ``_toy_batch`` above is ``N=12, batch=4, T=20`` -> ``batch*time ~= 76``,
    small enough that the raw (summed-over-batch-and-time) PC update stayed
    finite for 160 iters even before normalization existed -- which is exactly
    why the older tests did not catch the reduced-regime divergence. Here
    ``batch*time ~= 2500`` so the default normalize+clip path is actually needed.
    """
    cfg = Config.reduced(
        rule="pc", seed=5, N=N, total_time=float(T * 5), ready_onset=20.0,
        pulse_width=10.0, prod_hold=40.0, batch=batch, lr=1e-3,
        pc_inference_lr=0.1, pc_inference_steps=10,
    )
    inputs = torch.zeros(batch, T, 3)
    inputs[:, :, 1] = 0.3
    half = batch // 2
    inputs[:half, :, 2] = cfg.effector_context["eye"]
    inputs[half:, :, 2] = cfg.effector_context["hand"]
    inputs[:, 4:6, 0] = cfg.pulse_height
    inputs[:, 20:22, 0] = cfg.pulse_height
    target = torch.zeros(batch, T)
    target[:, 20:] = torch.linspace(0.0, 0.8, T - 20)
    mask = torch.zeros_like(target)
    mask[:, 20:] = 1.0
    return cfg, inputs, target, mask


def _global_step_norm(model, inputs, target, mask) -> float:
    """Global L2 norm of the parameter change from one applied PC update."""
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    model.infer_and_update(inputs, target, mask)
    sq = sum(((p.detach() - before[name]) ** 2).sum() for name, p in model.named_parameters())
    return float(sq.sqrt())


def test_applied_pc_step_is_norm_clipped():
    """Regression: the applied step obeys the norm clip regardless of batch*time.

    Raw PC updates sum over batch and time, so at reduced-regime scale they are
    ~B*T larger than the BPTT arm's mean-reduced step and diverge in ~7-10 iters.
    With the defaults (normalize + clip) the applied step is bounded by
    ``lr * pc_grad_clip``; with the fix disabled the very first step is far larger.
    """
    cfg, inputs, target, mask = _scale_batch()
    assert cfg.pc_normalize and cfg.pc_grad_clip > 0  # defaults keep the guard on
    clipped = _global_step_norm(PCRNN(cfg), inputs, target, mask)
    assert clipped <= cfg.lr * cfg.pc_grad_clip * (1 + 1e-4), clipped

    cfg_raw = Config.from_dict({**cfg.to_dict(), "pc_normalize": False, "pc_grad_clip": 0.0})
    raw = _global_step_norm(PCRNN(cfg_raw), inputs, target, mask)
    assert raw > 10 * clipped, (raw, clipped)  # the guard is doing real work


def test_pc_is_stable_over_iterations_at_scale():
    """Regression: PC trains without diverging at a scale that used to blow up."""
    cfg, inputs, target, mask = _scale_batch()
    model = PCRNN(cfg)
    j0 = model.J.detach().norm().item()
    losses = [model.infer_and_update(inputs, target, mask)["loss"] for _ in range(25)]
    assert all(np.isfinite(losses)), losses
    assert model.J.detach().norm().item() < 5 * j0  # weights stay bounded
    assert losses[-1] <= losses[0]                   # learning, not diverging


if __name__ == "__main__":
    test_forward_is_shared_bptt_computation()
    test_energy_descends_and_updates_are_finite()
    test_toy_loss_decreases_and_alignment_is_reported()
    test_applied_pc_step_is_norm_clipped()
    test_pc_is_stable_over_iterations_at_scale()
    print("PC-A deterministic checks passed")
