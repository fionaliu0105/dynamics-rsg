"""Self-checks for the BPTT track (plan 1.B), against the definition of done in
``src/models/bptt_rnn.py``: trains on trivial data, loss drops, tp is finite and
ordered, and the correct effector channel is the one that crosses threshold.

The real task generator (``src/task/rsg.py``, a separate track) is still a stub,
so this test builds its OWN trivial synthetic batches — a mock, not the real task
interface — just enough to exercise ``BPTTRNN.forward`` end to end. Swap in
``src.task.rsg.make_batch`` once that track lands; the model itself doesn't change.

Requires torch. Run from the repo root: ``python tests/test_bptt.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models.base import check_interface
from src.models.bptt_rnn import EFFECTOR_ORDER, BPTTRNN
from src.training.config import Config


def _make_trivial_batch(cfg: Config, ts_values, effectors, rng: torch.Generator):
    """Toy (inputs, target, mask, effector_labels) batch: one trial per (ts, effector).

    Ready/Set pulses ``ts`` ms apart; target ramps 0 -> threshold over the
    production epoch; mask is 1 over that epoch. No prior-mean jitter (that's the
    Bayesian-bias mechanism, owned by the task track, plan 1.A) — this is only
    meant to exercise the network's dynamics and readout.
    """
    n = len(ts_values) * len(effectors)
    inputs = torch.zeros(n, cfg.n_steps, 3)
    target = torch.zeros(n, cfg.n_steps)
    mask = torch.zeros(n, cfg.n_steps)
    effector_labels = []

    i = 0
    for ts in ts_values:
        for eff in effectors:
            ready_step = cfg.ready_onset_step
            ts_step = cfg.to_step(ts)
            set_step = ready_step + ts_step
            pw = cfg.pulse_width_step

            inputs[i, ready_step:ready_step + pw, 0] = cfg.pulse_height
            inputs[i, set_step:set_step + pw, 0] = cfg.pulse_height
            inputs[i, :, 1] = cfg.prior_context["short" if ts <= 800 else "long"]
            inputs[i, :, 2] = cfg.effector_context[eff]

            prod_end = min(set_step + ts_step, cfg.n_steps)
            ramp_len = max(prod_end - set_step, 1)
            ramp = torch.linspace(0.0, cfg.threshold, ramp_len)
            target[i, set_step:prod_end] = ramp
            hold_end = min(prod_end + cfg.prod_hold_step, cfg.n_steps)
            target[i, prod_end:hold_end] = cfg.threshold
            mask[i, set_step:hold_end] = 1.0

            effector_labels.append(eff)
            i += 1
    return inputs, target, mask, effector_labels


def first_crossing(outputs: torch.Tensor, threshold: float, start_step: int) -> torch.Tensor:
    """First step (relative to ``start_step``) each trial's output >= threshold.

    Returns ``float('nan')`` (as a tensor entry) for trials that never cross.
    """
    n, T = outputs.shape
    tp = torch.full((n,), float("nan"))
    for i in range(n):
        crossed = (outputs[i, start_step:] >= threshold).nonzero(as_tuple=True)[0]
        if len(crossed) > 0:
            tp[i] = float(crossed[0].item())
    return tp


def test_interface_shapes():
    cfg = Config.reduced(rule="bptt", seed=0)
    model = BPTTRNN(cfg)
    inputs = torch.zeros(5, cfg.n_steps, 3)
    outputs, states = model(inputs, noise=False)
    check_interface(outputs, states, n_trials=5, n_time=cfg.n_steps)
    print("test_interface_shapes OK")


def test_effector_gating():
    """The channel that crosses threshold matches the trial's effector context."""
    cfg = Config.reduced(rule="bptt", seed=1)
    model = BPTTRNN(cfg)
    ts_values = [480, 800]
    effectors = list(EFFECTOR_ORDER)
    inputs, _, _, effector_labels = _make_trivial_batch(
        cfg, ts_values, effectors, torch.Generator().manual_seed(0)
    )
    outputs, _ = model(inputs, noise=False)
    both = model._last_outputs_both  # [B, T, 2], both channels pre-gate
    for i, eff in enumerate(effector_labels):
        idx = EFFECTOR_ORDER.index(eff)
        assert torch.allclose(outputs[i], both[i, :, idx]), (
            f"trial {i} (effector={eff}) was not gated to its own channel"
        )
    print("test_effector_gating OK")


def test_trains_and_tp_finite_ordered():
    """Loss drops with training; produced tp is finite and increases with ts.

    Uses a deliberately SHORT sequence (not ``Config.reduced()``'s full 520-step
    task horizon) — this checks the RNN mechanics (dynamics + gradient flow +
    effector gating) converge, not whether the full task is Bayes-consistent.
    ``Config.reduced()`` itself is documented (README) to under-train on purpose;
    conflating that with this track's unit check would make the test flaky/slow.
    """
    cfg = Config.reduced(
        rule="bptt", seed=2, n_iter=1000, lr=1e-2,
        total_time=220.0, ready_onset=20.0, pulse_width=10.0, prod_hold=20.0,
    )
    torch.manual_seed(cfg.seed)
    model = BPTTRNN(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    ts_values = [40, 60, 80]
    inputs, target, mask, effector_labels = _make_trivial_batch(
        cfg, ts_values, ["eye"], torch.Generator().manual_seed(0)
    )

    losses = []
    for _ in range(cfg.n_iter):
        opt.zero_grad()
        outputs, _ = model(inputs, noise=True)
        loss = ((outputs - target) ** 2 * mask).sum() / mask.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, (
        f"loss did not drop enough: start={losses[0]:.4f} end={losses[-1]:.4f}"
    )

    with torch.no_grad():
        outputs, _ = model(inputs, noise=False)
    set_steps = [cfg.ready_onset_step + cfg.to_step(ts) for ts in ts_values]
    tp = torch.stack([
        first_crossing(outputs[i:i + 1], cfg.threshold, set_steps[i])[0]
        for i in range(len(ts_values))
    ])
    assert torch.isfinite(tp).all(), f"tp has non-finite entries: {tp}"
    tp_ms = tp * cfg.dt
    assert (tp_ms[1:] >= tp_ms[:-1]).all(), f"tp not ordered with ts: {tp_ms.tolist()}"
    print(f"test_trains_and_tp_finite_ordered OK (loss {losses[0]:.4f} -> {losses[-1]:.4f}, "
          f"tp={tp_ms.tolist()} for ts={ts_values})")


def main():
    test_interface_shapes()
    test_effector_gating()
    test_trains_and_tp_finite_ordered()
    print("\nall BPTT track checks passed")


if __name__ == "__main__":
    main()
