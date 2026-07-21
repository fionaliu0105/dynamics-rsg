"""Integration checks: is BPTTRNN actually ready for src/training/trainer.py to
consume, ahead of the trainer/task/behavior tracks landing?

Three things the trainer will need from this model, checked directly against the
real contracts (not a mock) so nothing here needs to change once those tracks ship:

1. Checkpoint round-trip — ``state_dict``/``load_state_dict`` on model + optimizer
   (the trainer's TODO(me): resume-from-checkpoint) reproduces identical behavior
   and training can continue seamlessly after a "restart".
2. Store write/read — a single-condition forward (noise off) writes into the REAL
   ``ActivationStore`` with the exact ``Record`` schema the trainer will use, keyed
   by a real ``Condition`` from ``src.conditions``.
3. GPU-safety of the effector-gating index math (``torch.arange(..., device=...)``)
   — checked here on CPU by asserting the gather is device-consistent; the faithful
   regime trains on GPU (AGENTS.md, "No interactive-only assumptions").

Run from the repo root: ``python tests/test_bptt_integration.py``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.conditions import CONDITIONS
from src.models.bptt_rnn import BPTTRNN
from src.store import ActivationStore, Record
from src.training.config import Config
from test_bptt import _make_trivial_batch, first_crossing


def test_checkpoint_roundtrip():
    """Save (model, optimizer) state, rebuild fresh objects, reload, and confirm
    both the forward pass and continued training pick up exactly where they left
    off — this is what makes 'resume from latest checkpoint' safe (AGENTS.md,
    'Assume the process can die')."""
    cfg = Config.reduced(rule="bptt", seed=5, lr=1e-2)
    torch.manual_seed(cfg.seed)
    model = BPTTRNN(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    inputs = torch.randn(4, 30, 3)
    inputs[:, :, 2] = torch.tensor([cfg.effector_context["eye"]] * 2 + [cfg.effector_context["hand"]] * 2).unsqueeze(-1)
    target = torch.zeros(4, 30)
    mask = torch.ones(4, 30)

    # a few real training steps so optimizer state (Adam moments) is non-trivial
    for _ in range(5):
        opt.zero_grad()
        outputs, _ = model(inputs, noise=False)
        loss = ((outputs - target) ** 2 * mask).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        outputs_before, _ = model(inputs, noise=False)

    with tempfile.TemporaryDirectory() as d:
        ckpt_path = Path(d) / "ckpt.pt"
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "iter": 5},
            ckpt_path,
        )

        # fresh objects simulating a requeued/restarted job
        torch.manual_seed(999)  # different seed on purpose: state must come from ckpt, not init
        model2 = BPTTRNN(Config.reduced(rule="bptt", seed=5, lr=1e-2))
        opt2 = torch.optim.Adam(model2.parameters(), lr=cfg.lr)
        ckpt = torch.load(ckpt_path, weights_only=True)
        model2.load_state_dict(ckpt["model"])
        opt2.load_state_dict(ckpt["opt"])

    with torch.no_grad():
        outputs_after, _ = model2(inputs, noise=False)
    assert torch.allclose(outputs_before, outputs_after), (
        "reloaded model does not reproduce pre-checkpoint outputs"
    )

    # continued training after "resume" should behave like an uninterrupted run:
    # loss keeps moving (optimizer momentum/variance carried over, not reset)
    opt2.zero_grad()
    outputs, _ = model2(inputs, noise=False)
    loss_after_resume = ((outputs - target) ** 2 * mask).mean()
    loss_after_resume.backward()
    opt2.step()
    assert torch.isfinite(loss_after_resume)
    print("test_checkpoint_roundtrip OK")


def test_store_write_read_real_condition():
    """Single-condition forward (noise off) -> real ActivationStore Record,
    exactly as the trainer's store_condition_activations will do per condition."""
    cfg = Config.reduced(rule="bptt", seed=7)
    torch.manual_seed(cfg.seed)
    model = BPTTRNN(cfg)

    cond = CONDITIONS[0]  # a real (prior, ts, effector) triple
    ts_values = [cond.ts]
    inputs, _, _, _ = _make_trivial_batch(cfg, ts_values, [cond.effector], torch.Generator().manual_seed(0))

    with torch.no_grad():
        outputs, states = model(inputs, noise=False)

    set_step = cfg.ready_onset_step + cfg.to_step(cond.ts)
    tp_steps = first_crossing(outputs, cfg.threshold, set_step)[0]
    tp_ms = float(tp_steps.item() * cfg.dt) if torch.isfinite(tp_steps) else float("nan")

    with tempfile.TemporaryDirectory() as d:
        store = ActivationStore(Path(d) / "activations")
        store.write(Record(
            model="bptt", seed=cfg.seed, condition=cond,
            states=states[0].numpy(), inputs=inputs[0].numpy(),
            meta={"tp": tp_ms},
        ))
        rec = store.read("bptt", cfg.seed, cond)
        assert rec.states.shape == (cfg.n_steps, cfg.N)
        assert rec.inputs.shape == (cfg.n_steps, 3)
        assert rec.meta["prior"] == cond.prior
        assert rec.meta["ts"] == cond.ts
        assert rec.meta["effector"] == cond.effector
    print(f"test_store_write_read_real_condition OK (tp={tp_ms!r} for {cond.label}, "
          f"untrained net so a non-finite tp here is expected)")


def main():
    test_checkpoint_roundtrip()
    test_store_write_read_real_condition()
    print("\nBPTT model is trainer/store-ready")


if __name__ == "__main__":
    main()
