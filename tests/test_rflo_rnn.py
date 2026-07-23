"""Deterministic RFLO checks for the shared continuous-time RNN.

Mirrors tests/test_pc_rnn.py: same toy batch, same shape of argument. The two things
these tests exist to catch are (a) architecture parity silently breaking, which would
make any rule-vs-rule result uninterpretable, and (b) a "loss falls but the recurrent
weights never moved" false pass, which a loss-only test happily accepts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models.base import check_interface
from src.models.bptt_rnn import BPTTRNN
from src.models.rflo_rnn import RFLORNN
from src.training.config import Config


def _toy_batch(**overrides):
    cfg = Config.reduced(
        rule="rflo", seed=17, N=12, total_time=100.0, ready_onset=10.0,
        pulse_width=5.0, prod_hold=10.0, lr=2e-3, noise_sd=0.0, **overrides,
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
    """Architecture parity: RFLO's rollout IS the BPTT arm's, not a copy of it."""
    cfg, inputs, _, _ = _toy_batch()
    rflo = RFLORNN(cfg)
    direct = BPTTRNN(cfg)
    rflo_outputs, rflo_states = rflo(inputs, noise=False)
    bptt_outputs, bptt_states = direct(inputs, noise=False)
    assert torch.equal(rflo_outputs, bptt_outputs)
    assert torch.equal(rflo_states, bptt_states)
    check_interface(rflo_outputs, rflo_states, inputs.shape[0], inputs.shape[1])


def test_initial_weights_match_bptt_at_the_same_seed():
    """The feedback matrix must not perturb the shared init RNG stream.

    RFLO draws its feedback from a separate, offset generator precisely so that seed N
    of each arm starts from bit-identical weights. Drawing it from the shared stream
    would shift J/B/c_x/x0 and quietly confound every rule-vs-rule comparison with an
    initialization difference.
    """
    cfg, _, _, _ = _toy_batch()
    rflo = RFLORNN(cfg)
    bptt = BPTTRNN(cfg)
    for name in ("J", "B", "c_x", "x0", "w_o", "c_z"):
        assert torch.equal(getattr(rflo, name), getattr(bptt, name)), name


def test_feedback_is_a_buffer_not_a_parameter():
    """A seventh *parameter* would KeyError in the trainer's update-assignment loop.

    trainer.py assigns `parameter.grad` for every named_parameters() entry out of the
    returned `updates` dict, so the feedback matrix must not appear there — but it must
    appear in state_dict(), or a resumed run would draw fresh feedback mid-training,
    which is a different learning rule.
    """
    cfg, _, _, _ = _toy_batch()
    model = RFLORNN(cfg)
    assert "feedback" not in dict(model.named_parameters())
    assert "feedback" in model.state_dict()

    updates = model.infer_and_update(
        *_toy_batch()[1:], apply_update=False, noise=False
    )["updates"]
    assert set(updates) == set(dict(model.named_parameters()))


def test_updates_are_normalized_and_recurrent_weights_train():
    """Loss falls AND J actually moves — the second assertion is the load-bearing one."""
    cfg, inputs, target, mask = _toy_batch()
    torch.manual_seed(0)
    model = RFLORNN(cfg)

    # Updates must be on the BPTT arm's scale (a masked mean), not a raw batch*time sum.
    with torch.no_grad():
        model.w_o.normal_(0.0, 0.05)
    updates = model.infer_and_update(
        inputs, target, mask, apply_update=False, noise=False
    )["updates"]
    assert updates["w_o"].norm() < 10.0, (
        f"readout update looks unnormalized: {updates['w_o'].norm():.3g}"
    )

    torch.manual_seed(0)
    model = RFLORNN(cfg)
    initial_J = model.J.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(150):
        result = model.infer_and_update(
            inputs, target, mask, apply_update=False, noise=False
        )
        for name, parameter in model.named_parameters():
            parameter.grad = result["updates"][name].clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)
        losses.append(result["loss"])

    assert all(torch.isfinite(torch.tensor(losses))), "RFLO training went non-finite"
    assert losses[-1] < losses[0] * 0.7, (losses[0], losses[-1])
    moved = (model.J.detach() - initial_J).norm() / initial_J.norm()
    assert moved > 1e-3, f"recurrent weights effectively frozen: relative move {moved:.2e}"


def test_readout_updates_are_exact_and_recurrent_updates_align():
    """The readout groups are computed exactly, so they must score cosine ~ 1.

    Anything else means the effector-gating or masking bookkeeping is wrong. The
    recurrent groups are approximate by construction (truncated RTRL), so they only
    have to point the same way as autograd, and only once training has begun: w_o
    starts at zeros, which makes the recurrent autograd gradient EXACTLY zero and its
    cosine nan. Read under symmetric feedback so this isolates the temporal truncation
    from the feedback-alignment error.
    """
    cfg, inputs, target, mask = _toy_batch(rflo_feedback="symmetric")
    torch.manual_seed(0)
    model = RFLORNN(cfg)
    for _ in range(20):
        model.infer_and_update(inputs, target, mask, apply_update=True, noise=False)

    alignment = model.bptt_update_alignment(inputs, target, mask)
    assert alignment.keys() == {"J", "B", "c_x", "x0", "w_o", "c_z"}

    for group in ("w_o", "c_z"):
        assert alignment[group]["cosine"] > 0.999, (group, alignment[group])
    for group in ("J", "B", "c_x"):
        assert alignment[group]["cosine"] > 0.0, (group, alignment[group])


def test_chunk_size_is_a_pure_performance_knob(monkeypatch):
    """_ACCUM_CHUNK reassociates the update sum; it must never change the result.

    The chunked fold-in is what makes RFLO cheaper than BPTT in wall-clock (0.214 vs
    0.347 s/iter measured in the reduced regime) by cutting per-timestep kernel
    launches. That optimization is only legitimate if the updates it produces are the
    same ones the literal online rule produces, including when the trajectory length is
    not a multiple of the chunk size.
    """
    from src.models import rflo_rnn

    cfg, inputs, target, mask = _toy_batch()
    reference = None
    for chunk in (1, 3, 7, 64, 1000):
        monkeypatch.setattr(rflo_rnn, "_ACCUM_CHUNK", chunk)
        torch.manual_seed(0)
        updates = RFLORNN(cfg).infer_and_update(
            inputs, target, mask, apply_update=False, noise=False
        )["updates"]
        if reference is None:
            reference = updates
            continue
        for name, value in updates.items():
            assert torch.allclose(value, reference[name], atol=1e-6), (chunk, name)


def test_random_feedback_does_not_track_the_readout():
    """Guard the defining property of the arm: no weight transport under "random".

    If this ever starts matching the symmetric result, the feedback pathway has been
    wired to w_o and the arm is no longer RFLO.
    """
    cfg, inputs, target, mask = _toy_batch()
    torch.manual_seed(0)
    model = RFLORNN(cfg)
    with torch.no_grad():
        model.w_o.normal_(0.0, 0.05)

    model.cfg.rflo_feedback = "random"
    random_updates = model.infer_and_update(
        inputs, target, mask, apply_update=False, noise=False
    )["updates"]
    model.cfg.rflo_feedback = "symmetric"
    symmetric_updates = model.infer_and_update(
        inputs, target, mask, apply_update=False, noise=False
    )["updates"]

    # Readout updates never touch the feedback pathway, so they must be identical...
    assert torch.allclose(random_updates["w_o"], symmetric_updates["w_o"])
    # ...while the recurrent update, which does, must not be.
    assert not torch.allclose(random_updates["J"], symmetric_updates["J"])


if __name__ == "__main__":
    test_forward_is_shared_bptt_computation()
    test_initial_weights_match_bptt_at_the_same_seed()
    test_feedback_is_a_buffer_not_a_parameter()
    test_updates_are_normalized_and_recurrent_weights_train()
    test_readout_updates_are_exact_and_recurrent_updates_align()
    test_random_feedback_does_not_track_the_readout()
    print("RFLO deterministic checks passed")
