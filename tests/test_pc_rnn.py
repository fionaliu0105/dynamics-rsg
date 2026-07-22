"""Deterministic PC-A checks for the shared continuous-time RNN."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


if __name__ == "__main__":
    test_forward_is_shared_bptt_computation()
    test_energy_descends_and_updates_are_finite()
    test_toy_loss_decreases_and_alignment_is_reported()
    print("PC-A deterministic checks passed")


def test_updates_are_normalized_and_recurrent_weights_train():
    """Guard the two defects that stopped PC learning anything recurrent.

    Before the fix ``_local_updates`` returned raw *sums* over batch x time with no
    normalization and no ``grad_clip``, so at ``cfg.lr`` the readout exploded and the
    run went non-finite within ~9 iterations. Normalizing alone is not enough: PC's
    recurrent update is orders of magnitude smaller than its readout update, so under
    plain SGD ``J`` stays frozen while the loss still falls -- which a loss-only test
    happily passes. Both assertions below fail on the pre-fix implementation.
    """
    cfg, inputs, target, mask = _toy_batch()
    torch.manual_seed(0)
    model = PCRNN(cfg)

    # 1. Updates must be on the BPTT arm's scale (a masked mean), not a raw sum.
    with torch.no_grad():
        model.w_o.normal_(0.0, 0.05)
    updates = model.infer_and_update(inputs, target, mask, apply_update=False)["updates"]
    assert updates["w_o"].norm() < 10.0, (
        f"readout update looks unnormalized: {updates['w_o'].norm():.3g}"
    )

    # 2. The recurrent matrix must actually move over training.
    torch.manual_seed(0)
    model = PCRNN(cfg)
    initial_J = model.J.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        result = model.infer_and_update(inputs, target, mask, apply_update=False)
        for name, parameter in model.named_parameters():
            parameter.grad = result["updates"][name].clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)
        losses.append(result["loss"])

    assert all(torch.isfinite(torch.tensor(losses))), "PC training went non-finite"
    assert losses[-1] < losses[0] * 0.7, (losses[0], losses[-1])
    moved = (model.J.detach() - initial_J).norm() / initial_J.norm()
    assert moved > 1e-3, f"recurrent weights effectively frozen: relative move {moved:.2e}"


def test_elementwise_clip_does_not_unfreeze_recurrent_weights_under_plain_sgd():
    """Documents a tested-and-rejected hypothesis: the clip TYPE is not the bottleneck.

    Millidge's reference ``rnn.py`` clamps each update element independently to
    ``[-clamp_val, clamp_val]`` rather than scaling the whole joint vector to a shared
    global norm (``cfg.pc_clip_mode="global_norm"``, the default). Reproducing that
    (``pc_clip_mode="elementwise"``) with plain SGD (``pc_optimizer="sgd"``, no
    external optimizer) was checked as an alternative to routing PC through Adam, on
    the theory that the global-norm clip's cross-parameter budget-sharing (J alone has
    ~25.6k elements competing with w_o for one pool) might be what starves J, not
    something inherent to PC's local rule.

    Measured: it isn't. Relative J movement is ~1e-5 under either clip mode -- clipping
    only caps values that are already large, and J's normalized gradient here is
    naturally tiny to begin with, so no clip variant amplifies it. This test guards
    against re-litigating that hypothesis without re-deriving it: if this ever starts
    passing (J moving meaningfully), that's a real change in the update math worth
    investigating, not a fluke to shrug off.
    """
    cfg, inputs, target, mask = _toy_batch()
    cfg.pc_optimizer = "sgd"
    cfg.pc_clip_mode = "elementwise"
    torch.manual_seed(0)
    model = PCRNN(cfg)
    initial_J = model.J.detach().clone()
    losses = []
    for _ in range(40):
        result = model.infer_and_update(inputs, target, mask, apply_update=True)
        losses.append(result["loss"])

    assert all(torch.isfinite(torch.tensor(losses))), "PC training went non-finite"
    moved = (model.J.detach() - initial_J).norm() / initial_J.norm()
    assert moved < 1e-3, (
        f"recurrent weights moved more than expected under elementwise-clip plain SGD "
        f"({moved:.2e}) -- the 'clip type is the bottleneck' hypothesis may no longer "
        f"be rejected; re-investigate rather than assuming Adam is still necessary."
    )
