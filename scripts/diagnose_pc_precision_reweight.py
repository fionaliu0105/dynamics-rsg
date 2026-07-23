"""Read-only diagnostic: does up-weighting temporal_error in relaxation change anything?

Does NOT modify `src/models/pc_rnn.py` or any other base file. This script
re-implements the relaxation math (`_relax`/`_energy_and_errors`/`_value_gradient`)
locally, as a copy with one added parameter (`temporal_weight`), using an
existing checkpoint's real, loaded (and never re-saved) J/B/c_x/w_o/c_z/x0. The
original model code and every helper it defines are untouched and unused here
except for read-only calls (`_effector_index`, `_raw_forward_values`,
`forward`).

Tests the precision-weighting idea from `.suplex/docs/discrepancy_log.md`
(2026-07-22, "what to do about it" discussion): if relaxation is exploiting a
huge imbalance in degrees of freedom to satisfy the target while barely
perturbing temporal self-consistency, then making self-consistency violations
much more costly in the SAME local energy (no new non-local channel) should
force the relaxed trajectory to stay closer to what `forward()` could actually
produce on its own -- at the cost of a worse target fit. This checks whether
that trade-off actually appears, and where.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.task import make_batch
from src.training.config import Config
from src.training.trainer import build_model, masked_mse

TEMPORAL_WEIGHTS = [1.0, 10.0, 100.0, 1000.0]


def energy_and_errors(model, values, inputs, target, mask, temporal_weight: float):
    alpha = model.cfg.alpha
    r = torch.tanh(values)
    pred_next = values[:, :-1] + alpha * (
        -values[:, :-1] + r[:, :-1] @ model.J.detach().t() + inputs[:, :-1] @ model.B.detach().t() + model.c_x.detach()
    )
    temporal_error = values[:, 1:] - pred_next
    both = r @ model.w_o.detach().t() + model.c_z.detach()
    idx = model._effector_index(inputs).view(-1, 1, 1).expand(-1, values.shape[1], 1)
    output = both.gather(-1, idx).squeeze(-1)
    output_error = (output - target) * mask
    energy = 0.5 * (temporal_weight * temporal_error.square().sum() + output_error.square().sum())
    return energy, temporal_error, output_error, r, output


def value_gradient(model, values, inputs, temporal_error, output_error, mask, temporal_weight: float):
    alpha = model.cfg.alpha
    derivative = 1.0 - torch.tanh(values).square()
    grad = torch.zeros_like(values)
    grad[:, 1:] += temporal_weight * temporal_error

    idx = model._effector_index(inputs)
    selected_w = model.w_o.detach()[idx]
    grad += (output_error.unsqueeze(-1) * selected_w.unsqueeze(1)) * derivative

    propagated = temporal_weight * (
        (1.0 - alpha) * temporal_error + alpha * (temporal_error @ model.J.detach()) * derivative[:, :-1]
    )
    grad[:, :-1] -= propagated
    return grad


def relax(model, values, inputs, target, mask, temporal_weight: float):
    energy, temporal_error, output_error, r, output = energy_and_errors(
        model, values, inputs, target, mask, temporal_weight
    )
    for _ in range(model.cfg.pc_inference_steps):
        grad = value_gradient(model, values, inputs, temporal_error, output_error, mask, temporal_weight)
        step = model.cfg.pc_inference_lr
        while True:
            candidate = values - step * grad
            candidate[:, 0] = model.x0.detach().to(candidate.dtype)
            cand_energy, cand_temp, cand_out, cand_r, cand_output = energy_and_errors(
                model, candidate, inputs, target, mask, temporal_weight
            )
            if cand_energy <= energy + 1e-7 or step < 1e-8:
                values, energy = candidate, cand_energy
                temporal_error, output_error, r, output = cand_temp, cand_out, cand_r, cand_output
                break
            step *= 0.5
    return values, temporal_error, output_error, output


def rms(tensor: torch.Tensor, count: float | None = None) -> float:
    if count is None:
        count = tensor.numel()
    return float(torch.sqrt(tensor.square().sum() / max(count, 1)).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="e.g. results/runs/pc_steps100/seed_0004")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.run_dir / "config.yaml")
    if cfg.task_source == "neurogym":
        cfg.task_source = "standalone"

    model = build_model(cfg)
    checkpoint = torch.load(args.run_dir / "model_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rng = np.random.default_rng(cfg.seed + 999)
    batch = make_batch(cfg, cfg.batch, rng)
    inputs = torch.as_tensor(batch.inputs, dtype=torch.float32)
    target = torch.as_tensor(batch.target, dtype=torch.float32)
    mask = torch.as_tensor(batch.mask, dtype=torch.float32)
    mask_count = float(mask.sum().item())

    with torch.no_grad():
        init_values = model._raw_forward_values(inputs)
        forward_output, _ = model(inputs, noise=False, return_states=False)
        forward_loss = float(masked_mse(forward_output, target, mask))

        print(f"run_dir:                 {args.run_dir}")
        print(f"forward-only loss (vs target): {forward_loss:.6g}   <- unchanged by this diagnostic, reference point")
        print()
        header = (
            "temporal_weight".ljust(17)
            + "temporal_err RMS".ljust(18)
            + "output_err RMS".ljust(16)
            + "relaxed-vs-forward output gap"
        )
        print(header)

        for w in TEMPORAL_WEIGHTS:
            values, temporal_error, output_error, output = relax(
                model, init_values.clone(), inputs, target, mask, temporal_weight=w
            )
            relaxed_vs_forward = float(masked_mse(output, forward_output, mask))
            row = (
                f"{w:<17g}"
                + f"{rms(temporal_error):<18.6g}"
                + f"{rms(output_error, mask_count):<16.6g}"
                + f"{relaxed_vs_forward:.6g}"
            )
            print(row)


if __name__ == "__main__":
    main()
