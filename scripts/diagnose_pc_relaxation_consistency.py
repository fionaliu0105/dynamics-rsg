"""Read-only diagnostic: does relaxation trade dynamical consistency for target-matching?

Loads an existing PC checkpoint, builds one training-shaped batch, and compares
the raw forward-sweep initialization (temporal_error ~ 0 by construction, output
untouched) against the fully relaxed trajectory (`_relax`, what `infer_and_update`
actually scores). Reports per-element RMS of `temporal_error` and `output_error`
at both points, so a shrinking output_error paired with a *growing* temporal_error
is directly visible.

No training happens; no checkpoint, config, or code path is modified. See
``.suplex/handoffs/active/current_handoff.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.task import make_batch
from src.training.config import Config
from src.training.trainer import build_model


def _rms(tensor: torch.Tensor, count: float | None = None) -> float:
    if count is None:
        count = tensor.numel()
    return float(torch.sqrt(tensor.square().sum() / max(count, 1)).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="e.g. results/runs/pc_steps100/seed_0004")
    parser.add_argument("--batch", type=int, default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.run_dir / "config.yaml")
    if cfg.rule != "pc":
        raise ValueError(f"{args.run_dir} is rule={cfg.rule!r}, this diagnostic is PC-only")
    if cfg.task_source == "neurogym":
        cfg.task_source = "standalone"  # byte-identical backend; neurogym not installed locally

    model = build_model(cfg)
    checkpoint = torch.load(args.run_dir / "model_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rng = np.random.default_rng(cfg.seed + 999)
    batch = make_batch(cfg, args.batch or cfg.batch, rng)
    inputs = torch.as_tensor(batch.inputs, dtype=torch.float32)
    target = torch.as_tensor(batch.target, dtype=torch.float32)
    mask = torch.as_tensor(batch.mask, dtype=torch.float32)
    mask_count = float(mask.sum().item())

    with torch.no_grad():
        # Before relaxation: the plain forward-sweep initialization.
        init_values = model._raw_forward_values(inputs)
        init_energy, init_temporal, init_output, _, _ = model._energy_and_errors(
            init_values, inputs, target, mask
        )

        # After relaxation: what infer_and_update() actually scores.
        _, final_energy, final_temporal, final_output, _, _, trace = model._relax(
            init_values, inputs, target, mask
        )

    print(f"run_dir:                       {args.run_dir}")
    print(f"relaxation steps:              {cfg.pc_inference_steps}")
    print(f"energy trace (first->last):    {trace[0]:.4g} -> {trace[-1]:.4g}")
    print()
    print("                                temporal_error RMS   output_error RMS (masked)")
    print(f"before relaxation (forward init): {_rms(init_temporal):.6g}          {_rms(init_output, mask_count):.6g}")
    print(f"after relaxation (scored loss):   {_rms(final_temporal):.6g}          {_rms(final_output, mask_count):.6g}")
    print()
    temporal_ratio = _rms(final_temporal) / max(_rms(init_temporal), 1e-12)
    output_ratio = _rms(final_output, mask_count) / max(_rms(init_output, mask_count), 1e-12)
    print(f"temporal_error RMS ratio (after/before): {temporal_ratio:.3g}x")
    print(f"output_error RMS ratio (after/before):   {output_ratio:.3g}x")


if __name__ == "__main__":
    main()
