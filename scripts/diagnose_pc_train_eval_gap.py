"""Read-only diagnostic: compare PC's reported training loss to a forward-only loss.

Loads an existing ``model_best.pt`` for a PC seed and, on a fresh training-shaped
batch, computes two numbers:

1. ``infer_and_update`` loss (apply_update=False) -- the quantity trainer.py
   actually logs as the seed's loss during training.
2. ``masked_mse`` on plain ``forward(noise=False)`` -- the same computation
   BPTT's loss uses, and the same forward path eval/behavior uses.

No training happens and no checkpoint or config file is modified. See
``.suplex/handoffs/active/current_handoff.md`` step 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.task import make_batch
from src.training.config import Config
from src.training.trainer import build_model, masked_mse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="e.g. results/runs/pc_steps100/seed_0004")
    parser.add_argument("--batch", type=int, default=None, help="override cfg.batch for the probe")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.run_dir / "config.yaml")
    if cfg.rule != "pc":
        raise ValueError(f"{args.run_dir} is rule={cfg.rule!r}, this diagnostic is PC-only")
    if cfg.task_source == "neurogym":
        # neurogym isn't installed in this local env; the two backends are
        # byte-identical (tests/test_task_neurogym.py), so this only changes
        # which code generates an equivalent batch, not what the batch is.
        cfg.task_source = "standalone"

    model = build_model(cfg)
    checkpoint = torch.load(args.run_dir / "model_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rng = np.random.default_rng(cfg.seed + 999)  # different draw than any training iteration
    batch_size = args.batch or cfg.batch
    batch = make_batch(cfg, batch_size, rng)
    inputs = torch.as_tensor(batch.inputs, dtype=torch.float32)
    target = torch.as_tensor(batch.target, dtype=torch.float32)
    mask = torch.as_tensor(batch.mask, dtype=torch.float32)

    with torch.no_grad():
        diagnostics = model.infer_and_update(inputs, target, mask, apply_update=False)
        reported_loss = float(diagnostics["loss"])

        outputs, _ = model(inputs, noise=False, return_states=False)
        forward_loss = float(masked_mse(outputs, target, mask))

    print(f"run_dir:                 {args.run_dir}")
    print(f"checkpoint best_loss:     {checkpoint.get('best_loss')}")
    print(f"reported (relaxed) loss:  {reported_loss:.6g}   <- what trainer.py logs during training")
    print(f"forward-only loss:        {forward_loss:.6g}   <- same path BPTT uses / eval uses")
    if reported_loss > 0:
        print(f"ratio forward/reported:   {forward_loss / reported_loss:.3g}x")


if __name__ == "__main__":
    main()
