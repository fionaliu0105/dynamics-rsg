"""Read-only diagnostic: is the *trained* network's own forward() sensitive to a tiny nudge?

Unlike `diagnose_g_sensitivity.py` (fresh, untrained, random networks across a
g sweep, which found no meaningful sensitivity), this loads an existing trained
checkpoint's actual J/B/c_x/x0 and tests the same thing: perturb x0 by a tiny
amount, run the deterministic forward() recursion from both the unperturbed and
perturbed start over a real trial, and see how the gap evolves.

Strictly read-only: opens `config.yaml` and `model_best.pt` with torch.load,
never writes to either, and writes nothing anywhere else (no results/, no
checkpoint, no config file is touched or modified).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.task import make_batch
from src.training.config import Config
from src.training.trainer import build_model

EPS = 1e-4
CHECKPOINTS_FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="e.g. results/runs/pc_steps100/seed_0004")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.run_dir / "config.yaml")
    if cfg.task_source == "neurogym":
        cfg.task_source = "standalone"  # byte-identical backend; neurogym not installed locally

    model = build_model(cfg)
    checkpoint = torch.load(args.run_dir / "model_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])  # read-only load; nothing saved back
    model.eval()

    alpha = cfg.alpha
    J, B, c_x = model.J.detach(), model.B.detach(), model.c_x.detach()
    x0 = model.x0.detach()

    rng = np.random.default_rng(cfg.seed + 999)
    batch = make_batch(cfg, 1, rng)
    inputs = torch.as_tensor(batch.inputs, dtype=torch.float32)

    dir_rng = np.random.default_rng(12345)  # same fixed direction convention as the g-sweep scripts
    direction = torch.as_tensor(dir_rng.standard_normal(cfg.N), dtype=torch.float32)
    direction = direction / direction.norm()

    x_ref = x0.clone().unsqueeze(0)
    x_pert = (x0 + EPS * direction).clone().unsqueeze(0)

    time = inputs.shape[1]
    checkpoints = [int(round(f * (time - 1))) for f in CHECKPOINTS_FRACTIONS]
    divergences = []
    with torch.no_grad():
        for t in range(time):
            if t in checkpoints:
                divergences.append(float((x_pert - x_ref).norm().item()))
            u = inputs[:, t, :]
            r_ref, r_pert = torch.tanh(x_ref), torch.tanh(x_pert)
            dx_ref = -x_ref + r_ref @ J.t() + u @ B.t() + c_x
            dx_pert = -x_pert + r_pert @ J.t() + u @ B.t() + c_x
            x_ref = x_ref + alpha * dx_ref
            x_pert = x_pert + alpha * dx_pert

    print(f"run_dir:                        {args.run_dir}")
    print(f"perturbation size (EPS):        {EPS}")
    print(f"checkpoint fractions of trial:  {CHECKPOINTS_FRACTIONS}")
    print()
    header = "".join(f"t={f:.2f}".ljust(14) for f in CHECKPOINTS_FRACTIONS) + "amplification"
    print(header)
    row = "".join(f"{d:.3g}".ljust(14) for d in divergences)
    row += f"{divergences[-1] / EPS:.3g}x"
    print(row)


if __name__ == "__main__":
    main()
