"""Read-only diagnostic: how fast do nearby trajectories diverge, as a function of cfg.g?

Fresh, untrained networks only -- no checkpoint, no training, nothing written to
results/. For each candidate g, build a model, perturb its initial state x0 by a
tiny fixed amount, run the same deterministic forward recursion (no process
noise) from both the unperturbed and perturbed start over one real trial's
inputs, and track how the gap between the two trajectories grows or shrinks
over time. This tests the "g=1 lets small nudges compound" hypothesis from
`.suplex/docs/discrepancy_log.md` directly, independent of the relaxation
machinery.

Uses BPTTRNN purely for its forward() (identical equations to PCRNN.forward());
the object under test is the shared architecture's dynamics, not either
learning rule.
"""

from __future__ import annotations

import numpy as np
import torch

from src.models.bptt_rnn import BPTTRNN
from src.task import make_batch
from src.training.config import Config

G_VALUES = [0.7, 0.85, 1.0, 1.15, 1.3]
EPS = 1e-4
CHECKPOINTS_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def run_one(g: float, direction: torch.Tensor, inputs: torch.Tensor) -> list[float]:
    cfg = Config.reduced(rule="bptt", seed=0, g=g)
    if cfg.task_source == "neurogym":
        cfg.task_source = "standalone"
    model = BPTTRNN(cfg)
    model.eval()

    alpha = cfg.alpha
    J, B, c_x = model.J.detach(), model.B.detach(), model.c_x.detach()
    x0 = model.x0.detach()

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
    return divergences


def main() -> None:
    base_cfg = Config.reduced(rule="bptt", seed=0)
    if base_cfg.task_source == "neurogym":
        base_cfg.task_source = "standalone"
    rng = np.random.default_rng(0)
    batch = make_batch(base_cfg, 1, rng)
    inputs = torch.as_tensor(batch.inputs, dtype=torch.float32)

    dir_rng = np.random.default_rng(12345)  # fixed across all g, isolates g's effect
    direction = torch.as_tensor(dir_rng.standard_normal(base_cfg.N), dtype=torch.float32)
    direction = direction / direction.norm()

    print(f"perturbation size (EPS):        {EPS}")
    print(f"checkpoint fractions of trial:  {CHECKPOINTS_FRACTIONS}")
    print()
    header = "g".ljust(6) + "".join(f"t={f:.2f}".ljust(14) for f in CHECKPOINTS_FRACTIONS) + "amplification"
    print(header)
    for g in G_VALUES:
        divergences = run_one(g, direction, inputs)
        row = f"{g:<6}" + "".join(f"{d:.3g}".ljust(14) for d in divergences)
        amplification = divergences[-1] / EPS
        row += f"{amplification:.3g}x"
        print(row)


if __name__ == "__main__":
    main()
