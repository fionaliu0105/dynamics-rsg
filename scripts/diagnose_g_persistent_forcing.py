"""Read-only diagnostic: does a small, steadily-repeated forcing settle far from

the unforced trajectory -- distinct from `diagnose_g_sensitivity.py`, which tested
a single one-off perturbation and found it decays regardless of g. This instead
re-applies a small, fixed-direction forcing vector at EVERY timestep (closer to
what relaxation actually does: `temporal_error` is nonzero and roughly consistent
in direction at each of the ~600 steps, not injected once). Fresh, untrained
networks only, no training, no checkpoint, nothing written to results/.
"""

from __future__ import annotations

import numpy as np
import torch

from src.models.bptt_rnn import BPTTRNN
from src.task import make_batch
from src.training.config import Config

G_VALUES = [0.7, 0.85, 1.0, 1.15, 1.3]
EPS = 1e-4  # same per-step forcing size as the single-shot perturbation test
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
    x_forced = x0.clone().unsqueeze(0)

    time = inputs.shape[1]
    checkpoints = [int(round(f * (time - 1))) for f in CHECKPOINTS_FRACTIONS]
    divergences = []
    with torch.no_grad():
        for t in range(time):
            if t in checkpoints:
                divergences.append(float((x_forced - x_ref).norm().item()))
            u = inputs[:, t, :]
            r_ref, r_forced = torch.tanh(x_ref), torch.tanh(x_forced)
            dx_ref = -x_ref + r_ref @ J.t() + u @ B.t() + c_x
            dx_forced = -x_forced + r_forced @ J.t() + u @ B.t() + c_x
            x_ref = x_ref + alpha * dx_ref
            # Forcing re-applied every step, same direction each time -- unlike the
            # one-off perturbation test, this never gets a chance to just decay away.
            x_forced = x_forced + alpha * dx_forced + EPS * direction
    return divergences


def main() -> None:
    base_cfg = Config.reduced(rule="bptt", seed=0)
    if base_cfg.task_source == "neurogym":
        base_cfg.task_source = "standalone"
    rng = np.random.default_rng(0)
    batch = make_batch(base_cfg, 1, rng)
    inputs = torch.as_tensor(batch.inputs, dtype=torch.float32)

    dir_rng = np.random.default_rng(12345)  # same direction/seed as the prior sensitivity test
    direction = torch.as_tensor(dir_rng.standard_normal(base_cfg.N), dtype=torch.float32)
    direction = direction / direction.norm()

    n_steps = inputs.shape[1]
    naive_linear_steady_state = EPS / base_cfg.alpha  # pure-leak baseline, ignores J entirely

    print(f"per-step forcing size (EPS):    {EPS}")
    print(f"trial length (n_steps):         {n_steps}")
    print(f"naive pure-leak steady state:    {naive_linear_steady_state:.3g}  (baseline if J contributed nothing)")
    print(f"checkpoint fractions of trial:  {CHECKPOINTS_FRACTIONS}")
    print()
    header = "g".ljust(6) + "".join(f"t={f:.2f}".ljust(14) for f in CHECKPOINTS_FRACTIONS) + "final/naive-baseline"
    print(header)
    for g in G_VALUES:
        divergences = run_one(g, direction, inputs)
        row = f"{g:<6}" + "".join(f"{d:.3g}".ljust(14) for d in divergences)
        row += f"{divergences[-1] / naive_linear_steady_state:.3g}x"
        print(row)


if __name__ == "__main__":
    main()
