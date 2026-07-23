"""Read-only diagnostic: confirm the point-attractor hypothesis directly.

For a trained PC checkpoint, runs the real `forward()` (no relaxation, no
target, `noise=False`) on all 20 real eval conditions, and checks two things
per (prior, effector) group:

1. Does the state stop changing well before the trial ends (a genuine fixed
   point), rather than just being slow-moving?
2. Do different `ts` values in the same group converge to the *same* state
   and the *same* output, rather than merely similar ones?

If both hold, that confirms the point-attractor explanation for the flat
`tp_vs_ts` plateaus directly, rather than leaving it as an inference from the
perturbation tests. Strictly read-only: loads `config.yaml`/`model_best.pt`,
writes nothing anywhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.conditions import CONDITIONS
from src.task import build_trial
from src.training.config import Config
from src.training.trainer import build_model


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

    print(f"run_dir: {args.run_dir}\n")

    by_group: dict[tuple[str, str], list[tuple[int, torch.Tensor, torch.Tensor]]] = {}
    with torch.no_grad():
        for condition in CONDITIONS:
            inputs_np, set_step = build_trial(cfg, condition, jitter=False)
            inputs = torch.as_tensor(inputs_np, dtype=torch.float32)  # already [1, time, 3]
            outputs, states = model(inputs, noise=False)

            state_final = states[0, -1]
            state_10_before = states[0, -10]
            delta = (state_final - state_10_before).norm().item()
            output_final = float(outputs[0, -1].item())

            group = (condition.prior, condition.effector)
            by_group.setdefault(group, []).append((condition.ts, state_final, output_final))

            print(
                f"{condition.label:>20s}  |state[-1]-state[-10]|={delta:.3g}   "
                f"output[-1]={output_final:.6f}"
            )

    print("\n--- within-group agreement (does ts matter once settled?) ---")
    for group, entries in by_group.items():
        entries.sort(key=lambda e: e[0])
        outputs_only = [e[2] for e in entries]
        states_only = torch.stack([e[1] for e in entries])
        max_pairwise_state_diff = 0.0
        n = states_only.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                d = (states_only[i] - states_only[j]).norm().item()
                max_pairwise_state_diff = max(max_pairwise_state_diff, d)
        print(
            f"{group}: output range across ts = [{min(outputs_only):.6f}, {max(outputs_only):.6f}]"
            f"   max pairwise final-state distance across ts = {max_pairwise_state_diff:.3g}"
        )


if __name__ == "__main__":
    main()
