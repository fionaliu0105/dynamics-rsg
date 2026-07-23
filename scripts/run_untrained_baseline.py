"""Untrained-RNN control: how much of the DMFC match is training, and how much is architecture?

Every distance-to-DMFC number in this project is uninterpretable without this arm.
Random RNNs with the right architecture and the right input drive routinely land
surprisingly close to neural data, because a good chunk of the geometry comes from
the task inputs and the recurrent timescale rather than from anything learned. If an
untrained net is about as close to DMFC as the trained ones, then the between-rule
differences are decoration on an architecture effect, and the honest headline changes.

This writes an ``untrained`` arm into an activation store using the SAME rollout and
the SAME extraction path as the trained runs (``trainer.store_condition_activations``),
so nothing about the comparison differs except that no gradient step was ever taken.
Seed *N* here is the same initialization seed *N* the trained arms started from, which
keeps the paired-by-seed structure intact.

    python scripts/run_untrained_baseline.py --out-store results/activations_untrained \\
        --seeds 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.store import ActivationStore
from src.training.config import Config
from src.training.trainer import build_model, store_condition_activations


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="results/runs/bptt/seed_0000/config.yaml",
                   help="config to inherit architecture from; only rule/seed are overridden")
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--out-store", type=str, default="results/activations_untrained")
    p.add_argument("--model-name", type=str, default="untrained",
                   help="name this arm gets inside the store")
    args = p.parse_args(argv)

    store = ActivationStore(args.out_store)
    device = torch.device("cpu")

    for seed in args.seeds:
        cfg = Config.from_yaml(Path(args.config))
        # Same architecture, same task constants, same seed -> the exact initialization
        # the trained arms departed from (BPTTRNN seeds its weights from cfg.seed, and
        # it is the shared forward model all three rules reuse).
        cfg.rule, cfg.seed = "bptt", seed
        model = build_model(cfg).to(device)
        cfg.rule = args.model_name          # store this arm under the control's own name
        store_condition_activations(model, cfg, store, device)
        print(f"[untrained] seed {seed}: wrote 20 conditions (no training performed)")

    print(f"[untrained] store ready at {args.out_store}/{args.model_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
