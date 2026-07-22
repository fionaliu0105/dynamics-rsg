"""Single entry point for training — interactive AND SLURM (plan 3.1).

ONE seed per invocation. A SLURM array task and a laptop run call this same script
with the same config; the batch wrapper only sets resources. torch is imported
lazily (inside training) so ``--dry-run`` works in the contracts-only env.

Examples::

    python scripts/train.py --config configs/bptt.yaml --seed 0
    python scripts/train.py --config configs/pc.yaml --seed $SLURM_ARRAY_TASK_ID
    python scripts/train.py --regime reduced --rule pc --seed 3 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.config import Config


def build_config(args) -> Config:
    if args.config:
        cfg = Config.from_yaml(args.config)
        overrides = {}
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.task_source is not None:
            overrides["task_source"] = args.task_source
        if args.pc_inference_steps is not None:
            overrides["pc_inference_steps"] = args.pc_inference_steps
        if overrides:
            cfg = Config.from_dict({**cfg.to_dict(), **overrides})
        return cfg
    make = Config.faithful if args.regime == "faithful" else Config.reduced
    overrides = {}
    if args.rule is not None:
        overrides["rule"] = args.rule
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.task_source is not None:
        overrides["task_source"] = args.task_source
    if args.pc_inference_steps is not None:
        overrides["pc_inference_steps"] = args.pc_inference_steps
    return make(**overrides)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train one RSG RNN seed (BPTT or PC).")
    p.add_argument("--config", type=str, help="path to a run config YAML")
    p.add_argument("--regime", choices=["reduced", "faithful"], default="reduced")
    p.add_argument("--rule", choices=["bptt", "pc"])
    p.add_argument("--seed", type=int)
    p.add_argument("--task-source", choices=["neurogym", "standalone"],
                   help="task data generator (default from config: neurogym)")
    p.add_argument("--pc-inference-steps", type=int, default=None,
                   help="override cfg.pc_inference_steps (PC value-relaxation steps; default from config: 20)")
    p.add_argument("--run-dir", type=str, default="results/runs")
    p.add_argument(
        "--activation-store",
        type=str,
        default=None,
        help="shared ActivationStore root (default: sibling 'activations' of --run-dir)",
    )
    p.add_argument("--dry-run", action="store_true", help="build+print config, don't train")
    args = p.parse_args(argv)

    cfg = build_config(args)
    run_dir = Path(args.run_dir) / cfg.rule / f"seed_{cfg.seed:04d}"
    print(f"[train] rule={cfg.rule} seed={cfg.seed} task_source={cfg.task_source} "
          f"regime dt={cfg.dt} N={cfg.N} pc_inference_steps={cfg.pc_inference_steps} -> {run_dir}")
    if args.dry_run:
        print("[train] --dry-run: config built OK, not training.")
        return 0

    from src.training.trainer import train_one_seed  # lazy: needs torch
    activation_store = (
        Path(args.activation_store)
        if args.activation_store
        else Path(args.run_dir).parent / "activations"
    )
    train_one_seed(cfg, run_dir, activation_store_root=activation_store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
