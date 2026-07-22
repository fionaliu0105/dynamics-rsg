"""Richer PC activity diagnostics: PCA trajectories, unit activity, output-vs-target.

Companion to ``scripts/plot_bptt_activity.py``, but reads a real completed
``train_one_seed`` run instead of retraining a toy model inline (AGENTS.md,
"Plotting reads saved metrics ... never retrains"). Pulls ``metrics.json`` for the
training-loss curve and the per-condition :class:`~src.store.ActivationStore`
records (written once by the trainer after training completes) for the rest.

Usage::

    python scripts/plot_pc_activity.py --run-dir results/runs/pc/seed_0000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.conditions import CONDITIONS, Condition
from src.store import ActivationStore
from src.task.rsg import ramp
from src.training.config import Config
from src.viz.figures import (
    output_vs_target_figure,
    pca_trajectories_figure,
    training_loss_figure,
    unit_activity_figure,
)


def _reconstruct_target(cfg: Config, condition: Condition, set_step: int) -> np.ndarray:
    """Rebuild the target ramp for one condition, matching ``src.task.rsg._build``.

    The activation store saves the produced output but not the target trace, so
    this mirrors the trainer's own target construction (ramp-to-threshold from
    Set onset, held through ``prod_hold``) rather than re-deriving it differently.
    """
    ts_steps = int(round(condition.ts / cfg.dt))
    prod_end = min(set_step + ts_steps + cfg.prod_hold_step, cfg.n_steps)
    target = np.zeros(cfg.n_steps, dtype=np.float32)
    rel = np.arange(prod_end - set_step)
    target[set_step:prod_end] = ramp(rel, ts_steps, cfg)
    return target


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=str, default="results/runs/pc/seed_0000")
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path("results/figures/pc_activity")

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"no metrics.json at {metrics_path} — this run hasn't completed yet; "
            "these plots read a finished run's saved metrics, they don't retrain."
        )
    metrics = json.loads(metrics_path.read_text())
    cfg = Config.from_yaml(run_dir / "config.yaml")

    training_loss_figure(metrics["losses"], name="pc_training_loss", out_dir=out_dir)

    store = ActivationStore(run_dir / "activations")
    records = {cond: store.read(cfg.rule, cfg.seed, cond) for cond in CONDITIONS if store.has(cfg.rule, cfg.seed, cond)}
    if not records:
        raise FileNotFoundError(f"no activation records found under {run_dir / 'activations'}")

    labels = [cond.label for cond in records]
    states_by_condition = {cond.label: rec.states for cond, rec in records.items()}

    first_cond = next(iter(records))
    unit_activity_figure(
        records[first_cond].states, dt=cfg.dt, out_dir=out_dir,
        name="pc_unit_activity", title=f"PC unit activity: {first_cond.label}",
    )

    outputs = np.stack([rec.meta["outputs"] for rec in records.values()])
    targets = np.stack([
        _reconstruct_target(cfg, cond, rec.meta["set_step"]) for cond, rec in records.items()
    ])
    output_vs_target_figure(
        outputs, targets, dt=cfg.dt, labels=labels, threshold=cfg.threshold,
        name="pc_output_vs_target", out_dir=out_dir,
    )

    prior_color = {"short": np.array([0.85, 0.3, 0.3, 1.0]), "long": np.array([0.2, 0.4, 0.8, 1.0])}
    effector_ls = {"eye": "-", "hand": "--"}
    color_by = {cond.label: prior_color[cond.prior] for cond in records}
    linestyle_by = {cond.label: effector_ls[cond.effector] for cond in records}
    pca_trajectories_figure(
        states_by_condition, out_dir=out_dir,
        name="pc_pca_trajectories", color_by=color_by, linestyle_by=linestyle_by,
    )

    print(f"[plot_pc_activity] {len(records)} conditions; figures written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
