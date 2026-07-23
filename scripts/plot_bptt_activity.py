"""Sanity-check plots of BPTTRNN activity: train briefly, look at what it does.

The real task generator (``src/task/rsg.py``) is a DIFFERENT track's module (Task &
behavior, plan 1.A, owned by another team member per docs/implementation_plan.md
"Team & division of labor") — this script does not touch it. It builds its own
trivial multi-condition batch instead, the same pattern ``tests/test_bptt.py``
uses, generalized over ``src.conditions.Condition`` so trials are labeled by real
(prior, ts, effector) triples. No prior-mean jitter (that Bayesian-bias mechanism
belongs to the task track); this is only meant to exercise the trained network's
dynamics and produce a first look at its activity, within the BPTT track's own
scope (plan 1.B: "implementation -> its test -> a small figure").

This is a diagnostic script, not the pipeline: swap in ``src.task.rsg.make_batch``
and ``src.training.trainer.train_one_seed`` once those land, and read saved
checkpoints instead of training inline (AGENTS.md, "Plotting reads saved metrics").

Usage::

    python scripts/plot_bptt_activity.py --conditions demo --n-iter 1500
    python scripts/plot_bptt_activity.py --conditions all --n-iter 600   # 20-condition pilot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.conditions import CONDITIONS, Condition
from src.training.config import Config


def make_demo_batch(cfg: Config, conditions: list[Condition], rng: torch.Generator):
    """Ready/Set pulses + ramp target for a handful of real Conditions. No jitter."""
    n = len(conditions)
    inputs = torch.zeros(n, cfg.n_steps, 3)
    target = torch.zeros(n, cfg.n_steps)
    mask = torch.zeros(n, cfg.n_steps)

    for i, cond in enumerate(conditions):
        ready_step = cfg.ready_onset_step
        ts_step = cfg.to_step(cond.ts)
        set_step = ready_step + ts_step
        pw = cfg.pulse_width_step

        inputs[i, ready_step:ready_step + pw, 0] = cfg.pulse_height
        inputs[i, set_step:set_step + pw, 0] = cfg.pulse_height
        inputs[i, :, 1] = cfg.prior_context[cond.prior]
        inputs[i, :, 2] = cfg.effector_context[cond.effector]

        prod_end = min(set_step + ts_step, cfg.n_steps)
        ramp_len = max(prod_end - set_step, 1)
        target[i, set_step:prod_end] = torch.linspace(0.0, cfg.threshold, ramp_len)
        hold_end = min(prod_end + cfg.prod_hold_step, cfg.n_steps)
        target[i, prod_end:hold_end] = cfg.threshold
        mask[i, set_step:hold_end] = 1.0

    return inputs, target, mask


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--conditions", choices=["demo", "all"], default="demo",
        help="'demo' = 6 short/long-eye conditions; 'all' = all 20 real CONDITIONS "
             "(both priors x both effectors), a small pilot pass, not full training.",
    )
    p.add_argument("--n-iter", type=int, default=None, help="default: 1500 (demo) / 600 (all)")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args(argv)

    from src.models.bptt_rnn import BPTTRNN
    from src.training.trainer import set_seeds
    from src.viz.figures import (
        output_vs_target_figure,
        pca_trajectories_figure,
        training_loss_figure,
        unit_activity_figure,
    )

    all_conditions = args.conditions == "all"
    n_iter = args.n_iter if args.n_iter is not None else (600 if all_conditions else 1500)
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        "results/figures/bptt_pilot_all" if all_conditions else "results/figures/bptt_activity_demo"
    )

    # total_time covers the longest condition (long/1200ms) + production hold;
    # 1300ms is enough for the demo's short-prior-only subset.
    total_time = 1700.0 if all_conditions else 1300.0
    cfg = Config.reduced(
        rule="bptt", seed=args.seed, n_iter=n_iter, lr=args.lr,
        grad_clip=args.grad_clip, total_time=total_time,
    )
    set_seeds(cfg.seed)
    if all_conditions:
        conditions = list(CONDITIONS)  # all 20: prior x ts x effector
    else:
        conditions = [
            Condition(prior="short", ts=ts, effector="eye")
            for ts in (480, 640, 800)
        ] + [
            Condition(prior="long", ts=ts, effector="eye")
            for ts in (800, 1000, 1200)
            if cfg.ready_onset + ts + cfg.prod_hold <= cfg.total_time
        ]
    labels = [c.label for c in conditions]

    model = BPTTRNN(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    inputs, target, mask = make_demo_batch(cfg, conditions, torch.Generator().manual_seed(0))

    losses = []
    best_loss, best_state = float("inf"), None
    for _ in range(cfg.n_iter):
        opt.zero_grad()
        outputs, _ = model(inputs, noise=True)
        loss = ((outputs - target) ** 2 * mask).sum() / mask.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        losses.append(loss.item())
        if loss.item() < best_loss:
            best_loss, best_state = loss.item(), {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"[plot_bptt_activity] {len(conditions)} conditions, {cfg.n_iter} iters: "
          f"loss {losses[0]:.4f} -> best {best_loss:.4f}")

    with torch.no_grad():
        outputs, states = model(inputs, noise=False)
    outputs_np, states_np = outputs.numpy(), states.numpy()

    training_loss_figure(losses, out_dir=out_dir)
    unit_activity_figure(
        states_np[labels.index(f"{conditions[0].prior}/{conditions[0].ts}ms/{conditions[0].effector}")],
        dt=cfg.dt, out_dir=out_dir, condition_label=labels[0],
    )
    output_vs_target_figure(
        outputs_np, target.numpy(), dt=cfg.dt, labels=labels,
        threshold=cfg.threshold, out_dir=out_dir,
    )
    states_by_condition = {lab: states_np[i] for i, lab in enumerate(labels)}
    prior_color = {"short": np.array([0.85, 0.3, 0.3, 1.0]), "long": np.array([0.2, 0.4, 0.8, 1.0])}
    effector_ls = {"eye": "-", "hand": "--"}
    color_by = {lab: prior_color[c.prior] for lab, c in zip(labels, conditions)}
    linestyle_by = {lab: effector_ls[c.effector] for lab, c in zip(labels, conditions)}
    pca_trajectories_figure(states_by_condition, out_dir=out_dir, color_by=color_by, linestyle_by=linestyle_by)

    print(f"[plot_bptt_activity] figures written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
