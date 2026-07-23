"""Checks for the PC-B task/trainer/behavior plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np

from src.behavior.slope import slopes_by_prior, tp
from src.conditions import Condition
from src.task.rsg import build_trial, make_batch
from src.training.config import Config
from src.training.trainer import train_one_seed


def _tiny_cfg(rule="bptt") -> Config:
    return Config.reduced(
        rule=rule,
        seed=4,
        N=8,
        n_iter=2,
        batch=3,
        total_time=260.0,
        ready_onset=20.0,
        pulse_width=10.0,
        prod_hold=20.0,
        lr=1e-3,
        pc_inference_steps=2,
        pc_inference_lr=0.05,
    )


def test_make_batch_and_build_trial_shapes_and_context():
    cfg = Config.reduced(seed=1, batch=5)
    rng = np.random.default_rng(12)
    batch = make_batch(cfg, 5, rng)
    assert batch.inputs.shape == (5, cfg.n_steps, 3)
    assert batch.target.shape == (5, cfg.n_steps)
    assert batch.mask.shape == (5, cfg.n_steps)
    assert len(batch.conditions) == 5
    assert np.isfinite(batch.inputs).all()
    assert np.isfinite(batch.target).all()
    assert np.isfinite(batch.mask).all()
    for row, condition in enumerate(batch.conditions):
        assert np.allclose(batch.inputs[row, :, 1], cfg.prior_context[condition.prior])
        assert np.allclose(batch.inputs[row, :, 2], cfg.effector_context[condition.effector])
        assert batch.mask[row].sum() > 0

    condition = Condition("short", 640, "hand")
    trial, set_step = build_trial(cfg, condition)
    assert trial.shape == (1, cfg.n_steps, 3)
    assert set_step == cfg.ready_onset_step + cfg.to_step(condition.ts)
    assert np.allclose(trial[0, :, 1], cfg.prior_context["short"])
    assert np.allclose(trial[0, :, 2], cfg.effector_context["hand"])


def test_tp_and_slopes_by_prior():
    cfg = Config.reduced()
    outputs = np.zeros(cfg.n_steps, dtype=float)
    set_step = 30
    outputs[set_step + cfg.pulse_width_step + 7] = cfg.threshold
    assert tp(outputs, set_step, cfg) == (cfg.pulse_width_step + 7) * cfg.dt

    slopes = slopes_by_prior(
        [480, 560, 640, 800, 900, 1000],
        [500, 570, 650, np.nan, 940, 1040],
        ["short", "short", "short", "long", "long", "long"],
    )
    assert np.isfinite(slopes["short"])
    assert np.isnan(slopes["long"])


def test_train_one_seed_tiny_bptt(tmp_path):
    run_dir = tmp_path / "bptt"
    train_one_seed(_tiny_cfg("bptt"), run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert np.isfinite(metrics["losses"]).all()
    assert len(metrics["losses"]) == 2
    assert len(metrics["behavior_by_condition"]) == 20
    assert (run_dir / "activations" / "bptt" / "seed_0004").exists()


def test_train_one_seed_tiny_pc(tmp_path):
    run_dir = tmp_path / "pc"
    train_one_seed(_tiny_cfg("pc"), run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert np.isfinite(metrics["losses"]).all()
    assert len(metrics["losses"]) == 2
    assert len(metrics["behavior_by_condition"]) == 20
    assert (run_dir / "activations" / "pc" / "seed_0004").exists()


def test_train_one_seed_tiny_rflo(tmp_path):
    run_dir = tmp_path / "rflo"
    train_one_seed(_tiny_cfg("rflo"), run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert np.isfinite(metrics["losses"]).all()
    assert len(metrics["losses"]) == 2
    assert len(metrics["behavior_by_condition"]) == 20
    assert (run_dir / "activations" / "rflo" / "seed_0004").exists()
