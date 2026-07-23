"""End-to-end tests for the shared NeuroGym trainer."""

from __future__ import annotations

import json

import pytest
import torch

pytest.importorskip("neurogym")

from src.conditions import N_CONDITIONS
from src.store import ActivationStore
from src.training.config import Config
from src.training.trainer import train_one_seed


def _smoke_config(rule: str, seed: int) -> Config:
    return Config.reduced(
        rule=rule,
        task_source="neurogym",
        seed=seed,
        N=8,
        dt=10.0,
        n_iter=2 if rule == "bptt" else 1,
        batch=2,
        noise_sd=0.0,
        pc_inference_steps=2,
    )


@pytest.mark.parametrize(("rule", "seed"), [("bptt", 4), ("pc", 5), ("rflo", 6)])
def test_neurogym_training_end_to_end(tmp_path, rule, seed):
    cfg = _smoke_config(rule, seed)
    run_dir = tmp_path / "runs" / rule / f"seed_{seed:04d}"
    store_root = tmp_path / "activations"
    train_one_seed(
        cfg,
        run_dir,
        activation_store_root=store_root,
        checkpoint_every=1,
    )

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["identity"]["task_source"] == "neurogym"
    assert len(metrics["losses"]) == cfg.n_iter
    assert (run_dir / "checkpoint_latest.pt").exists()
    assert (run_dir / "model_best.pt").exists()
    assert len(list(ActivationStore(store_root).keys())) == N_CONDITIONS

    # Requeuing the same completed seed does not train it again.
    train_one_seed(cfg, run_dir, activation_store_root=store_root)
    assert json.loads((run_dir / "metrics.json").read_text()) == metrics


def test_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    cfg = _smoke_config("bptt", 8)
    cfg.n_iter = 3
    store_root = tmp_path / "activations"

    import src.task

    original_make_batch = src.task.make_batch
    calls = 0

    def interrupt_after_first_batch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated preemption")
        return original_make_batch(*args, **kwargs)

    interrupted = tmp_path / "interrupted" / "bptt" / "seed_0008"
    monkeypatch.setattr(src.task, "make_batch", interrupt_after_first_batch)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        train_one_seed(
            cfg,
            interrupted,
            activation_store_root=store_root,
            checkpoint_every=1,
        )
    monkeypatch.setattr(src.task, "make_batch", original_make_batch)
    train_one_seed(
        cfg,
        interrupted,
        activation_store_root=store_root,
        checkpoint_every=1,
    )

    uninterrupted = tmp_path / "uninterrupted" / "bptt" / "seed_0008"
    train_one_seed(
        cfg,
        uninterrupted,
        activation_store_root=tmp_path / "other_activations",
        checkpoint_every=1,
    )

    resumed_metrics = json.loads((interrupted / "metrics.json").read_text())
    full_metrics = json.loads((uninterrupted / "metrics.json").read_text())
    assert resumed_metrics["losses"] == full_metrics["losses"]
    resumed = torch.load(interrupted / "model_best.pt", weights_only=False)
    full = torch.load(uninterrupted / "model_best.pt", weights_only=False)
    for name in resumed["model_state"]:
        assert torch.equal(resumed["model_state"][name], full["model_state"][name])


def test_training_entry_point_routes_to_shared_store(tmp_path):
    from scripts.train import main

    cfg = _smoke_config("bptt", 11)
    cfg.n_iter = 1
    config_path = tmp_path / "smoke.yaml"
    cfg.to_yaml(config_path)
    run_root = tmp_path / "runs"
    store_root = tmp_path / "shared_activations"

    assert main([
        "--config", str(config_path),
        "--run-dir", str(run_root),
        "--activation-store", str(store_root),
    ]) == 0
    assert len(list(ActivationStore(store_root).keys())) == N_CONDITIONS
    assert (run_root / "bptt" / "seed_0011" / "completed.json").exists()
