"""Shared, restart-safe trainer for one RSG network seed.

The task, checkpointing, metrics, and activation export paths are identical across all
three learning rules, and so is the optimizer (Adam).  Only the *direction* differs:
BPTT gets it from autograd, while the local rules in :data:`LOCAL_RULES` compute it
themselves -- ``PCRNN.infer_and_update`` from relaxed prediction errors, and
``RFLORNN.infer_and_update`` from online eligibility traces and random feedback.
Holding the optimizer fixed is what lets a rule-vs-rule difference be attributed to
the learning rule.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.conditions import CONDITIONS
from src.store import ActivationStore, Record
from src.task import active_backend
from src.training.config import Config

log = logging.getLogger(__name__)

#: Rules that compute their own update direction instead of getting it from autograd.
#: They share one contract -- ``model.infer_and_update(inputs, target, mask,
#: apply_update=...) -> {"loss": float, "updates": {param_name: tensor}}`` -- so the
#: training loop treats them identically and only BPTT needs a separate branch.
LOCAL_RULES = frozenset({"pc", "rflo"})


def _local_optimizer_choice(cfg: Config) -> str:
    """Which step-size policy a local rule uses: ``"adam"`` (default) or ``"sgd"``."""
    return cfg.pc_optimizer if cfg.rule == "pc" else cfg.rflo_optimizer


def set_seeds(seed: int) -> None:
    """Seed torch and numpy. Some CUDA kernels may remain nondeterministic."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_identity(cfg: Config) -> dict[str, Any]:
    """Return the provenance fields recorded with every run."""
    return {
        "rule": cfg.rule,
        "seed": cfg.seed,
        "task_source": active_backend(cfg),
        "host": socket.gethostname(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "git_sha": os.environ.get("GIT_SHA") or _git_sha(),
    }


def _git_sha() -> str | None:
    """Resolve the checkout revision without relying on the caller's cwd."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_model(cfg: Config):
    """Construct one of the architecture-matched models."""
    if cfg.rule == "bptt":
        from src.models.bptt_rnn import BPTTRNN

        return BPTTRNN(cfg)
    if cfg.rule == "pc":
        from src.models.pc_rnn import PCRNN

        return PCRNN(cfg)
    if cfg.rule == "rflo":
        from src.models.rflo_rnn import RFLORNN

        return RFLORNN(cfg)
    raise ValueError(f"unknown rule {cfg.rule!r}")


def masked_mse(outputs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over supervised samples only."""
    return ((outputs - target).square() * mask).sum() / mask.sum().clamp_min(1)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def _read_or_init_run_meta(run_dir: Path) -> dict[str, Any]:
    """Return this run's persisted metadata, creating it on the first invocation.

    ``started_at`` is written once and never overwritten, so it survives resumes —
    it answers "when did this run first start," not "when did this process start"
    (that's ``progress.json``'s ``elapsed_sec``, which is per-invocation).
    """
    path = run_dir / "run_meta.json"
    if path.exists():
        return json.loads(path.read_text())
    meta = {"started_at": datetime.now().astimezone().isoformat()}
    _atomic_json(meta, path)
    return meta


def _checkpoint_payload(
    model,
    optimizer,
    iteration: int,
    losses: list[float],
    best_loss: float,
    best_state: dict[str, torch.Tensor],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Capture enough state for an exact continuation after preemption."""
    return {
        "iteration": iteration,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "losses": losses,
        "best_loss": best_loss,
        "best_model_state": best_state,
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_checkpoint(path: Path, model, optimizer, rng, device):
    """Restore model, optimizer, histories, and random-number generators."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint["optimizer_state"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    rng.bit_generator.state = checkpoint["numpy_rng_state"]
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return (
        int(checkpoint["iteration"]) + 1,
        [float(value) for value in checkpoint["losses"]],
        float(checkpoint["best_loss"]),
        checkpoint["best_model_state"],
    )


def _validate_or_write_config(cfg: Config, path: Path) -> None:
    """Prevent a completed seed directory from silently changing meaning."""
    if path.exists():
        existing = Config.from_yaml(path)
        if existing.to_dict() != cfg.to_dict():
            raise ValueError(
                f"run directory already contains a different config: {path}. "
                "Choose another run directory or use the original config."
            )
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        cfg.to_yaml(tmp)
        tmp.replace(path)


def store_condition_activations(
    model,
    cfg: Config,
    store: ActivationStore,
    device: torch.device,
) -> tuple[dict[str, float | None], dict[str, dict[str, Any]]]:
    """Save deterministic states and aligned inputs for all 20 conditions."""
    from src.behavior.slope import slopes_by_prior, tp
    from src.task import build_trial

    behavior: dict[str, dict[str, Any]] = {}
    ts_values: list[float] = []
    tp_values: list[float] = []
    priors: list[str] = []

    model.eval()
    with torch.no_grad():
        for condition in CONDITIONS:
            inputs_np, set_step = build_trial(cfg, condition, jitter=False)
            inputs = torch.as_tensor(inputs_np, dtype=torch.float32, device=device)
            outputs, states = model(inputs, noise=False)
            output_np = outputs[0].detach().cpu().numpy()
            produced = float(tp(output_np, set_step, cfg))
            finite_tp = produced if np.isfinite(produced) else None
            store.write(
                Record(
                    model=cfg.rule,
                    seed=cfg.seed,
                    condition=condition,
                    states=states[0].detach().cpu().numpy(),
                    inputs=inputs_np[0],
                    meta={
                        "tp": finite_tp,
                        "set_step": set_step,
                        "threshold": cfg.threshold,
                        "outputs": output_np.astype(np.float32).tolist(),
                    },
                )
            )
            behavior[condition.key] = {
                "prior": condition.prior,
                "ts": condition.ts,
                "effector": condition.effector,
                "tp": finite_tp,
            }
            ts_values.append(condition.ts)
            tp_values.append(produced)
            priors.append(condition.prior)

    raw_slopes = slopes_by_prior(ts_values, tp_values, priors)
    slopes = {
        prior: float(value) if np.isfinite(value) else None
        for prior, value in raw_slopes.items()
    }
    return slopes, behavior


def train_one_seed(
    cfg: Config,
    run_dir: Path,
    *,
    activation_store_root: Path | None = None,
    checkpoint_every: int = 100,
) -> None:
    """Train one seed, resume if needed, and export its comparison activity.

    ``run_dir`` contains config, checkpoints, and metrics.  Model activations go
    to ``activation_store_root`` so all seeds/rules share the store consumed by
    RSA and iDSA. Re-running a completed seed with the same config is a no-op.
    """
    from src.task import make_batch

    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least one")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    _validate_or_write_config(cfg, config_path)
    run_meta = _read_or_init_run_meta(run_dir)
    completed_path = run_dir / "completed.json"
    if completed_path.exists():
        log.info("run already complete: %s", run_dir)
        return

    set_seeds(cfg.seed)
    identity = run_identity(cfg)
    device = torch.device(identity["device"])
    log.info("run identity: %s", identity)

    model = build_model(cfg).to(device)
    # Default: all three arms use Adam. The contrast this project makes is about the
    # *learning rule* -- the direction each rule proposes -- so the step-size policy is
    # held fixed alongside the architecture, or a rule-vs-rule difference also confounds
    # Adam-vs-SGD. It is load-bearing in practice too: PC's recurrent update is ~4
    # orders of magnitude smaller than its readout update, so under plain SGD J moves
    # by ~2e-4 over 150 iterations (frozen) while Adam moves it by ~6.3. Set
    # cfg.pc_optimizer / cfg.rflo_optimizer to "sgd" for the pure local rule.
    # See docs/RUNBOOK.md "Gaps".
    uses_optimizer = cfg.rule not in LOCAL_RULES or _local_optimizer_choice(cfg) == "adam"
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=cfg.lr) if uses_optimizer else None
    )
    rng = np.random.default_rng(cfg.seed)
    start_iter = 0
    losses: list[float] = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    latest_path = run_dir / "checkpoint_latest.pt"
    if latest_path.exists():
        start_iter, losses, best_loss, best_state = _restore_checkpoint(
            latest_path, model, optimizer, rng, device
        )
        log.info("resuming at iteration %d", start_iter)

    progress_every = max(1, checkpoint_every // 10)
    start_time = time.time()

    model.train()
    for iteration in range(start_iter, cfg.n_iter):
        batch = make_batch(cfg, cfg.batch, rng)
        inputs = torch.as_tensor(batch.inputs, dtype=torch.float32, device=device)
        target = torch.as_tensor(batch.target, dtype=torch.float32, device=device)
        mask = torch.as_tensor(batch.mask, dtype=torch.float32, device=device)

        if cfg.rule == "bptt":
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            outputs, _ = model(inputs, noise=True, return_states=False)
            loss_tensor = masked_mse(outputs, target, mask)
            if not torch.isfinite(loss_tensor):
                raise FloatingPointError(f"non-finite BPTT loss at iteration {iteration}")
            loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            loss = float(loss_tensor.detach().cpu())
        elif cfg.rule in LOCAL_RULES:
            # PC and RFLO compute their own local updates instead of autograd, but they
            # are gradients in the same sense, so they are handed to the same optimizer
            # rather than applied as plain SGD inside the model.
            diagnostics = model.infer_and_update(
                inputs, target, mask, apply_update=optimizer is None
            )
            loss = float(diagnostics["loss"])
            if not np.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite {cfg.rule} loss at iteration {iteration}"
                )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=False)
                for name, parameter in model.named_parameters():
                    parameter.grad = diagnostics["updates"][name].to(parameter.device).clone()
                optimizer.step()
        else:
            raise ValueError(f"unknown rule {cfg.rule!r}")

        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_state = copy.deepcopy(model.state_dict())

        if (iteration + 1) % progress_every == 0 or (iteration + 1) == cfg.n_iter:
            elapsed_sec = time.time() - start_time
            done_this_run = iteration + 1 - start_iter
            _atomic_json(
                {
                    "started_at": run_meta["started_at"],
                    "updated_at": datetime.now().astimezone().isoformat(),
                    "iteration": iteration + 1,
                    "n_iter": cfg.n_iter,
                    "latest_loss": loss,
                    "elapsed_sec": elapsed_sec,
                    "iters_per_sec": done_this_run / elapsed_sec if elapsed_sec > 0 else 0.0,
                },
                run_dir / "progress.json",
            )

        if (iteration + 1) % checkpoint_every == 0 or iteration + 1 == cfg.n_iter:
            _atomic_torch_save(
                _checkpoint_payload(
                    model, optimizer, iteration, losses, best_loss, best_state, rng
                ),
                latest_path,
            )
            log.info("iteration %d/%d loss=%.6g", iteration + 1, cfg.n_iter, loss)

    model.load_state_dict(best_state)
    finite_best = float(best_loss) if np.isfinite(best_loss) else None
    _atomic_torch_save(
        {"model_state": best_state, "best_loss": finite_best},
        run_dir / "model_best.pt",
    )

    store_root = Path(activation_store_root or run_dir / "activations")
    slopes, behavior = store_condition_activations(
        model, cfg, ActivationStore(store_root), device
    )
    metrics = {
        "identity": identity,
        "started_at": run_meta["started_at"],
        "finished_at": datetime.now().astimezone().isoformat(),
        "n_iter": cfg.n_iter,
        "losses": losses,
        "best_loss": finite_best,
        "behavior_slopes": slopes,
        "behavior_by_condition": behavior,
        "activation_store": str(store_root),
    }
    _atomic_json(metrics, run_dir / "metrics.json")
    _atomic_json({"status": "complete", "n_iter": cfg.n_iter}, completed_path)
