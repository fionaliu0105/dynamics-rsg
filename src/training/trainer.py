"""Rule-agnostic trainer.  [FOUNDATION — I own the infra; tracks fill the rule steps; plan 2.1]

ONE seed per invocation (so an interactive run and a SLURM array task are the same
code path). Checkpoints periodically and resumes from the latest — re-running a
finished seed is a cheap no-op, which makes requeuing an array safe. Logs what
identifies the run (config, seed, git SHA, host, device, array id). Writes each
condition's states + inputs into the activation store.

The rule-specific bit is small and lives behind the model interface:
    * BPTT: forward -> masked MSE -> loss.backward() -> optimizer.step()
    * PC  : model.infer_and_update(inputs, target, mask)  (PC-B)
Everything else — seeding, checkpoint/resume, logging, store writes — is shared and
provided here.

Imports torch (won't import in the contracts-only env; that's expected).
"""

from __future__ import annotations

import logging
import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.behavior.slope import slopes_by_prior, tp
from src.conditions import CONDITIONS
from src.store import ActivationStore, Record
from src.task.rsg import build_trial, make_batch
from src.training.config import Config

log = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    """Seed torch + numpy. Full determinism isn't guaranteed on all CUDA kernels."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_identity(cfg: Config) -> dict:
    """Collect the fields that let a stray result be traced back to its run."""
    return {
        "rule": cfg.rule,
        "seed": cfg.seed,
        "host": socket.gethostname(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "git_sha": os.environ.get("GIT_SHA"),  # entry point can set this
    }


def build_model(cfg: Config):
    """Construct the model for this rule. Both share the same forward dynamics."""
    if cfg.rule == "bptt":
        from src.models.bptt_rnn import BPTTRNN
        return BPTTRNN(cfg)
    if cfg.rule == "pc":
        from src.models.pc_rnn import PCRNN
        return PCRNN(cfg)
    raise ValueError(f"unknown rule {cfg.rule!r}")


def train_one_seed(cfg: Config, run_dir: Path) -> None:
    """Train a single seed end to end, checkpoint/resume, then store activations.

    Skeleton (the loop body's rule step is the member tracks' job):

        set_seeds(cfg.seed)
        model, opt = build_model(cfg), Adam(...)
        resume_from_latest_checkpoint(run_dir, model, opt)      # TODO(me)
        for it in range(start_iter, cfg.n_iter):
            batch = make_batch(cfg, cfg.batch, rng)
            if cfg.rule == "bptt":  loss = masked_mse(model(batch.inputs), batch.target, batch.mask); loss.backward(); opt.step()
            else:                   model.infer_and_update(batch.inputs, batch.target, batch.mask)  # PC-B
            if it % ckpt_every == 0:  save_checkpoint(run_dir, model, opt, it)   # TODO(me)
        store_condition_activations(model, cfg, store)          # TODO(me)

    Stores a compact ``metrics.json`` beside checkpoints and writes one activation
    record per canonical condition under ``run_dir / "activations"``.
    """
    set_seeds(cfg.seed)
    log.info("run identity: %s", run_identity(cfg))
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(run_dir / "config.yaml")
    run_meta = _read_or_init_run_meta(run_dir)

    metrics_path = run_dir / "metrics.json"
    complete_path = run_dir / "COMPLETE"
    if complete_path.exists() and metrics_path.exists():
        log.info("run already complete: %s", run_dir)
        return _read_json(metrics_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr) if cfg.rule == "bptt" else None
    rng = np.random.default_rng(cfg.seed)

    start_iter, losses = _load_latest_checkpoint(run_dir, model, optimizer, device)
    ckpt_every = max(1, min(250, cfg.n_iter // 10 if cfg.n_iter else 1))
    progress_every = max(1, ckpt_every // 10)
    start_time = time.time()

    for iteration in range(start_iter, cfg.n_iter):
        batch = make_batch(cfg, cfg.batch, rng)
        inputs = torch.as_tensor(batch.inputs, dtype=torch.float32, device=device)
        target = torch.as_tensor(batch.target, dtype=torch.float32, device=device)
        mask = torch.as_tensor(batch.mask, dtype=torch.float32, device=device)

        if cfg.rule == "bptt":
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            outputs, states = model(inputs, noise=True)
            loss_tensor = ((outputs - target).square() * mask).sum() / mask.sum().clamp_min(1.0)
            if not torch.isfinite(loss_tensor) or not torch.isfinite(outputs).all() or not torch.isfinite(states).all():
                raise FloatingPointError(f"non-finite BPTT loss/output/state at iter {iteration}")
            loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            loss = float(loss_tensor.detach().cpu().item())
        else:
            result = model.infer_and_update(inputs, target, mask)
            loss = float(result["loss"])
            if not np.isfinite(loss) or not all(result["finite"].values()):
                raise FloatingPointError(f"non-finite PC loss/update at iter {iteration}")

        losses.append(loss)
        if (iteration + 1) % progress_every == 0 or (iteration + 1) == cfg.n_iter:
            elapsed_sec = time.time() - start_time
            done_this_run = iteration + 1 - start_iter
            _write_json(run_dir / "progress.json", {
                "started_at": run_meta["started_at"],
                "updated_at": datetime.now().astimezone().isoformat(),
                "iteration": iteration + 1,
                "n_iter": cfg.n_iter,
                "latest_loss": loss,
                "elapsed_sec": elapsed_sec,
                "iters_per_sec": done_this_run / elapsed_sec if elapsed_sec > 0 else 0.0,
            })
        if (iteration + 1) % ckpt_every == 0 or (iteration + 1) == cfg.n_iter:
            _save_checkpoint(run_dir, model, optimizer, iteration + 1, losses)

    eval_records = _store_condition_activations(model, cfg, run_dir / "activations", device)
    slopes = slopes_by_prior(
        [rec["ts"] for rec in eval_records],
        [rec["tp"] for rec in eval_records],
        [rec["prior"] for rec in eval_records],
    )
    metrics = {
        "identity": run_identity(cfg),
        "started_at": run_meta["started_at"],
        "finished_at": datetime.now().astimezone().isoformat(),
        "n_iter": cfg.n_iter,
        "losses": losses,
        "finite_loss": bool(np.isfinite(np.asarray(losses, dtype=float)).all()),
        "eval": eval_records,
        "slopes_by_prior": slopes,
        "valid_tp_count": int(np.isfinite([rec["tp"] for rec in eval_records]).sum()),
    }
    _write_json(metrics_path, metrics)
    complete_path.write_text("complete\n")
    return metrics


def _read_or_init_run_meta(run_dir: Path) -> dict:
    """Return this run's persisted metadata, creating it on the first invocation.

    ``started_at`` is written once and never overwritten, so it survives resumes —
    it answers "when did this run first start," not "when did this process start"
    (that's ``progress.json``'s ``elapsed_sec``, which is per-invocation).
    """
    path = run_dir / "run_meta.json"
    if path.exists():
        return _read_json(path)
    meta = {"started_at": datetime.now().astimezone().isoformat()}
    _write_json(path, meta)
    return meta


def _checkpoint_dir(run_dir: Path) -> Path:
    return run_dir / "checkpoints"


def _save_checkpoint(run_dir: Path, model, optimizer, iteration: int, losses: list[float]) -> None:
    ckpt_dir = _checkpoint_dir(run_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"checkpoint_iter_{iteration:06d}.pt"
    payload = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "losses": losses,
    }
    torch.save(payload, path)


def _load_latest_checkpoint(run_dir: Path, model, optimizer, device) -> tuple[int, list[float]]:
    ckpts = sorted(_checkpoint_dir(run_dir).glob("checkpoint_iter_*.pt"))
    if not ckpts:
        return 0, []
    payload = torch.load(ckpts[-1], map_location=device)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload["iteration"]), list(payload.get("losses", []))


def _store_condition_activations(model, cfg: Config, store_root: Path, device) -> list[dict]:
    store = ActivationStore(store_root)
    records: list[dict] = []
    model.eval()
    with torch.no_grad():
        for condition in CONDITIONS:
            trial_inputs, set_step = build_trial(cfg, condition, jitter=False)
            tensor_inputs = torch.as_tensor(trial_inputs, dtype=torch.float32, device=device)
            outputs, states = model(tensor_inputs, noise=False)
            if not torch.isfinite(outputs).all() or not torch.isfinite(states).all():
                raise FloatingPointError(f"non-finite eval output/state for {condition.key}")
            output_np = outputs[0].detach().cpu().numpy().astype(np.float32)
            states_np = states[0].detach().cpu().numpy().astype(np.float32)
            produced = tp(output_np, set_step, cfg)
            meta = {
                "tp": produced,
                "set_step": int(set_step),
                "threshold": cfg.threshold,
                "outputs": output_np.tolist(),
            }
            store.write(
                Record(
                    cfg.rule,
                    cfg.seed,
                    condition,
                    states_np,
                    trial_inputs[0].astype(np.float32),
                    meta,
                )
            )
            records.append(
                {
                    "condition_key": condition.key,
                    "prior": condition.prior,
                    "ts": condition.ts,
                    "effector": condition.effector,
                    "tp": produced,
                    "set_step": int(set_step),
                }
            )
    return records


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())
