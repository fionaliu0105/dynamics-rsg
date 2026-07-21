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
import os
import socket
from pathlib import Path

import numpy as np
import torch

from src.task import active_backend
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
        "task_source": active_backend(cfg),    # neurogym (default) or standalone
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

        from src.task import make_batch, build_trial   # backend = cfg.task_source (neurogym default)
        set_seeds(cfg.seed)
        rng = np.random.default_rng(cfg.seed)           # thread this into make_batch (deterministic)
        model, opt = build_model(cfg), Adam(...)
        resume_from_latest_checkpoint(run_dir, model, opt)      # TODO(me)
        for it in range(start_iter, cfg.n_iter):
            batch = make_batch(cfg, cfg.batch, rng)
            if cfg.rule == "bptt":  loss = masked_mse(model(batch.inputs), batch.target, batch.mask); loss.backward(); opt.step()
            else:                   model.infer_and_update(batch.inputs, batch.target, batch.mask)  # PC-B
            if it % ckpt_every == 0:  save_checkpoint(run_dir, model, opt, it)   # TODO(me)
        store_condition_activations(model, cfg, store)          # TODO(me)

    TODO(me): implement resume, checkpointing, and the per-condition store write
    (run model on each Condition with noise off, save states + inputs + tp).
    """
    set_seeds(cfg.seed)
    log.info("run identity: %s", run_identity(cfg))
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(run_dir / "config.yaml")
    raise NotImplementedError("Foundation: implement train_one_seed loop/checkpoint (plan 2.1)")
