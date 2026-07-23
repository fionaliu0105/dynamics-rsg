"""Shared post-processing for learning rules that compute their own updates.

BPTT gets its direction from autograd on ``masked_mse``, which is a masked *mean*
(``trainer.py``: ``.sum() / mask.sum()``) and is then clipped by ``cfg.grad_clip``.
Local rules (PC, RFLO) accumulate raw *sums* over batch and time and are clipped by
nothing at all unless someone does it here.  Left alone the two differ by ~1e4 at the
same ``cfg.lr``, which is not a learning-rule difference -- it is a units difference.

:func:`rescale_updates` applies both corrections so a local rule's update lands on the
BPTT arm's footing.  Keeping optimizer *handling* fixed alongside the architecture is
what lets a rule-vs-rule difference be attributed to the rule (AGENTS.md).

``PCRNN._rescale_updates`` predates this module and implements the same two steps
inline.  It is deliberately NOT refactored to call this: the PC arm has committed
10-seed results, and a bit-level perturbation of a validated arm is not worth the
deduplication.  If PC is ever re-validated, collapsing the two is a safe follow-up.
"""

from __future__ import annotations

from typing import Dict

import torch


def rescale_updates(
    updates: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    grad_clip: float,
    clip_mode: str,
) -> Dict[str, torch.Tensor]:
    """Put raw summed local updates on the BPTT arm's scale, then clip them.

    Args:
        updates: raw per-parameter update tensors, accumulated as sums over batch
            and time. Treated as *gradients* (applied as ``p -= lr * update``), so
            the sign convention matches autograd's.
        mask: ``[batch, time]`` supervision mask. Its sum is the normalizer that
            turns a summed objective into ``masked_mse``'s mean.
        grad_clip: clip budget. Values ``<= 0`` disable clipping entirely.
        clip_mode: ``"global_norm"`` scales the whole joint update vector down to
            norm ``<= grad_clip`` when it exceeds it -- every parameter shares one
            budget, so ``J`` (N x N) competes with the ~2N-element readout.
            ``"elementwise"`` clamps each element of each tensor independently to
            ``[-grad_clip, grad_clip]``, with no cross-parameter competition.

    Returns:
        A new dict with the same keys. Normalization is a scalar rescale of the
        objective, so it changes step size without touching update *direction*.

    Raises:
        ValueError: if ``clip_mode`` is not one of the two supported modes.
    """
    if clip_mode not in ("global_norm", "elementwise"):
        raise ValueError(f"unknown clip_mode {clip_mode!r}")

    scale = float(mask.sum().clamp_min(1.0).item())
    updates = {name: value / scale for name, value in updates.items()}

    clip = float(grad_clip)
    if clip <= 0:
        return updates
    if clip_mode == "elementwise":
        return {name: torch.clamp(value, -clip, clip) for name, value in updates.items()}

    total = torch.sqrt(sum(value.square().sum() for value in updates.values()))
    if bool(torch.isfinite(total)) and float(total) > clip:
        return {name: value * (clip / (total + 1e-12)) for name, value in updates.items()}
    return updates


def update_alignment(update: torch.Tensor, reference_grad: torch.Tensor) -> Dict[str, float]:
    """Score one local update against the matching autograd gradient.

    ``update`` is a *gradient*, not a descent direction -- callers apply it as
    ``parameter.add_(-lr * update)`` -- so the reference is ``+reference_grad`` and a
    correct rule scores ``cosine -> +1``. Returns ``nan`` for ``cosine`` when either
    vector is exactly zero (e.g. the recurrent gradient at init, where ``w_o`` starts
    at zeros and no error can reach ``J``).
    """
    denom = update.norm() * reference_grad.norm()
    cosine = (
        float((update.flatten() @ reference_grad.flatten() / denom).item())
        if denom > 0
        else float("nan")
    )
    relative_error = float(
        (update - reference_grad).norm().div(reference_grad.norm().clamp_min(1e-12)).item()
    )
    return {"cosine": cosine, "relative_error": relative_error}
