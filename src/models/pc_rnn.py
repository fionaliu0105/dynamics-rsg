"""Predictive-coding RNN — same dynamics as BPTT, local learning.  [MEMBER TRACK: PC (pair) — plan 1.C]

THE CORE NOVELTY AND THE SINGLEST BIGGEST RISK. Build it in two SEPARATED stages:

PC-A — validate the UPDATE RULE before any RSG training
    Port Millidge's PC equations
    (https://github.com/BerenMillidge/PredictiveCodingBackprop/blob/master/rnn.py)
    onto THIS continuous-time net — reuse ``BPTTRNN.forward`` dynamics exactly; do
    NOT introduce a second architecture. PC treats hidden states as value nodes,
    forms prediction errors between successive-time predictions and at the readout,
    relaxes the latents for ``cfg.pc_inference_steps``, then applies local weight
    updates.

    Gate for PC-B (rule-internal, NO reference to BPTT):
      * PC energy decreases monotonically during latent relaxation to a fixed point;
      * weight updates are finite and stable;
      * on a toy supervised task, PC learning reduces the output loss.
    These separate a correct port from a buggy one WITHOUT assuming PC should match
    BPTT — whether it does is a QUESTION THIS PROJECT STUDIES, not a pass condition.

    Also MEASURE and report (as an observable, not a gate): per-parameter-group
    cosine similarity and relative error vs BPTT gradients, swept over
    ``cfg.pc_inference_steps``, on a deterministic toy sequence (noise off, same net,
    same init, same batch, same loss).

PC-B — train on RSG (only after PC-A's gate passes)
    Runs through the shared trainer with ``rule="pc"``: records loss + behavior,
    computes produced intervals, stores inputs/outputs/inferred-states/metadata, and
    confirms PC and BPTT expose interface-compatible states.

DEFINITION OF DONE
    PC-A: the three rule-internal checks pass and the BPTT-alignment sweep is
    reported. PC-B: trains and writes a compatible store like the BPTT arm.

REFERENCE
    Millidge et al. (PC approximates backprop) · Whittington & Bogacz 2019 ·
    Bogacz PredictiveCoding repo. Requires torch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import Model
from src.models.bptt_rnn import BPTTRNN
from src.training.config import Config


class PCRNN(nn.Module, Model):
    """Predictive-coding arm. Shares BPTTRNN's forward dynamics; differs in learning."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # Reuse the shared dynamics/params so architecture is identical across rules.
        # TODO(pc-track): hold a BPTTRNN (or share its parameter tensors) so forward()
        # is literally the same computation; PC only changes how params are updated.
        raise NotImplementedError("PC track: construct shared-parameter net (plan 1.C)")

    def forward(self, inputs, *, noise: bool = True, return_states: bool = True):
        """Identical to the BPTT forward — delegate to the shared dynamics."""
        raise NotImplementedError("PC track: delegate to shared forward (plan 1.C)")

    def infer_and_update(self, inputs, target, mask):
        """One PC step: relax latents (`cfg.pc_inference_steps`) then local updates.

        TODO(pc-track): implement PC-A here and expose the energy trace + update
        vectors so the validation harness can check the rule-internal gate and
        report BPTT-alignment as an observable.
        """
        raise NotImplementedError("PC track: implement PC inference+update (plan 1.C)")
