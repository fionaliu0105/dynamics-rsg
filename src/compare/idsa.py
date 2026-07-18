"""iDSA — input-driven dynamics (InputDSA).  [MEMBER TRACK: iDSA — plan 2.4/2.5/2.6]

THE CENTRAL METHOD. Ready/Set are strong external drive, so plain DSA (intrinsic
dynamics) is not enough — InputDSA demixes input-driven from recurrent structure
(AGENTS.md, "iDSA, not plain DSA"). Fit DMDc-based input + recurrent operators on
each system's trajectories WITH their stored inputs, then compare.

STAGES (build in order; see plan decision 11)
    Stage 3  BPTT-vs-PC        rule-vs-rule dynamics distance   -> answers RQ1
    Stage 4  model-to-DMFC     each model vs DMFC               -> answers RQ3 (HEADLINE)
    2.6      across-ts         Stage 3/4 resolved by interval   -> answers RQ2

Requirements (both systems): same conditions, inputs aligned to states, identical
preprocessing, matched k and time bins, finite & reproducible distances. Return
PER-SEED distances.

Install/validate FIRST (plan 0.6/0.7): the DSA repo (pulls kooplearn + pot) is the
shakiest install; import-check and run synthetic sanity trajectories before this
module is depended on.

DEFINITION OF DONE
    Stage 3: finite, reproducible BPTT<->PC distances on smoke-scale data.
    Stage 4: per-model, per-seed distance to DMFC with the neural noise ceiling.

REFERENCE: DSA/InputDSA — https://github.com/mitchellostrow/DSA ·
    InputDSA paper https://arxiv.org/abs/2510.25943
"""

from __future__ import annotations

import numpy as np


def fit_operators(states: np.ndarray, inputs: np.ndarray):
    """Fit DMDc input + recurrent operators from [cond, time, k] states + aligned inputs.

    TODO(idsa-track): wrap InputDSA's operator fit; keep inputs time-aligned to states.
    """
    raise NotImplementedError("iDSA track: implement fit_operators (plan 2.4)")


def dsa_distance(op_a, op_b) -> float:
    """Dynamical similarity distance between two fitted operators. TODO(idsa-track)."""
    raise NotImplementedError("iDSA track: implement dsa_distance (plan 2.4)")
