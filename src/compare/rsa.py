"""RSA — representational geometry.  [MEMBER TRACK: Preprocess & RSA — plan 2.3]

Build condition-averaged RDMs over the 20 conditions (prior x ts x effector) and
compare geometries. TWO comparisons, same shape of output:
    * rule-vs-rule  : BPTT RDM vs PC RDM               (available in the prototype)
    * model-to-DMFC : each model's RDM vs the DMFC RDM (needs neural ingestion)
with a noise ceiling estimated from neural split-halves. Return PER-SEED distances
(never a point estimate — AGENTS.md, "Seeds are the unit of evidence").

INPUT: preprocessed [condition, time, k] tensors (from src.preprocess), so both
systems already share k and the time base.

DEFINITION OF DONE (plan 2.3 check)
    Runs on neural data and on a surrogate/other-RNN; dtypes and standardized inputs
    align in time; RDMs are 20x20 over the canonical condition order.

REFERENCE: rsatoolbox — https://rsatoolbox.readthedocs.io/
"""

from __future__ import annotations

import numpy as np


def build_rdm(activity: np.ndarray) -> np.ndarray:
    """Condition x condition RDM from preprocessed [cond, time, k] activity.

    TODO(rsa-track): flatten each condition's [time, k] to a vector (or use a
    time-resolved distance) and compute the 20x20 dissimilarity matrix in the
    canonical condition order.
    """
    raise NotImplementedError("RSA track: implement build_rdm (plan 2.3)")


def rdm_distance(model_rdm: np.ndarray, ref_rdm: np.ndarray) -> float:
    """Distance between two RDMs (e.g. 1 - correlation). TODO(rsa-track)."""
    raise NotImplementedError("RSA track: implement rdm_distance (plan 2.3)")


def noise_ceiling(neural_trials) -> tuple[float, float]:
    """Upper/lower noise ceiling from neural split-halves. TODO(rsa-track)."""
    raise NotImplementedError("RSA track: implement noise_ceiling (plan 2.3)")
