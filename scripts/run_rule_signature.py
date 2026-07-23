"""Learning-rule signature figures: the within-seed null, the paired contrast, and prior structure.

Companion to scripts/run_rsa.py and scripts/run_idsa.py, which answer "how far is
each arm from DMFC". This script answers the three questions those scalars cannot:

1. **Is there a signature at all?** (RQ1) An arm x arm distance matrix whose diagonal
   is the WITHIN-arm, seed-to-seed distance. A between-rule difference only counts if
   it clears that null. Drawn for RSA (geometry) and iDSA (dynamics).
2. **Which arm is closer to DMFC, seed for seed?** (RQ3) Seed *N* of every arm starts
   from bit-identical weights, so distance-to-DMFC differences are PAIRED. The paired
   contrast uses that; the independent-bars view throws it away.
3. **Does the answer depend on interval length or prior?** (RQ2) A per-ts curve (so
   interval length is a continuous axis rather than two bands) plus the ts=800 overlap,
   where the same physical interval carries opposite priors.

Reads the activation store and data/processed/ only; retrains nothing. Same
Preprocessor instance routes every system, model and neural (AGENTS.md, "Identical
preprocessing"). No seed is excluded on behavioral grounds (AGENTS.md, "Behavior is
measured, never a filter") -- the only exclusion below is a data-provenance one,
documented at ARMS.

    python scripts/run_rule_signature.py --out-dir results/signature
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.compare.baseline import (
    paired_seed_contrast,
    signature_margin,
    within_between_matrix,
)
from src.compare.idsa import InputDSAConfig, fit_operators, input_dsa, load_system
from src.compare.rsa import build_rdm, rdm_distance
from src.conditions import CONDITIONS, OVERLAP_TS_MS, TS_BY_PRIOR
from src.preprocess.pipeline import PreprocessConfig, Preprocessor
from src.store import ActivationStore
from src.viz.figures import (
    overlap_separation_figure,
    paired_contrast_figure,
    per_ts_curve_figure,
    within_between_matrix_figure,
)

# (label, store root, rule name inside that store, seeds).
#
# pc_steps20 omits seed 1: its stored activations are BIT-IDENTICAL to pc_steps100
# seed 1 (verified with np.array_equal), so that seed carries no information about
# the 20-step inference regime and would spuriously pull the two PC arms together.
# This is a provenance exclusion, NOT a behavioral filter.
#
# Order is the LOCALITY AXIS the three-arm design exists to trace -- BPTT (nonlocal in
# time and space) -> PC (local in space, relaxed over the trajectory) -> RFLO (local in
# both) -- with the untrained control first, so a graded trend reads left to right.
# The two PC entries are inference-step variants of one arm, not two points on that axis.
ARMS = [
    ("untrained", "results/activations_untrained", "untrained", list(range(10))),
    ("bptt", "results/activations", "bptt", list(range(10))),
    ("pc_steps20", "results/activations/pc_steps20", "pc", [5, 7, 9]),
    ("pc_steps100", "results/activations/pc_steps100", "pc", list(range(10))),
    ("rflo", "results/activations", "rflo", list(range(10))),
]

LABELS = {"untrained": "Untrained", "bptt": "BPTT", "rflo": "RFLO",
          "pc_steps100": "PC (100 steps)", "pc_steps20": "PC (20 steps)",
          "DMFC": "DMFC (split halves)"}


def load_all(pre: Preprocessor) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    """{arm: {seed: {"states": [20,T,k], "inputs": [20,T,n_in]}}} through ONE preprocessor.

    Delegates to ``src.compare.idsa.load_system`` rather than reimplementing the read:
    states get z-score + PCA(k) + warp, but inputs get the WARP ONLY. The input drive
    is the physical Ready/Set stimulus, and z-scoring or PCA-projecting a 3-channel
    stimulus into k dims destroys the very quantity iDSA's B operator is estimated
    against.
    """
    out: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    for label, root, rule, seeds in ARMS:
        store = ActivationStore(root)
        by_seed = {}
        for seed in seeds:
            states, inputs = load_system(store, rule, seed, pre)
            by_seed[seed] = {"states": states, "inputs": inputs}
        out[label] = by_seed
        print(f"[signature] loaded {label}: {len(by_seed)} seeds")
    return out


def _cond_index(prior: str, ts: int) -> List[int]:
    """Indices into CONDITIONS for one (prior, ts) cell, across both effectors."""
    return [i for i, c in enumerate(CONDITIONS) if c.prior == prior and c.ts == ts]


def per_condition_fit(model_rdm: np.ndarray, neural_rdm: np.ndarray) -> np.ndarray:
    """Per-condition RSA fit: correlate each condition's ROW of the two RDMs.

    A single (prior, ts) cell has only 2 conditions (eye/hand), so a sub-RDM over one
    ts is degenerate and its distance is undefined. The row of condition *c* -- its
    dissimilarity to all 20 conditions -- is well defined, and comparing that row
    between model and DMFC localises the geometry fit to condition *c*. Returns a
    length-20 vector of distances (1 - Pearson r), one per condition.
    """
    n = model_rdm.shape[0]
    out = np.empty(n)
    for c in range(n):
        keep = np.arange(n) != c            # drop the trivially-zero self entry
        a, b = model_rdm[c, keep], neural_rdm[c, keep]
        if np.std(a) == 0 or np.std(b) == 0:
            out[c] = np.nan
        else:
            out[c] = 1.0 - float(np.corrcoef(a, b)[0, 1])
    return out


def overlap_separation(rdm: np.ndarray) -> float:
    """Dissimilarity between short-800 and long-800, averaged over effectors.

    Same physical interval, opposite prior -- so this is a direct read of how strongly
    a system encodes prior context with interval length held fixed.
    """
    vals = []
    for eff in ("eye", "hand"):
        i = next(k for k, c in enumerate(CONDITIONS)
                 if c.prior == "short" and c.ts == OVERLAP_TS_MS and c.effector == eff)
        j = next(k for k, c in enumerate(CONDITIONS)
                 if c.prior == "long" and c.ts == OVERLAP_TS_MS and c.effector == eff)
        vals.append(float(rdm[i, j]))
    return float(np.mean(vals))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store-neural", type=str, default="data/processed/dmfc_rsg.npy")
    p.add_argument("--neural-inputs", type=str, default="data/processed/dmfc_inputs.npy")
    p.add_argument("--neural-splits", type=str, default="data/processed/dmfc_rsg_splits.npy")
    p.add_argument("--out-dir", type=str, default="results/signature")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-time-bins", type=int, default=25)
    p.add_argument("--backend", type=str, default="builtin")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- shared preprocessing, fit on the neural reference -------------------------
    pre = Preprocessor(PreprocessConfig(k=args.k, n_time_bins=args.n_time_bins))
    neural_raw = np.load(args.store_neural, allow_pickle=True)
    pre.fit(neural_raw)
    neural = pre.transform(neural_raw)
    # Warp only — see load_all: the input drive must not be z-scored or PCA-projected.
    neural_inputs = pre.transform_inputs(np.load(args.neural_inputs, allow_pickle=True))
    neural_rdm = build_rdm(neural)

    ceiling = None
    splits_path = Path(args.neural_splits)
    if splits_path.exists():
        from src.compare.rsa import noise_ceiling
        splits_pp = [pre.transform(s) for s in np.load(splits_path, allow_pickle=True)]
        ceiling = tuple(noise_ceiling(splits_pp))

    systems = load_all(pre)
    arms = list(systems)
    # Neural data is partially observed, so the repo convention (scripts/run_idsa.py)
    # is the subspace estimator whenever DMFC is in the comparison. DMFC sits inside
    # the matrix below, so subspace applies to every cell -- one estimator, one scale.
    cfg = InputDSAConfig(backend=args.backend, method="subspace")
    payload: Dict[str, object] = {"arms": arms, "noise_ceiling": ceiling}

    # DMFC enters the matrix as an extra "arm" whose two members are the disjoint
    # trial halves. Its diagonal is then the split-half distance -- the neural noise
    # floor -- expressed in the SAME units as every model cell. That is what turns the
    # matrix from "which arm is closer" into "is any arm closer than the data is to
    # itself", which is the question that actually has a scale.
    split_states = [pre.transform(s) for s in np.load(splits_path, allow_pickle=True)] \
        if splits_path.exists() else []

    # --- 1. RQ1: is the between-rule difference bigger than the within-rule null? ---
    rdms = {arm: {s: build_rdm(v["states"]) for s, v in by_seed.items()}
            for arm, by_seed in systems.items()}
    if split_states:
        rdms["DMFC"] = {i: build_rdm(s) for i, s in enumerate(split_states)}
    rsa_matrix = within_between_matrix(rdms, lambda a, b: rdm_distance(a, b, method="spearman"))
    payload["rsa_matrix"] = rsa_matrix
    payload["rsa_margin"] = signature_margin(rsa_matrix)
    within_between_matrix_figure(rsa_matrix, "RSA", fig_dir,
                                 "rq1_rsa_within_between", labels=LABELS)
    print("[signature] RSA margins:", json.dumps(payload["rsa_margin"], indent=2))

    ops = {arm: {s: fit_operators(v["states"], v["inputs"], cfg) for s, v in by_seed.items()}
           for arm, by_seed in systems.items()}
    if split_states:
        ops["DMFC"] = {i: fit_operators(s, neural_inputs, cfg)
                       for i, s in enumerate(split_states)}
    idsa_matrix = within_between_matrix(ops, lambda a, b: input_dsa(a, b, cfg)["distance"])
    payload["idsa_matrix"] = idsa_matrix
    payload["idsa_margin"] = signature_margin(idsa_matrix)
    within_between_matrix_figure(idsa_matrix, "iDSA", fig_dir,
                                 "rq1_idsa_within_between", labels=LABELS)
    print("[signature] iDSA margins:", json.dumps(payload["idsa_margin"], indent=2))

    # --- 2. RQ3: paired per-seed contrast against BPTT -----------------------------
    # Model arms only; DMFC is the reference here, not a compared arm.
    model_rdms = {a: v for a, v in rdms.items() if a != "DMFC"}
    model_ops = {a: v for a, v in ops.items() if a != "DMFC"}
    rsa_to_dmfc = {arm: {s: rdm_distance(r, neural_rdm, method="spearman")
                         for s, r in by_seed.items()}
                   for arm, by_seed in model_rdms.items()}
    neural_ops = fit_operators(neural, neural_inputs, cfg)
    idsa_to_dmfc = {arm: {s: input_dsa(o, neural_ops, cfg)["distance"]
                          for s, o in by_seed.items()}
                    for arm, by_seed in model_ops.items()}
    payload["rsa_to_dmfc"] = {a: {str(s): v for s, v in d.items()} for a, d in rsa_to_dmfc.items()}
    payload["idsa_to_dmfc"] = {a: {str(s): v for s, v in d.items()} for a, d in idsa_to_dmfc.items()}

    for metric, dist, tag in (("RSA", rsa_to_dmfc, "rsa"), ("iDSA", idsa_to_dmfc, "idsa")):
        contrast = paired_seed_contrast(dist, reference_arm="bptt")
        payload[f"{tag}_paired_vs_bptt"] = contrast
        paired_contrast_figure(contrast, "BPTT", metric, fig_dir,
                               f"rq3_{tag}_paired_vs_bptt", labels=LABELS)
        print(f"[signature] {metric} paired vs BPTT:",
              {a: round(c["mean"], 4) for a, c in contrast.items()})

    # --- 3. RQ2: per-ts curve + the ts=800 overlap ---------------------------------
    curves: Dict[str, Dict[str, List[float]]] = {}
    for arm, by_seed in model_rdms.items():
        fits = np.vstack([per_condition_fit(r, neural_rdm) for r in by_seed.values()])
        per_cond = np.nanmean(fits, axis=0)          # mean over seeds, per condition
        curves[arm] = {
            prior: [float(np.nanmean(per_cond[_cond_index(prior, ts)]))
                    for ts in TS_BY_PRIOR[prior]]
            for prior in TS_BY_PRIOR
        }
    payload["per_ts_curves"] = curves
    per_ts_curve_figure(curves, TS_BY_PRIOR, "RSA", fig_dir, "rq2_per_ts_distance",
                        ceiling=None, labels=LABELS)

    seps = {arm: [overlap_separation(r) for r in by_seed.values()]
            for arm, by_seed in model_rdms.items()}
    neural_sep = overlap_separation(neural_rdm)
    payload["overlap_800_separation"] = {"arms": seps, "dmfc": neural_sep}
    overlap_separation_figure(seps, neural_sep, fig_dir, "rq2_overlap_800", labels=LABELS)
    print(f"[signature] ts=800 prior separation -- DMFC {neural_sep:.3f}; "
          + ", ".join(f"{a} {np.mean(v):.3f}" for a, v in seps.items()))

    (out_dir / "signature.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"[signature] wrote {out_dir}/signature.json and figures under {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
