"""iDSA depth: fitted-operator eigenvalue spectra + ts-band-resolved distance.

Companion to scripts/run_idsa.py (which produces one scalar distance per seed).
fit_operators already computes each system's A/B matrices; this script is the first
caller to keep them (rather than reducing straight to input_dsa's scalar) so their
eigenvalue spectrum -- stability/oscillation structure -- can be plotted. Also invokes
src.compare.idsa.across_ts (built for plan 2.6, never called from any script before
this), which resolves the same comparison by short/long prior band. No new distance
math: reuses fit_operators / Operators / across_ts as-is.

    python scripts/run_idsa_operators.py --store results/activations --rule bptt \\
        --seeds 0 1 2 3 4 5 6 7 8 9 --neural data/processed/dmfc_rsg.npy \\
        --neural-inputs data/processed/dmfc_inputs.npy --out-dir results/idsa/bptt_vs_dmfc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.compare.idsa import InputDSAConfig, across_ts, fit_operators, load_system
from src.preprocess.pipeline import PreprocessConfig, Preprocessor
from src.store import ActivationStore
from src.viz.figures import eigenvalue_spectrum_figure


def run(
    store_root: Path,
    rule: str,
    seeds: List[int],
    neural_path: Path,
    neural_inputs_path: Path,
    out_dir: Path,
    k: int,
    n_time_bins: int,
    backend: str,
    figures_root: Path = None,
) -> None:
    store = ActivationStore(store_root)
    pre = Preprocessor(PreprocessConfig(k=k, n_time_bins=n_time_bins))

    neural_states = np.load(neural_path, allow_pickle=True)
    neural_inputs = np.load(neural_inputs_path, allow_pickle=True)
    pre.fit(neural_states)
    dmfc_states, dmfc_inputs = pre.transform_with_inputs(neural_states, neural_inputs)

    cfg = InputDSAConfig(backend=backend, method="subspace")
    dmfc_op = fit_operators(dmfc_states, dmfc_inputs, cfg)
    dmfc_eigs = np.linalg.eigvals(dmfc_op.A)

    ops_dir = out_dir / "operators"
    ops_dir.mkdir(parents=True, exist_ok=True)
    np.save(ops_dir / "dmfc_eigs.npy", dmfc_eigs)

    seed_eigs = []
    for seed in seeds:
        states, inputs = load_system(store, rule, seed, pre)
        op = fit_operators(states, inputs, cfg)
        eigs = np.linalg.eigvals(op.A)
        np.save(ops_dir / f"{rule}_seed{seed:04d}_eigs.npy", eigs)
        seed_eigs.append(eigs)

        if figures_root is not None:
            eigenvalue_spectrum_figure(
                {rule: [eigs]}, dmfc_eigs, out_dir=figures_root / f"seed_{seed:04d}",
                name=f"{rule}_eigenvalue_spectrum",
            )

    fig_dir = out_dir / "figures"
    eigenvalue_spectrum_figure(
        {rule: seed_eigs}, dmfc_eigs, out_dir=fig_dir, name=f"eigenvalue_spectrum_{rule}",
    )

    # ts-band-resolved (short vs long prior) distance to DMFC -- plan 2.6, RQ2.
    band_cfg = InputDSAConfig(backend=backend, method="subspace")
    band_result = across_ts(
        store, seeds, pre, cfg=band_cfg, dmfc=(dmfc_states, dmfc_inputs), models=(rule,),
    )
    # across_ts's model-to-DMFC branch loops `models` but only the first is relevant
    # here (single-rule call); collapse to {band: {seed: distance_dict}}.
    across_ts_out = {
        band: {str(seed): band_result[band][(rule, seed)] for seed in seeds}
        for band in band_result
    }
    (out_dir / "across_ts.json").write_text(json.dumps(across_ts_out, indent=2))

    print(f"[idsa_operators] {rule}: wrote {len(seed_eigs)} operator eigenvalue sets, "
          f"across_ts bands {list(across_ts_out)}, figures under {fig_dir}"
          + (f", per-seed figures under {figures_root}" if figures_root else ""))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="iDSA depth: operator eigenvalues + ts-band-resolved distance.")
    p.add_argument("--store", type=str, required=True, help="ActivationStore root")
    p.add_argument("--rule", type=str, required=True, help="model/rule name in the store (e.g. bptt, pc)")
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--neural", type=str, required=True, help="path to DMFC state tensor [cond,time,unit]")
    p.add_argument("--neural-inputs", type=str, required=True, help="path to DMFC input tensor [cond,time,n_in]")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-time-bins", type=int, default=25)
    p.add_argument("--backend", choices=["dsa-metric", "builtin"], default="builtin")
    p.add_argument("--figures-root", type=str, default=None,
                   help="if set, also write each seed's eigenvalue spectrum into "
                        "<figures-root>/seed_XXXX/ (e.g. results/figures/bptt)")
    args = p.parse_args(argv)

    run(
        store_root=Path(args.store), rule=args.rule, seeds=args.seeds,
        neural_path=Path(args.neural), neural_inputs_path=Path(args.neural_inputs),
        out_dir=Path(args.out_dir), k=args.k, n_time_bins=args.n_time_bins,
        backend=args.backend,
        figures_root=Path(args.figures_root) if args.figures_root else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
