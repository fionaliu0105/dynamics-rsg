"""RSA geometry depth: RDMs, MDS, and time-resolved distance-to-DMFC.

Companion to scripts/run_rsa.py (which produces one scalar distance per seed). This
script persists the RDMs and a time-resolved distance curve that scalar collapses
away, then draws the RDM gallery / MDS / temporal figures notebooks/rsa_summary.ipynb
reads. Same store/Preprocessor/--neural conventions as run_rsa.py; reuses
src.compare.rsa.build_rdm / build_rdms_over_time / rdm_distance (no new distance math).

    python scripts/run_rsa_geometry.py --store results/activations --rule bptt \\
        --seeds 0 1 2 3 4 5 6 7 8 9 --neural data/processed/dmfc_rsg.npy \\
        --out-dir results/rsa/bptt_vs_dmfc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.compare.rsa import build_rdm, build_rdms_over_time, rdm_distance
from src.conditions import CONDITIONS
from src.preprocess.pipeline import PreprocessConfig, Preprocessor
from src.store import ActivationStore
from src.viz.figures import (
    mds_embedding_figure,
    rdm_gallery_figure,
    rdm_heatmap,
    rsa_temporal_figure,
)


def stack_system(store: ActivationStore, model: str, seed: int) -> np.ndarray:
    conds = []
    for cond in CONDITIONS:
        rec = store.read(model, seed, cond)
        conds.append(np.asarray(rec.states, dtype=np.float64))
    return np.array(conds, dtype=object)


def _total_time_ms(neural_path: Path) -> float:
    """Read the neural canvas's total trial duration from build_neural's sibling meta."""
    meta_path = neural_path.with_name("dmfc_meta.json")
    meta = json.loads(meta_path.read_text())
    return float(meta["total_time_ms"])


def run(
    store_root: Path,
    rule: str,
    seeds: List[int],
    neural_path: Path,
    out_dir: Path,
    k: int,
    n_time_bins: int,
    figures_root: Path = None,
    arm_label: str = None,
) -> None:
    arm_label = arm_label or rule
    store = ActivationStore(store_root)
    pre = Preprocessor(PreprocessConfig(k=k, n_time_bins=n_time_bins))

    neural_raw = np.load(neural_path, allow_pickle=True)
    pre.fit(neural_raw)
    dmfc_pp = pre.transform(neural_raw)
    dmfc_rdm = build_rdm(dmfc_pp)
    dmfc_rdms_t = build_rdms_over_time(dmfc_pp)

    rdms_dir = out_dir / "rdms"
    rdms_dir.mkdir(parents=True, exist_ok=True)
    np.save(rdms_dir / "dmfc.npy", dmfc_rdm)

    # Shared DMFC-only reference figure (same every arm run; cheap to overwrite).
    dmfc_fig_dir = Path("results/figures/dmfc")
    rdm_heatmap(dmfc_rdm, name="dmfc_rdm_heatmap", out_dir=dmfc_fig_dir, system_label="DMFC")

    total_time_ms = _total_time_ms(neural_path)
    times_ms = (np.arange(n_time_bins) + 0.5) * (total_time_ms / n_time_bins)

    seed_rdms = []
    temporal_curves = []
    for seed in seeds:
        raw = stack_system(store, rule, seed)
        pp = pre.transform(raw)
        rdm = build_rdm(pp)
        np.save(rdms_dir / f"{rule}_seed{seed:04d}.npy", rdm)
        seed_rdms.append(rdm)

        rdms_t = build_rdms_over_time(pp)
        curve = np.array([rdm_distance(rdms_t[t], dmfc_rdms_t[t]) for t in range(rdms_t.shape[0])])
        temporal_curves.append(curve)

        if figures_root is not None:
            seed_fig_dir = figures_root / f"seed_{seed:04d}"
            rdm_heatmap(
                rdm, name=f"{rule}_rdm_heatmap", out_dir=seed_fig_dir,
                system_label=f"{arm_label} seed {seed}",
            )
            rsa_temporal_figure(
                times_ms, {rule: curve[None, :]}, out_dir=seed_fig_dir, name=f"{rule}_rsa_temporal",
            )

    temporal = np.stack(temporal_curves, axis=0)  # [n_seeds, n_time_bins]
    np.save(out_dir / "temporal_distance.npy", temporal)

    fig_dir = out_dir / "figures"
    rdm_gallery_figure(
        {"dmfc": [dmfc_rdm], arm_label: seed_rdms}, out_dir=fig_dir, name=f"rdm_gallery_{rule}",
        seeds_by_label={"dmfc": [0], arm_label: seeds},
    )

    mean_rdm = np.mean(seed_rdms, axis=0)
    mds_embedding_figure({"dmfc": dmfc_rdm, arm_label: mean_rdm}, out_dir=fig_dir, name=f"mds_{rule}")

    rsa_temporal_figure(times_ms, {arm_label: temporal}, out_dir=fig_dir, name=f"temporal_{rule}")

    meta = {
        "rule": rule, "seeds": seeds, "n_time_bins": n_time_bins, "total_time_ms": total_time_ms,
        "mean_temporal_distance": temporal.mean(axis=0).tolist(),
    }
    (out_dir / "geometry_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[rsa_geometry] {rule}: wrote {len(seed_rdms)} RDMs, temporal curve "
          f"{temporal.shape}, figures under {fig_dir}"
          + (f", per-seed figures under {figures_root}" if figures_root else ""))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="RSA geometry depth (RDMs, MDS, time-resolved distance).")
    p.add_argument("--store", type=str, required=True, help="ActivationStore root")
    p.add_argument("--rule", type=str, required=True, help="model/rule name in the store (e.g. bptt, pc)")
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--neural", type=str, required=True, help="path to DMFC tensor [cond,time,unit]")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-time-bins", type=int, default=25)
    p.add_argument("--figures-root", type=str, default=None,
                   help="if set, also write each seed's RDM heatmap + temporal curve "
                        "into <figures-root>/seed_XXXX/ (e.g. results/figures/bptt)")
    p.add_argument("--arm-label", type=str, default=None,
                   help="figure/title label for this arm, defaults to --rule; needed "
                        "to distinguish e.g. pc_steps20 vs pc_steps100, which share "
                        "the store rule name 'pc'")
    args = p.parse_args(argv)

    run(
        store_root=Path(args.store), rule=args.rule, seeds=args.seeds,
        neural_path=Path(args.neural), out_dir=Path(args.out_dir),
        k=args.k, n_time_bins=args.n_time_bins,
        figures_root=Path(args.figures_root) if args.figures_root else None,
        arm_label=args.arm_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
