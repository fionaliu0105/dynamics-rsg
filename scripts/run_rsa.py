"""Run the RSA comparison end to end.  [Preprocess & RSA track — plan 2.3 / 3.3]

This is the ONLY place that reads raw activations off disk and stacks them; the math
in ``src.compare.rsa`` stays pure. Every system is routed through ONE shared
``Preprocessor`` instance, which is the structural guarantee that model and neural
data receive identical preprocessing (AGENTS.md, "Identical preprocessing").

Two modes:
    * rule-vs-rule (default)  : BPTT vs PC per seed — runs today from the store alone.
    * model-to-DMFC (--neural): each model seed vs the DMFC RDM — needs the neural
      tensor from src.data.build_neural (data/processed/), so it activates once
      ingestion (1.D) has landed.

Interactive == SLURM: this is a thin entry point (no cluster values baked in).

    python scripts/run_rsa.py --store results/activations --seeds 0 1 2
    python scripts/run_rsa.py --store results/activations --seeds 0 1 2 \\
        --neural data/processed/dmfc_rsg.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.compare.rsa import rsa_distances_per_seed
from src.conditions import CONDITIONS
from src.preprocess.pipeline import PreprocessConfig, Preprocessor
from src.store import ActivationStore
from src.viz.figures import summary_distance_figure


def stack_system(store: ActivationStore, model: str, seed: int) -> np.ndarray:
    """Read all 20 canonical conditions for (model, seed) into a ragged system.

    Returns a list-like object array of per-condition [time, units]. Time length
    varies with ts/tp, so we keep it ragged and let the Preprocessor warp it.
    """
    conds = []
    for cond in CONDITIONS:
        rec = store.read(model, seed, cond)
        conds.append(np.asarray(rec.states, dtype=np.float64))
    return np.array(conds, dtype=object)


def load_neural(path: Path) -> np.ndarray:
    """Load the preprocessed-ready DMFC tensor [cond, time, unit] from data/processed."""
    arr = np.load(path, allow_pickle=True)
    return arr


def run(
    store_root: Path,
    seeds: List[int],
    rules: List[str],
    neural_path: Optional[Path],
    out_dir: Path,
    k: int,
    n_time_bins: int,
) -> Dict[str, Dict[str, List[float]]]:
    store = ActivationStore(store_root)
    cfg = PreprocessConfig(k=k, n_time_bins=n_time_bins)
    pre = Preprocessor(cfg)

    # Fit the shared time base. Use the neural reference if present (so the model side
    # is warped onto the same base as the brain); otherwise fit on the first system.
    if neural_path is not None:
        neural_raw = load_neural(neural_path)
        pre.fit(neural_raw)
        reference = pre.transform(neural_raw)
    else:
        first = stack_system(store, rules[0], seeds[0])
        pre.fit(first)
        reference = None

    systems_by_rule: Dict[str, Dict[int, np.ndarray]] = {}
    for rule in rules:
        by_seed = {}
        for seed in seeds:
            raw = stack_system(store, rule, seed)
            by_seed[seed] = pre.transform(raw)      # SAME Preprocessor instance
        systems_by_rule[rule] = by_seed

    per_seed = rsa_distances_per_seed(systems_by_rule, reference=reference)
    distances = {"RSA": per_seed}

    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(out_dir / "rsa_distances.json", distances)
    summary_distance_figure(distances, out_dir=out_dir / "figures")
    return distances


def _atomic_json(path: Path, obj) -> None:
    """Write JSON atomically (temp then replace), mirroring the store's write idiom."""
    tmp = path.parent / f"{path.stem}.tmp.json"
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="RSA: per-seed RDM distances + summary figure.")
    p.add_argument("--store", type=str, default="results/activations",
                   help="ActivationStore root")
    p.add_argument("--rules", nargs="+", default=["bptt", "pc"])
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--neural", type=str, default=None,
                   help="path to DMFC tensor [cond,time,unit]; enables model-to-DMFC")
    p.add_argument("--out-dir", type=str, default="results/rsa")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-time-bins", type=int, default=25)
    args = p.parse_args(argv)

    distances = run(
        store_root=Path(args.store),
        seeds=args.seeds,
        rules=args.rules,
        neural_path=Path(args.neural) if args.neural else None,
        out_dir=Path(args.out_dir),
        k=args.k,
        n_time_bins=args.n_time_bins,
    )
    mode = "model-to-DMFC" if args.neural else "rule-vs-rule"
    print(f"[rsa] {mode}: {json.dumps(distances)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
