"""Run the iDSA (input-driven dynamics) comparison end to end.  [iDSA track; plan 2.4/2.5/3.1]

Thin entry point, the same interactively or under SLURM (no cluster values baked in).
It reads activations off the store, routes every system through one shared
``Preprocessor`` so model and neural data get identical preprocessing (AGENTS.md), and
compares dynamics with InputDSA.

Two modes, mirroring scripts/run_rsa.py:
    * rule-vs-rule (default)  : every unordered pair of --rules, per seed. Runs today
      from the store alone.
    * model-to-DMFC (--neural): each model seed vs the DMFC operators. Needs the neural
      state tensor AND its external-input representation (--neural-inputs), so it
      activates once ingestion (1.D) and the neural input rep have landed (plan 2.5).

Backend is InputDSAConfig.backend: "dsa-metric" (official package, own env) with a
"builtin" numpy fallback. See src/compare/idsa.py.

    python scripts/run_idsa.py --store results/activations --seeds 0 1 2
    python scripts/run_idsa.py --store results/activations --seeds 0 1 2 \\
        --neural data/processed/dmfc_rsg.npy --neural-inputs data/processed/dmfc_inputs.npy
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.compare.idsa import InputDSAConfig, stage3_bptt_vs_pc, stage4_model_to_dmfc
from src.conditions import CONDITIONS
from src.preprocess.pipeline import PreprocessConfig, Preprocessor
from src.store import ActivationStore
from src.viz.figures import summary_distance_figure


def stack_states(store: ActivationStore, model: str, seed: int) -> list:
    """Read the 20 canonical conditions for (model, seed) as a ragged list of states.

    Time length varies with ts/tp, so we keep it ragged and let the Preprocessor warp
    it (same pattern as scripts/run_rsa.py::stack_system).
    """
    return [np.asarray(store.read(model, seed, c).states, dtype=np.float64) for c in CONDITIONS]


def run(
    store_root: Path,
    seeds: List[int],
    rules: List[str],
    neural_path: Optional[Path],
    neural_inputs_path: Optional[Path],
    out_dir: Path,
    cfg: InputDSAConfig,
    k: int,
    n_time_bins: int,
) -> Dict:
    store = ActivationStore(store_root)
    pre = Preprocessor(PreprocessConfig(k=k, n_time_bins=n_time_bins))
    out_dir.mkdir(parents=True, exist_ok=True)

    if neural_path is not None:
        if neural_inputs_path is None:
            raise SystemExit(
                "model-to-DMFC (--neural) also needs --neural-inputs: iDSA compares "
                "input-driven dynamics, so it requires the neural external-input "
                "representation on the shared conditions (plan 2.5 / 1.D)."
            )
        # Fit the shared time base on the neural reference so the model side warps onto
        # the same base as the brain, then preprocess the neural system once.
        neural_states = np.load(neural_path, allow_pickle=True)
        neural_inputs = np.load(neural_inputs_path, allow_pickle=True)
        pre.fit(neural_states)
        dmfc_states, dmfc_inputs = pre.transform_with_inputs(neural_states, neural_inputs)
        per = stage4_model_to_dmfc(
            store, rules, seeds, pre, dmfc_states, dmfc_inputs, cfg=cfg,
        )
        # {metric: {rule: [per-seed distance to DMFC]}} for the summary figure
        by_rule = {rule: [per[(rule, s)]["distance"] for s in seeds] for rule in rules}
        distances = {"iDSA": by_rule}
        summary_distance_figure(distances, out_dir=out_dir / "figures")
        result = {"mode": "model-to-DMFC", "distances": by_rule,
                  "components": {f"{r}:{s}": per[(r, s)] for r in rules for s in seeds}}
    else:
        # Fit the shared base on the first model system, then compare EVERY unordered
        # pair of rules per seed. Looping over pairs (rather than just rules[0] vs
        # rules[1]) is what keeps a third learning rule from being silently dropped
        # here -- with three arms the old form compared two of them and discarded the
        # rest without a word. Mirrors src.compare.rsa.rsa_distances_per_seed.
        if len(rules) < 2:
            raise SystemExit(
                f"rule-vs-rule needs at least 2 rules, got {rules}; pass --neural for "
                f"the model-to-DMFC comparison instead."
            )
        pre.fit(stack_states(store, rules[0], seeds[0]))
        by_pair = {}
        distances = {"iDSA": {}}
        for rule_a, rule_b in itertools.combinations(rules, 2):
            per = stage3_bptt_vs_pc(
                store, seeds, pre, cfg=cfg, model_a=rule_a, model_b=rule_b,
            )
            label = f"{rule_a}_vs_{rule_b}"
            by_pair[label] = {str(s): per[s] for s in seeds}
            distances["iDSA"][label] = [per[s]["distance"] for s in seeds]
        result = {"mode": "rule-vs-rule", "per_seed_by_pair": by_pair}
        summary_distance_figure(
            distances, out_dir=out_dir / "figures",
            title_suffix="distance, per seed (model-to-model, no DMFC)",
            name="summary_distance_model_to_model",
        )

    _atomic_json(out_dir / "idsa_distances.json", result)
    return result


def _atomic_json(path: Path, obj) -> None:
    """Write JSON atomically (temp then replace), mirroring the store's write idiom."""
    tmp = path.parent / f"{path.stem}.tmp.json"
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="iDSA: per-seed input-driven dynamics distances.")
    p.add_argument("--store", type=str, default="results/activations", help="ActivationStore root")
    p.add_argument("--rules", nargs="+", default=["bptt", "pc"])
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--neural", type=str, default=None,
                   help="path to DMFC state tensor [cond,time,unit]; enables model-to-DMFC")
    p.add_argument("--neural-inputs", type=str, default=None,
                   help="path to the DMFC external-input tensor [cond,time,n_in] (required with --neural)")
    p.add_argument("--out-dir", type=str, default="results/idsa")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-time-bins", type=int, default=25)
    p.add_argument("--method", choices=["dmdc", "subspace"], default=None,
                   help="operator estimator; defaults to subspace for --neural (partial obs), else dmdc")
    p.add_argument("--backend", choices=["dsa-metric", "builtin"], default="dsa-metric")
    p.add_argument("--rank", type=int, default=10)
    args = p.parse_args(argv)

    method = args.method or ("subspace" if args.neural else "dmdc")
    cfg = InputDSAConfig(backend=args.backend, method=method, rank=args.rank)

    result = run(
        store_root=Path(args.store),
        seeds=args.seeds,
        rules=args.rules,
        neural_path=Path(args.neural) if args.neural else None,
        neural_inputs_path=Path(args.neural_inputs) if args.neural_inputs else None,
        out_dir=Path(args.out_dir),
        cfg=cfg,
        k=args.k,
        n_time_bins=args.n_time_bins,
    )
    print(f"[idsa] {result['mode']}: {json.dumps(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
