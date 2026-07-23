"""Aggregate per-seed metrics.json files into one summary table.

Reads saved metrics only (AGENTS.md, "Plotting reads saved metrics ... never
retrains") — does not touch checkpoints or retrain anything. Meant to be the
single source of truth a results notebook reads from, rather than every
notebook recomputing its own per-seed loop over metrics.json.

CAVEAT on the loss columns: ``best_loss``/``final_loss`` are comparable between the
bptt and rflo variants (both report ``trainer.masked_mse``), but the pc variants
report ``0.5 *`` that quantity, so PC sits at half scale. Do not read a PC-vs-other
loss gap off this table as a performance difference. The similarity metrics, not the
training loss, are what the study compares across rules.

Usage::

    python scripts/summarize_runs.py --out results/runs_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# (label, run_dir, activation_store_dir) — the current sweep conditions.
# activation_store_dir is the root passed to ActivationStore; seeds missing an
# entry there simply won't have activation-derived figures, not an error.
RUN_VARIANTS = [
    ("bptt", Path("results/runs/bptt"), Path("results/activations")),
    ("pc_steps20", Path("results/runs/pc"), Path("results/activations/pc_steps20")),
    ("pc_steps100", Path("results/runs/pc_steps100"), Path("results/activations/pc_steps100")),
    ("rflo", Path("results/runs/rflo"), Path("results/activations/rflo")),
]


def _activation_seeds(store_dir: Path, rule: str) -> set[int]:
    rule_dir = store_dir / rule
    if not rule_dir.is_dir():
        return set()
    seeds = set()
    for p in rule_dir.iterdir():
        if p.is_dir() and p.name.startswith("seed_"):
            seeds.add(int(p.name.removeprefix("seed_")))
    return seeds


def summarize() -> list[dict]:
    rows = []
    for label, run_dir, store_dir in RUN_VARIANTS:
        if not run_dir.is_dir():
            continue
        rule = label.split("_steps")[0]  # "pc_steps20" -> "pc", "bptt" -> "bptt"
        act_seeds = _activation_seeds(store_dir, rule)
        for seed_dir in sorted(run_dir.glob("seed_*")):
            seed = int(seed_dir.name.removeprefix("seed_"))
            metrics_path = seed_dir / "metrics.json"
            row = {
                "variant": label,
                "rule": rule,
                "seed": seed,
                "complete": metrics_path.exists(),
                "has_activations": seed in act_seeds,
            }
            if metrics_path.exists():
                m = json.loads(metrics_path.read_text())
                bc = m.get("behavior_by_condition", {})
                valid = sum(1 for v in bc.values() if v.get("tp") is not None)
                sl = m.get("behavior_slopes", {})
                row.update(
                    best_loss=m.get("best_loss"),
                    final_loss=m.get("losses", [None])[-1],
                    n_iter_run=len(m.get("losses", [])),
                    valid_tp_count=valid,
                    valid_tp_total=len(bc),
                    slope_short=sl.get("short"),
                    slope_long=sl.get("long"),
                )
            else:
                progress_path = seed_dir / "progress.json"
                if progress_path.exists():
                    p = json.loads(progress_path.read_text())
                    row.update(
                        best_loss=None, final_loss=p.get("latest_loss"),
                        n_iter_run=p.get("iteration"), valid_tp_count=None,
                        valid_tp_total=None, slope_short=None, slope_long=None,
                    )
                else:
                    row.update(
                        best_loss=None, final_loss=None, n_iter_run=None,
                        valid_tp_count=None, valid_tp_total=None,
                        slope_short=None, slope_long=None,
                    )
            rows.append(row)
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="results/runs_summary.csv")
    args = p.parse_args(argv)

    rows = summarize()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant", "rule", "seed", "complete", "has_activations",
        "best_loss", "final_loss", "n_iter_run",
        "valid_tp_count", "valid_tp_total", "slope_short", "slope_long",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[summarize_runs] {len(rows)} rows written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
