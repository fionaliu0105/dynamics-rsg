"""Smoke tests for the dependency-light foundation contracts.

Covers what can run WITHOUT torch/neurogym/rsatoolbox/DSA: the condition schema,
the config round-trip, and the activation store. Model / task / compare tracks add
their own tests as those modules land.

Run from the repo root::

    python tests/test_foundation.py        # plain-asserts, no pytest needed
    pytest tests/test_foundation.py         # also works if pytest is installed
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# make `import src...` work whether launched as a script or via pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.conditions import (
    CONDITIONS,
    N_CONDITIONS,
    Condition,
    all_conditions,
    condition_by_key,
    condition_index,
)
from src.store import ActivationStore, Record
from src.training.config import Config, sweep_configs


def test_conditions_schema():
    # 20 = prior(2) x ts(5) x effector(2), direction marginalized out.
    assert N_CONDITIONS == 20
    assert len(set(c.key for c in CONDITIONS)) == 20        # all keys unique
    # 800 ms appears in BOTH priors and they are DISTINCT conditions.
    short_800 = Condition("short", 800, "eye")
    long_800 = Condition("long", 800, "eye")
    assert short_800 != long_800
    assert condition_index(short_800) != condition_index(long_800)
    # round-trip through the string key
    for c in CONDITIONS:
        assert condition_by_key(c.key) == c
    # enumeration order is stable
    assert all_conditions() == CONDITIONS
    # schema rejects invalid cells
    for bad in [("short", 900, "eye"), ("long", 480, "hand"), ("short", 640, "foot")]:
        try:
            Condition(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")
    print("test_conditions_schema OK")


def test_config_roundtrip():
    cfg = Config.reduced(rule="pc", seed=3, pc_inference_steps=50)
    # derived step counts are computed, not stored
    assert cfg.n_steps == int(round(cfg.total_time / cfg.dt))
    assert abs(cfg.alpha - cfg.dt / cfg.tau) < 1e-12
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg.yaml"
        cfg.to_yaml(p)
        back = Config.from_yaml(p)
    assert back.rule == "pc" and back.seed == 3 and back.pc_inference_steps == 50
    assert back.N == 160 and back.dt == 5.0                 # reduced regime preserved
    assert back.n_steps == cfg.n_steps                      # derived recomputed on load
    print("test_config_roundtrip OK")


def test_sweep_grid():
    grid = sweep_configs(pc_inference_steps=[5, 20], n_seeds=4, regime="reduced")
    # The inference-step axis varies PC only; BPTT and RFLO each get a single point.
    # BPTT: 1 x 4 seeds; PC: 2 x 4; RFLO: 1 x 4  => 4 + 8 + 4 = 16
    assert len(grid) == 16
    assert sum(c.rule == "bptt" for c in grid) == 4
    assert sum(c.rule == "pc" for c in grid) == 8
    assert sum(c.rule == "rflo" for c in grid) == 4
    print("test_sweep_grid OK")


def test_store_roundtrip():
    cond = Condition("long", 1000, "hand")
    T, U, n_in = 40, 12, 3
    states = np.random.default_rng(0).standard_normal((T, U)).astype("float32")
    inputs = np.random.default_rng(1).standard_normal((T, n_in)).astype("float32")
    with tempfile.TemporaryDirectory() as d:
        store = ActivationStore(Path(d) / "activations")
        assert not store.has("pc", 7, cond)
        store.write(Record("pc", 7, cond, states, inputs, {"tp": 1012.0}))
        assert store.has("pc", 7, cond)
        rec = store.read("pc", 7, cond)
        assert np.allclose(rec.states, states)
        assert np.allclose(rec.inputs, inputs)
        assert rec.meta["prior"] == "long" and rec.meta["ts"] == 1000
        assert rec.meta["effector"] == "hand" and abs(rec.meta["tp"] - 1012.0) < 1e-6
        # idempotent overwrite: re-writing does not duplicate or error
        store.write(Record("pc", 7, cond, states * 2, inputs, {"tp": 1012.0}))
        assert np.allclose(store.read("pc", 7, cond).states, states * 2)
        assert list(store.keys()) == [("pc", 7, cond)]
        # mismatched time axes are rejected
        try:
            store.write(Record("pc", 8, cond, states, inputs[:10], {}))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for time-axis mismatch")
    print("test_store_roundtrip OK")


def main():
    test_conditions_schema()
    test_config_roundtrip()
    test_sweep_grid()
    test_store_roundtrip()
    print("\nall foundation smoke tests passed")


if __name__ == "__main__":
    main()
