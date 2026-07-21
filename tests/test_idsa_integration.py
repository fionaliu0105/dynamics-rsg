"""Integration: iDSA Stage 3 through the REAL Preprocessor (plan 1.E + 2.4).

Proves the iDSA and Preprocess tracks connect: read variable-length trajectories from
an activation store, run them through the shared ``Preprocessor.transform_with_inputs``
(states z-scored/PCA'd/warped, inputs warped onto the same time base), then fit and
compare per seed. Uses ``backend="builtin"`` so it runs anywhere, no dsa-metric.

    python tests/test_idsa_integration.py
    pytest tests/test_idsa_integration.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.compare.idsa import InputDSAConfig, across_ts, _load_system, stage3_bptt_vs_pc
from src.conditions import CONDITIONS
from src.preprocess.pipeline import PreprocessConfig, Preprocessor
from src.store import ActivationStore, Record

UNITS = 8
N_IN = 2


def _populate_store(root, models, seeds, rng):
    """Write 20 conditions of VARIABLE-length states+inputs for each (model, seed)."""
    store = ActivationStore(root)
    for model in models:
        for seed in seeds:
            for ci, cond in enumerate(CONDITIONS):
                T = 22 + ci                       # length varies with condition (ts/tp)
                states = rng.standard_normal((T, UNITS))
                inputs = rng.standard_normal((T, N_IN))
                store.write(Record(model, seed, cond, states, inputs))
    return store


def _fitted_preprocessor(store, model, seed, k, n_time_bins):
    ref = [store.read(model, seed, c).states for c in CONDITIONS]
    return Preprocessor(PreprocessConfig(k=k, n_time_bins=n_time_bins)).fit(ref)


def test_stage3_through_real_preprocessor():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        store = _populate_store(d, ["bptt", "pc"], [0, 1], rng)
        pre = _fitted_preprocessor(store, "bptt", 0, k=6, n_time_bins=15)
        cfg = InputDSAConfig(method="dmdc", rank=6, backend="builtin")

        out = stage3_bptt_vs_pc(store, [0, 1], pre, cfg=cfg)
        assert set(out) == {0, 1}
        for seed, dist in out.items():
            assert set(dist) == {"distance", "state_distance", "input_distance"}
            assert all(np.isfinite(v) for v in dist.values()), (seed, dist)

        # states and inputs come out on the same preprocessed time base (what iDSA needs)
        s, u = _load_system(store, "bptt", 0, pre)
        assert s.shape == (20, 15, 6), s.shape
        assert u.shape == (20, 15, N_IN), u.shape
        assert s.shape[1] == u.shape[1]
    print("test_stage3_through_real_preprocessor OK")


def test_band_subsetting_respects_20_condition_contract():
    """across_ts must preprocess all 20 conditions, then subset a 10-condition band.

    (The Preprocessor rejects a partial condition set, so preprocessing a band
    directly would raise. This guards the fix.)
    """
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as d:
        store = _populate_store(d, ["bptt", "pc"], [0], rng)
        pre = _fitted_preprocessor(store, "bptt", 0, k=6, n_time_bins=12)
        cfg = InputDSAConfig(method="dmdc", rank=6, backend="builtin")

        out = across_ts(store, [0], pre, cfg=cfg)
        assert set(out) == {"short", "long"}
        for band in ("short", "long"):
            assert 0 in out[band]
            assert all(np.isfinite(v) for v in out[band][0].values()), (band, out[band])
    print("test_band_subsetting_respects_20_condition_contract OK")


if __name__ == "__main__":
    test_stage3_through_real_preprocessor()
    test_band_subsetting_respects_20_condition_contract()
    print("\niDSA integration tests passed")
