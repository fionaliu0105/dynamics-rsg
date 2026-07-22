"""Unit tests for the DMFC ingestion core (plan 1.D).

Exercises the PURE numpy/pandas transforms in ``src.data.build_neural`` -- condition
resolution, direction-balanced averaging, stratified splits, input construction, NaN
filling, and the load-time verification -- WITHOUT nlb_tools/dandi or any download.
The thin nlb I/O (``_load_rates`` / ``_aligned_trial_arrays`` / ``_ensure_download``)
is covered by the real end-to-end run, not here.

    python tests/test_build_neural.py     # plain asserts, no pytest needed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.conditions import CONDITIONS, N_CONDITIONS, Condition, condition_index
from src.data.build_neural import (
    average_conditions,
    build_input_tensor,
    canvas_len,
    condition_event_bins,
    resolve_trials,
    stratified_splits,
    verify_neural_tensors,
    _fill_time_nans,
)
from src.training.config import Config


def _fixture(reps: int = 3, in_seconds: bool = False):
    """A synthetic trial_info + parallel per-trial arrays covering all 20 conditions.

    Two directions per condition, ``reps`` trials each; plus one outlier and one
    invalid-ts trial that must be dropped. Each trial's rate array is a constant equal
    to its condition index, so a correct direction-balanced average returns exactly the
    condition index everywhere.
    """
    rows, arrays = [], []
    T, U = 8, 4
    for c in CONDITIONS:
        for direction in (0, 1):
            for _ in range(reps):
                ts = c.ts / 1000.0 if in_seconds else c.ts
                tp = ts  # produced ~ sample for the fixture
                rows.append(dict(is_eye=(c.effector == "eye"), is_short=(c.prior == "short"),
                                 t_s=ts, t_p=tp, theta=direction, is_outlier=False))
                arrays.append(np.full((T, U), float(condition_index(c))))
    # one outlier (dropped) and one structurally invalid trial (short prior, ts=1000)
    rows.append(dict(is_eye=True, is_short=True, t_s=(0.56 if in_seconds else 560),
                     t_p=(0.56 if in_seconds else 560), theta=0, is_outlier=True))
    arrays.append(np.full((T, U), -99.0))
    rows.append(dict(is_eye=True, is_short=True, t_s=(1.0 if in_seconds else 1000),
                     t_p=(1.0 if in_seconds else 1000), theta=0, is_outlier=False))
    arrays.append(np.full((T, U), -99.0))
    return pd.DataFrame(rows), arrays


def test_resolve_trials():
    df, _ = _fixture(reps=2)
    r = resolve_trials(df)
    valid = 2 * 2 * N_CONDITIONS  # reps x directions x conditions
    assert int(r["keep"].sum()) == valid, (r["keep"].sum(), valid)
    # the outlier and the invalid-ts trial are excluded
    assert not r["keep"][-1] and not r["keep"][-2]
    # every kept trial maps to a real condition; both directions present
    kept = np.where(r["keep"])[0]
    assert set(r["cond_idx"][kept]) == set(range(N_CONDITIONS))
    assert set(np.unique(r["direction"][kept])) == {0, 1}
    print("test_resolve_trials OK")


def test_resolve_trials_units_seconds():
    # ts/tp given in SECONDS must be normalized to ms and still resolve + verify.
    df, _ = _fixture(reps=1, in_seconds=True)
    r = resolve_trials(df)
    assert int(r["keep"].sum()) == 2 * N_CONDITIONS
    # tp normalized to ms (~ hundreds/thousands, not < 10)
    assert np.nanmax(r["tp"][r["keep"]]) > 100
    print("test_resolve_trials_units_seconds OK")


def test_average_is_direction_balanced():
    df, arrays = _fixture(reps=3)
    r = resolve_trials(df)
    keep = np.where(r["keep"])[0]
    states = average_conditions(arrays, r["cond_idx"], r["direction"], keep)
    assert states.shape == (N_CONDITIONS, 8, 4)
    for c in CONDITIONS:
        i = condition_index(c)
        assert np.allclose(states[i], float(i)), (i, states[i, 0, 0])
    assert np.isfinite(states).all()
    print("test_average_is_direction_balanced OK")


def test_average_balances_imbalanced_directions():
    # dir 0 has value 0 with many trials, dir 1 has value 10 with one trial:
    # a POOLED mean would be ~0; a direction-BALANCED mean is 5.
    T, U = 4, 2
    arrays = [np.zeros((T, U)) for _ in range(9)] + [np.full((T, U), 10.0)]
    cond_idx = np.zeros(10, dtype=int)
    direction = np.array([0] * 9 + [1])
    out = average_conditions(arrays, cond_idx, direction, np.arange(10))
    assert np.allclose(out[0], 5.0), out[0, 0, 0]
    print("test_average_balances_imbalanced_directions OK")


def test_stratified_splits():
    df, _ = _fixture(reps=4)
    r = resolve_trials(df)
    keep = np.where(r["keep"])[0]
    rng = np.random.default_rng(0)
    parts = stratified_splits(r["cond_idx"], keep, 2, rng)
    assert len(parts) == 2
    # disjoint and exhaustive over the kept trials
    assert set(parts[0]).isdisjoint(set(parts[1]))
    assert set(parts[0]) | set(parts[1]) == set(keep.tolist())
    # every condition present in every split (so each split-average is well-defined)
    for p in parts:
        assert set(r["cond_idx"][p]) == set(range(N_CONDITIONS))
    print("test_stratified_splits OK")


def test_build_input_tensor_matches_model_encoding():
    cfg = Config()
    bin_ms = 20.0
    inp = build_input_tensor(cfg, bin_ms)
    T = canvas_len(cfg, bin_ms)
    assert inp.shape == (N_CONDITIONS, T, 3), inp.shape
    for c in CONDITIONS:
        i = condition_index(c)
        # ch1/ch2 are the model's tonic context values, everywhere
        assert np.allclose(inp[i, :, 1], cfg.prior_context[c.prior])
        assert np.allclose(inp[i, :, 2], cfg.effector_context[c.effector])
        # ch0 carries the two Ready/Set pulses (non-zero somewhere)
        assert inp[i, :, 0].max() > 0
    print("test_build_input_tensor_matches_model_encoding OK")


def test_event_bins_ordered():
    cfg = Config()
    bin_ms = 20.0
    mean_tp = {c.key: float(c.ts) for c in CONDITIONS}
    ev = condition_event_bins(cfg, bin_ms, mean_tp)
    T = canvas_len(cfg, bin_ms)
    for c in CONDITIONS:
        e = ev[c.key]
        assert e["ready"] < e["set"] < e["go"] <= T - 1, (c.key, e)
    print("test_event_bins_ordered OK")


def test_fill_time_nans():
    x = np.array([[np.nan, np.nan, 2.0, np.nan, 4.0, np.nan]]).reshape(1, 6, 1)
    y = _fill_time_nans(x)[0, :, 0]
    assert np.isfinite(y).all()
    assert y[0] == 2.0 and y[1] == 2.0          # leading back-filled
    assert y[-1] == 4.0                          # trailing forward-filled
    assert abs(y[3] - 3.0) < 1e-9                # interior linearly interpolated
    print("test_fill_time_nans OK")


def test_verify_neural_tensors():
    T, U = 10, 5
    states = np.zeros((N_CONDITIONS, T, U))
    inputs = np.zeros((N_CONDITIONS, T, 3))
    splits = np.zeros((2, N_CONDITIONS, T, U))
    verify_neural_tensors(states, inputs, splits)     # OK
    for bad in (
        lambda: verify_neural_tensors(states[:5], inputs, splits),           # wrong n_cond
        lambda: verify_neural_tensors(states, inputs[..., :2], splits),       # n_in != 3
        lambda: verify_neural_tensors(states, np.zeros((N_CONDITIONS, T + 1, 3)), splits),
        lambda: verify_neural_tensors(states * np.nan, inputs, splits),       # NaN
        lambda: verify_neural_tensors(states, inputs, splits[:1]),            # < 2 splits
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    print("test_verify_neural_tensors OK")


def test_noise_ceiling_roundtrip_and_figure():
    # The split-half tensors build_neural emits must flow through the SAME preprocessor
    # into rsa.noise_ceiling, and the figure must render with and without the band. The
    # figure half needs matplotlib (modeling env only); guard it so the numpy noise
    # ceiling still runs in the ingestion env.
    import importlib.util
    import tempfile

    from src.compare.rsa import noise_ceiling
    from src.preprocess.pipeline import PreprocessConfig, Preprocessor

    rng = np.random.default_rng(0)
    T, U = 30, 8
    base = rng.standard_normal((N_CONDITIONS, T, U))
    splits = [base + 0.1 * rng.standard_normal((N_CONDITIONS, T, U)) for _ in range(2)]
    pre = Preprocessor(PreprocessConfig(k=5, n_time_bins=12)).fit(base)
    splits_pp = [pre.transform(s) for s in splits]
    lo, hi = noise_ceiling(splits_pp)
    assert lo <= hi, (lo, hi)

    if importlib.util.find_spec("matplotlib") is None:
        print("test_noise_ceiling_roundtrip_and_figure OK (figure part SKIP: no matplotlib)")
        return
    from src.viz.figures import summary_distance_figure

    dist = {"RSA": {"bptt": [0.4, 0.5], "pc": [0.3, 0.35]}}
    with tempfile.TemporaryDirectory() as d:
        assert summary_distance_figure(dist, out_dir=Path(d), ceilings=None).exists()
        assert summary_distance_figure(dist, out_dir=Path(d), ceilings={"RSA": (lo, hi)}).exists()
    print("test_noise_ceiling_roundtrip_and_figure OK")


def test_integration_real_data_if_available():
    # Gated: runs the real DANDI->tensors pipeline only when nlb_tools AND the
    # downloaded dandiset are present (i.e. in the ingestion env). Skips elsewhere so
    # CI in the modeling/base env stays green.
    import importlib.util
    import tempfile

    if importlib.util.find_spec("nlb_tools") is None:
        print("test_integration_real_data_if_available ... SKIP (no nlb_tools)")
        return
    raw = Path("data/raw/000130")
    if not (raw.exists() and list(raw.glob("**/*train*.nwb"))):
        print("test_integration_real_data_if_available ... SKIP (no downloaded dandiset)")
        return

    from src.data.build_neural import build_neural, verify_neural_tensors

    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        build_neural(out_dir=out, download_dir=Path("data/raw"))
        s = np.load(out / "dmfc_rsg.npy", allow_pickle=True)
        i = np.load(out / "dmfc_inputs.npy", allow_pickle=True)
        sp = np.load(out / "dmfc_rsg_splits.npy", allow_pickle=True)
        verify_neural_tensors(s, i, sp)
        assert s.shape == (N_CONDITIONS, s.shape[1], 54) and i.shape[2] == 3
    print("test_integration_real_data_if_available ... OK (real data)")


def main():
    test_resolve_trials()
    test_resolve_trials_units_seconds()
    test_average_is_direction_balanced()
    test_average_balances_imbalanced_directions()
    test_stratified_splits()
    test_build_input_tensor_matches_model_encoding()
    test_event_bins_ordered()
    test_fill_time_nans()
    test_verify_neural_tensors()
    test_noise_ceiling_roundtrip_and_figure()
    test_integration_real_data_if_available()
    print("\nall build_neural unit tests passed")


if __name__ == "__main__":
    main()
