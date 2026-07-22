"""Parity tests: NeuroGym subclass (``src/task/rsg_neurogym.py``) vs the standalone
generator (``src/task/rsg.py``).

The two task sources MUST produce identical batches wherever both can run, so a model
trained locally on the standalone generator and one trained on the cluster NeuroGym
subclass face matched task statistics (AGENTS.md "NeuroGym is the task source of truth" /
"matched task statistics"; ``docs/env_spike.md`` Solution D). These tests assert exactly
that — and **skip cleanly** when neurogym is not importable (a standalone-only env), so
they never block CI.

Torch-free. Run from the repo root::

    python tests/test_task_neurogym.py     # plain asserts; skips if neurogym absent
    pytest tests/test_task_neurogym.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.behavior.slope import tp
from src.conditions import Condition
from src.task import rsg as standalone
from src.task import rsg_neurogym as ng
from src.training.config import Config

#: neurogym importable? The subclass module binds ``ReadySetGo`` to ``None`` when neurogym
#: is absent (guarded import), so this is the single availability switch.
HAVE_NEUROGYM = ng.ReadySetGo is not None
_SKIP_REASON = (
    "neurogym not importable (standalone-only env); see docs/env_spike.md Solution D"
)

try:  # under pytest: skip the whole module when neurogym is absent
    import pytest

    pytestmark = pytest.mark.skipif(not HAVE_NEUROGYM, reason=_SKIP_REASON)
except ImportError:  # pragma: no cover - plain-python runs need no pytest
    pytest = None


def _set_step_from_mask(mask_row):
    """First index where the production mask turns on == Set step."""
    return int(np.argmax(mask_row > 0))


def test_batch_parity_both_regimes():
    """Same seed -> byte-identical inputs/target/mask/conditions, reduced AND faithful."""
    for maker in (Config.reduced, Config.faithful):
        cfg = maker()
        a = standalone.make_batch(cfg, 16, np.random.default_rng(0))
        b = ng.make_batch(cfg, 16, np.random.default_rng(0))
        assert a.conditions == b.conditions, maker.__name__
        assert np.array_equal(a.inputs, b.inputs), f"{maker.__name__}: inputs differ"
        assert np.array_equal(a.target, b.target), f"{maker.__name__}: target differ"
        assert np.array_equal(a.mask, b.mask), f"{maker.__name__}: mask differ"
    print("test_batch_parity_both_regimes OK")


def test_build_trial_parity():
    """Eval path (no jitter) matches: identical inputs + identical set_step, both regimes.

    Covers the shortest, longest, and both overlap (short/800, long/800) conditions.
    """
    conds = [
        Condition("short", 480, "eye"),
        Condition("long", 1200, "hand"),
        Condition("short", 800, "eye"),
        Condition("long", 800, "hand"),
    ]
    for maker in (Config.reduced, Config.faithful):
        cfg = maker()
        for cond in conds:
            ia, sa = standalone.build_trial(cfg, cond)
            ib, sb = ng.build_trial(cfg, cond)
            assert sa == sb, f"{maker.__name__} {cond.label}: set_step {sa} != {sb}"
            assert np.array_equal(ia, ib), f"{maker.__name__} {cond.label}: inputs differ"
    print("test_build_trial_parity OK")


def test_subclass_uses_neurogym_timeline():
    """The subclass genuinely drives timing through neurogym: ``start_ind`` places Ready at
    ready_onset_step and Set at ready_onset_step + ts_steps (no jitter)."""
    cfg = Config.reduced()
    env = ng.TwoPriorRSG(cfg)
    for cond in (Condition("short", 640, "eye"), Condition("long", 1000, "hand")):
        ts_steps = cfg.to_step(cond.ts)
        env.new_trial(condition=cond, m_steps=ts_steps)
        assert int(env.start_ind["ready"]) == cfg.ready_onset_step
        assert int(env.start_ind["set"]) == cfg.ready_onset_step + ts_steps
    print("test_subclass_uses_neurogym_timeline OK")


def test_subclass_target_tp_recovers_ts():
    """tp() on the subclass target must recover ~ts — ties task and behavior modules."""
    cfg = Config.reduced()
    b = ng.make_batch(cfg, 12, np.random.default_rng(3))
    for i, cond in enumerate(b.conditions):
        s = _set_step_from_mask(b.mask[i])
        assert abs(tp(b.target[i], s, cfg) - cond.ts) < cfg.dt
    print("test_subclass_target_tp_recovers_ts OK")


def test_determinism_given_rng():
    cfg = Config.reduced()
    b1 = ng.make_batch(cfg, 8, np.random.default_rng(5))
    b2 = ng.make_batch(cfg, 8, np.random.default_rng(5))
    assert np.array_equal(b1.inputs, b2.inputs)
    assert np.array_equal(b1.target, b2.target)
    assert b1.conditions == b2.conditions
    print("test_determinism_given_rng OK")


def test_facade_defaults_to_neurogym():
    """The src.task facade with the default cfg routes to the neurogym backend, matching
    the standalone generator (this is the path the trainer uses)."""
    from src.task import make_batch as facade_make_batch, active_backend
    cfg = Config.reduced()                          # default task_source == "neurogym"
    assert active_backend(cfg) == "neurogym"
    a = standalone.make_batch(cfg, 8, np.random.default_rng(0))
    b = facade_make_batch(cfg, 8, np.random.default_rng(0))
    assert a.conditions == b.conditions
    assert np.array_equal(a.inputs, b.inputs)
    assert np.array_equal(a.target, b.target) and np.array_equal(a.mask, b.mask)
    print("test_facade_defaults_to_neurogym OK")


def main():
    if not HAVE_NEUROGYM:
        print(f"SKIP: {_SKIP_REASON}")
        return
    test_batch_parity_both_regimes()
    test_build_trial_parity()
    test_subclass_uses_neurogym_timeline()
    test_subclass_target_tp_recovers_ts()
    test_determinism_given_rng()
    test_facade_defaults_to_neurogym()
    print("\nall neurogym parity tests passed")


if __name__ == "__main__":
    main()
