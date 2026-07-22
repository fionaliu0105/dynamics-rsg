"""Two-prior Ready-Set-Go task — the training-data generator (single import point).

The trainer and eval/store code import the task from HERE, never from a backend directly::

    from src.task import make_batch, build_trial

Two interchangeable backends, selected by ``cfg.task_source`` (AGENTS.md "config-driven
runs" — switching sources is a config change, not a code edit):

    "neurogym"   -> src/task/rsg_neurogym.py  TwoPriorRSG(neurogym.envs.ReadySetGo)  [DEFAULT]
    "standalone" -> src/task/rsg.py           pure-numpy generator (no neurogym)

neurogym is the default per AGENTS.md "NeuroGym is the task source of truth". The two
backends are **byte-identical** (``tests/test_task_neurogym.py``), so a run's DATA does not
depend on the choice — the selector only fixes which task-source-of-truth code path runs.
The standalone generator remains the documented fallback for a numpy<2 env without neurogym
(``docs/env_spike.md`` Solution A); select it explicitly with ``cfg.task_source="standalone"``.

Importing this package is cheap and does NOT require neurogym: the standalone backend
(``rsg.py``, numpy-only) is imported here for the shared ``Batch`` type, while the neurogym
backend is imported lazily only when actually selected. So ``import src.task`` works on a
standalone-only env; only *calling* ``make_batch`` with ``task_source="neurogym"`` there raises.
"""

from __future__ import annotations

from src.task.rsg import Batch  # shared return type; both backends return this (numpy-only)

_BACKENDS = ("neurogym", "standalone")

_MISSING_NEUROGYM = (
    "cfg.task_source='neurogym' but neurogym is not importable in this environment. "
    "Install it (docs/env_spike.md Solution D: `pip install neurogym --no-deps` plus its "
    "pure-python deps), or set cfg.task_source='standalone' to use the byte-identical "
    "pure-numpy generator (src/task/rsg.py)."
)


def _select(cfg) -> str:
    """The requested task source for this cfg (validated). Defaults to 'neurogym'."""
    src = getattr(cfg, "task_source", "neurogym")
    if src not in _BACKENDS:
        raise ValueError(f"cfg.task_source={src!r} is not one of {_BACKENDS}")
    return src


def _backend(cfg):
    """The task module to dispatch to, imported lazily. Raises if neurogym is selected
    but unavailable — no silent fallback, so a run that asked for neurogym gets neurogym."""
    if _select(cfg) == "standalone":
        from src.task import rsg
        return rsg
    from src.task import rsg_neurogym as ng
    if ng.ReadySetGo is None:
        raise ImportError(_MISSING_NEUROGYM)
    return ng


def active_backend(cfg) -> str:
    """The task source that will run for this cfg — record it in the run identity/log."""
    return _select(cfg)


def make_batch(cfg, batch, rng):
    """Training batch from the configured task source. See ``src.task.rsg.make_batch``."""
    return _backend(cfg).make_batch(cfg, batch, rng)


def build_trial(cfg, condition, jitter: bool = False):
    """Single eval/store trial from the configured source. See ``src.task.rsg.build_trial``."""
    return _backend(cfg).build_trial(cfg, condition, jitter=jitter)


__all__ = ["make_batch", "build_trial", "Batch", "active_backend"]
