"""Two-prior Ready-Set-Go task — NeuroGym-subclass implementation (Solution B).

This is the SECOND of two interchangeable task sources (see ``docs/env_spike.md``)::

    src/task/rsg.py           standalone numpy generator     -> local x86-mac (numpy<2)
    src/task/rsg_neurogym.py  THIS: TwoPriorRSG(ReadySetGo)  -> cluster / arm64 (numpy 2)

Both expose the SAME interface and MUST produce structurally matched batches, so a model
trained locally (standalone) and one trained on the cluster (this) face matched task
statistics (AGENTS.md "NeuroGym is the task source of truth"; "matched task statistics")::

    make_batch(cfg, batch, rng) -> Batch(inputs[B,T,3], target[B,T], mask[B,T], conditions)
    build_trial(cfg, condition, jitter=False) -> (inputs[1,T,3], set_step:int)

HOW IT USES NEUROGYM
    ``TwoPriorRSG`` subclasses neurogym's ``ReadySetGo`` (a ``TrialEnv``) and drives the
    trial timeline through its period machinery: ``add_period`` lays out the SAME skeleton
    ``fixation -> ready -> measure -> set -> production`` with matched durations
    (fixation = ``cfg.ready_onset``, ready/set = ``cfg.pulse_width``, measure = the
    Ready->Set gap). ``set_step`` is read back from neurogym's ``start_ind``. We override
    the three things ReadySetGo builds for discrete RL:
      * observation -> our 3 channels {Ready/Set pulse, prior-context, effector-context}
        (base uses {fixation, ready, set}), written with ``add_ob``/``set_ob``.
      * ``_new_trial`` -> receives the Condition from ``make_batch`` (it never self-samples
        ``measure``); ``measure`` = ``t_m`` (jittered in training) while the target stays
        timed to the TRUE ts.
      * target -> a continuous ramp-to-threshold + hold (shared ``rsg.ramp``), NOT the
        go-impulse, timed from Set ONSET to match ``rsg.py`` and ``behavior/slope.py``.
    Each trial is rendered onto the fixed ``[n_steps, 3]`` canvas aligned at trial start
    (neurogym's ``fixation`` starts at t=0), tail-padded.

To keep the two sources matched, the condition draw, the ``t_m ~ N(ts, ts*w_m)`` jitter,
its canvas-fit clip, and the ramp are the SHARED helpers from ``rsg.py`` (not re-derived).
neurogym's own ``self.rng`` is left unused for sampling; determinism is controlled by the
caller's ``rng`` exactly as in ``rsg.make_batch``.

Numpy at the boundary — no torch import here. This module DOES import neurogym, so it only
runs in the numpy-2 env; the import is guarded so the rest of the repo still imports where
neurogym is absent (local x86-mac). Import errors surface only when this module is used.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.conditions import CONDITIONS, N_CONDITIONS, Condition
from src.task.rsg import Batch, _measurement_steps, ramp  # SHARE: matched batches
from src.training.config import Config

# neurogym import is guarded: the module is importable-checked lazily so a repo checkout on
# the standalone (numpy<2) env does not fail at import time — only constructing the env does.
try:  # pragma: no cover - exercised only in the numpy-2 env
    from neurogym.envs.native.readysetgo import ReadySetGo
except Exception:  # noqa: BLE001 - fall back across neurogym layouts
    try:
        from neurogym.envs.readysetgo import ReadySetGo  # type: ignore
    except Exception:  # noqa: BLE001
        ReadySetGo = None  # type: ignore

try:  # pragma: no cover
    from neurogym.utils import spaces as _ngym_spaces
except Exception:  # noqa: BLE001
    _ngym_spaces = None


def _require_neurogym() -> None:
    if ReadySetGo is None:
        raise ImportError(
            "neurogym is not importable in this environment. This module is the "
            "numpy-2 (cluster / native-arm64) task source; on local x86-mac use the "
            "standalone generator src/task/rsg.py instead. See docs/env_spike.md."
        )


class TwoPriorRSG(ReadySetGo if ReadySetGo is not None else object):  # type: ignore[misc]
    """ReadySetGo extended to two priors + effectors, our 3-channel encoding, and a
    continuous ramp target. Constructed per-run from a :class:`Config`; used only inside
    this module's ``make_batch``/``build_trial``. See the module docstring."""

    def __init__(self, cfg: Config) -> None:
        _require_neurogym()
        self.cfg = cfg
        timing = {
            "fixation": cfg.ready_onset,   # pre-Ready dead time (neurogym default 100 ms)
            "ready": cfg.pulse_width,      # Ready pulse   (neurogym default 83 ms)
            "measure": cfg.ready_onset,    # placeholder — overridden per trial via duration=
            "set": cfg.pulse_width,        # Set pulse     (neurogym default 83 ms)
        }
        super().__init__(dt=cfg.dt, timing=timing)
        # Re-map the 3 obs channels to our encoding: pulse / prior-context / effector-context.
        if _ngym_spaces is not None:
            name = {"pulse": 0, "prior": 1, "effector": 2}
            self.observation_space = _ngym_spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float32, name=name
            )

    def _new_trial(self, **kwargs):
        """Build one trial from a Condition passed in by the caller (never self-sampled).

        ``m_steps`` (Ready->Set gap in steps) is drawn by the caller via the shared
        :func:`rsg._measurement_steps`, so jitter/clip match the standalone generator.
        """
        cfg = self.cfg
        cond: Condition = kwargs["condition"]
        m_steps: int = int(kwargs["m_steps"])
        ts_steps = int(round(cond.ts / cfg.dt))

        # Same skeleton as ReadySetGo; measure passed in, production long enough to hold
        # the onset-timed ramp (set_onset + ts + prod_hold < set_end + ts + prod_hold).
        self.add_period(["fixation", "ready"])
        self.add_period("measure", duration=m_steps * cfg.dt, after="fixation")
        self.add_period("set", after="measure")
        self.add_period(
            "production", duration=(ts_steps + cfg.prod_hold_step) * cfg.dt, after="set"
        )

        set_step = int(self.start_ind["set"])

        # Observation: pulse on ch0 during ready & set; tonic context on ch1 / ch2.
        self.add_ob(cfg.pulse_height, "ready", where=0)
        self.add_ob(cfg.pulse_height, "set", where=0)
        self.set_ob(cfg.prior_context[cond.prior], where=1)
        self.set_ob(cfg.effector_context[cond.effector], where=2)

        return {
            "condition": cond,
            "ts": cond.ts,
            "measure_steps": m_steps,
            "set_step": set_step,
        }


def _render(env: "TwoPriorRSG", cfg: Config, cond: Condition, jitter: bool, rng):
    """Run one neurogym trial and render it onto the fixed ``[n_steps, 3]`` canvas.

    Returns ``(inputs[T,3], target[T], mask[T], set_step)``. neurogym owns the TIMELINE —
    it places the Ready/Set periods (``env.start_ind``). We render the fixed canvas from
    those indices with the SHARED helpers (:func:`rsg.ramp`, ``cfg.pulse_width_step``) so
    the batch is byte-identical to :func:`rsg._build` at every ``dt``. (Copying ``env.ob``
    directly would differ by one pulse-tail step at coarse ``dt`` because neurogym floors
    the pulse period, e.g. 83 ms -> 16 steps at dt=5 vs our rounded 17.) The neurogym-native
    observation is still built in ``_new_trial`` (valid if the env is driven via gym).
    """
    m_steps = _measurement_steps(cfg, cond.ts, jitter, rng)
    env.new_trial(condition=cond, m_steps=m_steps)

    ready_step = int(env.start_ind["ready"])                # both from neurogym's timeline
    set_step = int(env.start_ind["set"])
    ts_steps = int(round(cond.ts / cfg.dt))
    T = cfg.n_steps
    pw = cfg.pulse_width_step

    inputs = np.zeros((T, 3), dtype=np.float32)
    inputs[ready_step:ready_step + pw, 0] = cfg.pulse_height  # Ready pulse (neurogym onset)
    inputs[set_step:set_step + pw, 0] = cfg.pulse_height      # Set pulse   (neurogym onset)
    inputs[:, 1] = cfg.prior_context[cond.prior]             # tonic prior context, full canvas
    inputs[:, 2] = cfg.effector_context[cond.effector]       # tonic effector context, full canvas

    target = np.zeros(T, dtype=np.float32)
    mask = np.zeros(T, dtype=np.float32)
    prod_end = min(set_step + ts_steps + cfg.prod_hold_step, T)
    rel = np.arange(prod_end - set_step)
    target[set_step:prod_end] = ramp(rel, ts_steps, cfg)    # ramp-to-threshold + hold
    mask[set_step:prod_end] = 1.0                           # supervise the production epoch
    return inputs, target, mask, set_step


def make_batch(cfg: Config, batch: int, rng: np.random.Generator) -> Batch:
    """Sample a training batch via the NeuroGym subclass. Interface- and statistics-matched
    to :func:`rsg.make_batch`: sole condition sampler, jittered Ready-Set gap, target timed
    to the true ts, deterministic given ``rng``."""
    _require_neurogym()
    env = TwoPriorRSG(cfg)
    idxs = rng.integers(0, N_CONDITIONS, size=batch)
    conditions: List[Condition] = [CONDITIONS[int(k)] for k in idxs]

    inputs = np.empty((batch, cfg.n_steps, 3), dtype=np.float32)
    target = np.empty((batch, cfg.n_steps), dtype=np.float32)
    mask = np.empty((batch, cfg.n_steps), dtype=np.float32)
    for i, cond in enumerate(conditions):
        inputs[i], target[i], mask[i], _ = _render(env, cfg, cond, jitter=True, rng=rng)
    return Batch(inputs=inputs, target=target, mask=mask, conditions=conditions)


def build_trial(cfg: Config, condition: Condition, jitter: bool = False):
    """Single-condition input (no jitter by default) for eval/storage. Mirrors
    :func:`rsg.build_trial`: returns ``(inputs[1, T, 3], set_step)``."""
    _require_neurogym()
    env = TwoPriorRSG(cfg)
    rng = np.random.default_rng() if jitter else None
    inputs, _, _, set_step = _render(env, cfg, condition, jitter=jitter, rng=rng)
    return inputs[None, ...], int(set_step)
