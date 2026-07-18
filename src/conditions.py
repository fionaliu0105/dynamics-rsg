"""Condition schema for the two-prior Ready-Set-Go task — the SINGLE source of truth.

Both the task generator (``src/task``) and the neural loader (``src/data``) import
from here. **Never** redefine ts / prior / effector anywhere else: two definitions
drift, and the drift is invisible until the model-vs-neural comparison is already
wrong (see AGENTS.md, "One condition schema").

Verified facts (Sohn et al. 2019; NLB DMFC_RSG dataset):

- **prior**: ``"short"`` and ``"long"`` blocks, cued by fixation color.
- **ts** (sample interval, ms), 5 discrete values per prior, overlapping at 800::

      short = [480, 560, 640, 720, 800]
      long  = [800, 900, 1000, 1100, 1200]

  The shared 800 ms is TWO distinct conditions (short-800 vs long-800). That
  overlap is the experiment's identifiability point for the prior: same stimulus,
  opposite bias, so any activity difference there is the prior, not the interval.
- **effector**: ``"eye"`` (saccade) and ``"hand"`` (joystick). Both are modeled.
- **direction**: the neural data ALSO has 2 target directions (=> 40 real
  conditions), but we MARGINALIZE direction (average the neural data over it). It
  is NOT part of the modeled condition schema. See ``docs/implementation_plan.md``
  decision 7.

Comparison condition set = prior x ts x effector = **20 conditions**.

This module is intentionally dependency-light (stdlib only) so every other module
and every test can import it freely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# --- The factors. These five lines are the whole experimental design. -----------
PRIORS: Tuple[str, ...] = ("short", "long")
EFFECTORS: Tuple[str, ...] = ("eye", "hand")
TS_BY_PRIOR: Dict[str, Tuple[int, ...]] = {
    "short": (480, 560, 640, 720, 800),
    "long": (800, 900, 1000, 1100, 1200),
}

#: The overlap interval that appears in both priors (identifiability point).
OVERLAP_TS_MS: int = 800


@dataclass(frozen=True, order=True)
class Condition:
    """One cell of the comparison design: a (prior, ts, effector) triple.

    Frozen + ordered so Conditions are hashable dict/set keys and sort into a
    stable, reproducible order. ``ts`` is in milliseconds.
    """

    prior: str
    ts: int
    effector: str

    def __post_init__(self) -> None:
        if self.prior not in PRIORS:
            raise ValueError(f"unknown prior {self.prior!r}; expected one of {PRIORS}")
        if self.effector not in EFFECTORS:
            raise ValueError(
                f"unknown effector {self.effector!r}; expected one of {EFFECTORS}"
            )
        if self.ts not in TS_BY_PRIOR[self.prior]:
            raise ValueError(
                f"ts={self.ts} is not in the {self.prior} prior support "
                f"{TS_BY_PRIOR[self.prior]}"
            )

    @property
    def key(self) -> str:
        """Filesystem/HDF5-safe stable identifier, e.g. ``short_ts0800_eye``."""
        return f"{self.prior}_ts{self.ts:04d}_{self.effector}"

    @property
    def label(self) -> str:
        """Human-readable label, e.g. ``short/800ms/eye``."""
        return f"{self.prior}/{self.ts}ms/{self.effector}"


def all_conditions() -> Tuple[Condition, ...]:
    """Enumerate the 20 comparison conditions in a canonical, stable order.

    Order: prior (short, long) -> effector (eye, hand) -> ts (ascending). This
    order is the row/column order for RDMs and the condition axis of every stored
    tensor, so it must never change silently.
    """
    conds = []
    for prior in PRIORS:
        for effector in EFFECTORS:
            for ts in TS_BY_PRIOR[prior]:
                conds.append(Condition(prior=prior, ts=ts, effector=effector))
    return tuple(conds)


#: The canonical condition list and count. Import these, don't recompute.
CONDITIONS: Tuple[Condition, ...] = all_conditions()
N_CONDITIONS: int = len(CONDITIONS)

_INDEX_BY_KEY: Dict[str, int] = {c.key: i for i, c in enumerate(CONDITIONS)}


def condition_index(cond: Condition) -> int:
    """Position of ``cond`` on the canonical condition axis."""
    return _INDEX_BY_KEY[cond.key]


def condition_by_key(key: str) -> Condition:
    """Inverse of :attr:`Condition.key` — recover a Condition from its string id."""
    return CONDITIONS[_INDEX_BY_KEY[key]]


if __name__ == "__main__":  # a quick, dependency-free way for the team to eyeball it
    print(f"{N_CONDITIONS} conditions (prior x ts x effector):")
    for i, c in enumerate(CONDITIONS):
        marker = "  <- overlap" if c.ts == OVERLAP_TS_MS else ""
        print(f"  [{i:2d}] {c.key:>22s}   {c.label}{marker}")
