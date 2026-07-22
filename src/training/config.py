"""Run configuration — one dataclass that fully specifies a run.

A new experiment is a **new config file, not an edited constant** (AGENTS.md,
"Config-driven runs"). Each run writes its config next to its seed, metrics, and
checkpoint so any result can be traced back to what produced it.

Two things this file deliberately does:

1. **Exposes every reconstruction constant as a field** — ramp shape/amplitude,
   output threshold, Weber fraction ``w_m`` — because those values DISAGREE between
   the paper text and the saved network and are **unvalidated** (AGENTS.md, "Do not
   treat reconstruction constants as validated"). They are knobs to sweep, never
   hardcoded magic numbers.
2. **Names the two regimes** — reduced (``dt=5, N=160``) for smoke tests, faithful
   (``dt=1, N=200``) for the GPU sweeps — via :meth:`Config.reduced` /
   :meth:`Config.faithful`, so the same entry point covers laptop and cluster.

Dependency-light: stdlib + PyYAML only (no torch), so configs load anywhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal

import yaml

Rule = Literal["bptt", "pc"]


@dataclass
class Config:
    """Everything a single run needs. One seed per invocation (see the trainer)."""

    # --- identity ---------------------------------------------------------------
    rule: Rule = "bptt"                       # "bptt" or "pc"; selects the learning loop
    seed: int = 0                             # THIS run's seed (sweep = many of these)

    # --- network / integration regime -------------------------------------------
    N: int = 200                              # units (faithful=200, reduced=160)
    dt: float = 1.0                           # ms per step (faithful=1, reduced=5)
    tau: float = 10.0                         # membrane time constant, ms
    noise_sd: float = 0.01                    # private per-unit process noise
    g: float = 1.0                            # recurrent gain; J ~ N(0, g^2/N)

    # --- task input encoding ----------------------------------------------------
    task_source: str = "neurogym"             # "neurogym" (default) | "standalone";
                                              # backends are byte-identical (src/task/__init__.py)
    pulse_height: float = 0.4                 # Ready/Set pulse amplitude
    pulse_width: float = 83.0                 # ms; = neurogym ReadySetGo ready/set
                                              # period duration (both 83 ms)
    ready_onset: float = 100.0                # ms from trial start to Ready
    # total_time: raised 2600 -> 3000 on 2026-07-20. The old 2600 could not contain the
    # longest condition. A trial spans ready_onset + ts (Ready->Set) + ts (Set->Go, tp~=ts)
    # + prod_hold; for max ts=1200 (long/1200) that is 100 + 1200 + 1200 + 300 = 2800 ms
    # even with NO jitter, and long/1100 was 2600 exactly (zero headroom). Training jitter
    # t_m~N(ts, ts*w_m) pushes Set later still (~+180 ms = 3*sigma at ts=1200, default
    # w_m=0.05), so at 2600 the long-prior production epoch/prod_hold truncated off the
    # [B, n_steps, 3] canvas and corrupted the target+mask. 3000 = 100 + 2*1200 + 300 + 200
    # headroom; the task generator should clip rarer t_m beyond that. n_steps stays an
    # integer at dt=1 (3000) and dt=5 (600).
    total_time: float = 3000.0               # ms; ready + 2*max_ts + prod_hold + jitter headroom
    prod_hold: float = 300.0                  # ms held at threshold after ts

    # --- UNVALIDATED reconstruction constants (expose, never trust) -------------
    # Paper vs saved-net disagreements; see AGENTS.md. Sweep these, don't assume.
    w_m: float = 0.05                         # Weber fraction of scalar meas. noise
    threshold: float = 1.0                    # Go threshold (Methods z=1; net 0.99)
    ramp_a: float = 2.8                       # ramp shape (paper 2.8; saved 3.3)
    ramp_A: float = 3.0                       # ramp amplitude (paper 3; saved 2.85)

    # --- optimisation -----------------------------------------------------------
    lr: float = 1e-3
    n_iter: int = 2500
    batch: int = 64
    grad_clip: float = 1.0

    # --- predictive coding (ignored when rule == "bptt") ------------------------
    pc_inference_steps: int = 20              # value-relaxation steps; SWEEP THIS
    pc_inference_lr: float = 0.1              # latent-state update rate
    # How PC's local updates become a step. "adam" matches the BPTT arm, so a
    # PC-vs-BPTT difference is attributable to the rule rather than to the optimizer;
    # "sgd" is the pure local rule Millidge runs, under which PC's recurrent update is
    # ~1e4 smaller than its readout update and J stays effectively frozen. This is a
    # scientific choice, so it is a config knob rather than a constant.
    pc_optimizer: str = "adam"                # "adam" | "sgd"

    # --- sweep bookkeeping ------------------------------------------------------
    n_seeds: int = 10                         # default seeds per (rule x sweep-point)

    # --- effector-context input values (tonic offsets, per effector) ------------
    prior_context: Dict[str, float] = field(
        default_factory=lambda: {"short": 0.3, "long": 0.4}
    )
    effector_context: Dict[str, float] = field(
        default_factory=lambda: {"eye": -0.2, "hand": 0.2}
    )

    # --- derived (steps); computed in __post_init__, not stored to YAML ---------
    def __post_init__(self) -> None:
        self.n_steps: int = int(round(self.total_time / self.dt))
        self.ready_onset_step: int = int(round(self.ready_onset / self.dt))
        self.pulse_width_step: int = max(1, int(round(self.pulse_width / self.dt)))
        self.prod_hold_step: int = int(round(self.prod_hold / self.dt))
        self.alpha: float = self.dt / self.tau     # Euler step for the leaky RNN

    def to_step(self, ms: float) -> int:
        """Convert milliseconds to an integer step index at this dt."""
        return int(round(ms / self.dt))

    # --- regime presets ---------------------------------------------------------
    @classmethod
    def reduced(cls, **overrides) -> "Config":
        """Smoke/CI regime: coarse and small, CPU-friendly. Under-trains on purpose."""
        base = dict(dt=5.0, N=160, n_iter=3000, batch=48)
        base.update(overrides)
        return cls(**base)

    @classmethod
    def faithful(cls, **overrides) -> "Config":
        """Paper-faithful regime: dt=1, N=200. Needs a GPU and a real iter budget."""
        base = dict(dt=1.0, N=200, n_iter=6000, batch=64)
        base.update(overrides)
        return cls(**base)

    # --- serialization ----------------------------------------------------------
    # Only the declared fields round-trip; derived step counts are recomputed.
    _DERIVED = ("n_steps", "ready_onset_step", "pulse_width_step", "prod_hold_step", "alpha")

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items()}

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_dict(cls, d: Dict) -> "Config":
        known = {f for f in cls.__dataclass_fields__}          # ignore stray/derived keys
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))


def sweep_configs(
    rules: List[Rule] = ("bptt", "pc"),
    pc_inference_steps: List[int] = (5, 20, 100),
    n_seeds: int = 10,
    regime: str = "faithful",
    **overrides,
) -> List[Config]:
    """Expand the (rule x inference-step x seed) grid into one Config per run.

    The inference-step axis only varies PC; BPTT gets a single point. This is the
    grid the team runs on GPUs. See ``docs/implementation_plan.md`` 3.2.
    """
    make = Config.faithful if regime == "faithful" else Config.reduced
    configs: List[Config] = []
    for rule in rules:
        steps_axis = pc_inference_steps if rule == "pc" else (0,)
        for steps in steps_axis:
            for seed in range(n_seeds):
                configs.append(
                    make(rule=rule, seed=seed, pc_inference_steps=steps, **overrides)
                )
    return configs
