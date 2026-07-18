# Implementation Plan — Learning Rules Shape Latent Geometry & Dynamics (RSG)

Status: agreed after grilling session, 2026-07-17. This is the build guide for the
one-week sprint. It records the decisions we made, the task breakdown, and the
risks to watch. It does **not** restate AGENTS.md — read that first; this plan
assumes its invariants.

## Context

We train one RNN architecture under two learning rules — BPTT and predictive
coding (PC) — on the two-prior Ready-Set-Go task, and ask whether the *rule*
leaves a measurable signature on latent geometry (RSA) and input-driven dynamics
(iDSA), and whether PC sits closer to macaque DMFC than BPTT. Architecture is held
fixed so any difference is attributable to the rule.

## Decisions locked in this session

1. **Same architecture, PC as the optimizer.** Both arms share **one** forward
   model: the continuous-time leaky tanh RNN from the reconstruction (Sohn-
   faithful). Only the learning loop differs. BPTT and PC call the *same*
   `forward()`.
2. **PC is ported onto the continuous-time net.** Millidge's `rnn.py` is a
   discrete Elman RNN on char-prediction — we use its **PC update equations**
   (hidden prediction errors, `n_inference_steps` of value relaxation, local
   weight updates), not its network. **This port is the single biggest technical
   risk.** Correctness is checked **rule-internally** (PC-A / 1.C): the PC energy
   descends during latent relaxation and the updates are finite and stable —
   **not** by requiring the updates to match BPTT. Whether, and how closely, PC
   tracks BPTT is a *measured observable this project investigates*, not an
   assumption and not a pass/fail bar.
3. **Inference-step count is a swept experimental axis**, not a fixed constant.
   Prior theory (Millidge & Bogacz) predicts PC *may* approach BPTT as inference
   steps grow, but we treat that as a **hypothesis the sweep tests, not a
   guaranteed control** — for this continuous-time recurrent port it is an open
   question. The phenomenon of interest is how the latent-geometry/dynamics
   signature varies across the sweep, whatever the high-step limit turns out to be.
4. **No behavioral gating, ever.** Every seed enters RSA/iDSA regardless of
   behavior. The tp-vs-ts slope is computed and carried as a **reported covariate**
   next to every similarity value. This formally overrides the Proposal's "for
   networks that pass" language — a conscious, documented departure.
5. **Package stays `src`.** Imports are `from src....`. No rename.
6. **Neural data via `nlb_tools`** (DMFC_RSG is a first-class NLB dataset). The
   load-bearing design is the `data/processed/` handoff, not the env split:
   ingestion runs once and writes tensors; modeling code never imports
   `pynwb`/`dandi`. Attempt a **single env first**; isolate ingestion in its own
   env **only if** the combined install hits the known `numpy`/`scipy` conflict
   (dandi/pynwb/nlb_tools vs torch/rsatoolbox/DSA+kooplearn+pot). The seam is the
   same either way, so splitting later costs no code.
7. **Matched condition set = prior × ts × effector (20).** The neural data has
   **40 real conditions** (prior 2 × ts 5 × effector 2 × direction 2). We compare
   on **20**: keep prior, ts, and effector; **marginalize direction** by averaging
   the neural data over its 2 target directions. ts is kept resolved because the
   Bayesian computation lives in the ts→tp mapping and Research Question 2 is
   explicitly about interval length — marginalizing ts would make RQ2 unanswerable
   and average away the prior's signature. Effector enters the network as its own
   tonic context input and drives a **per-effector (effector-gated) readout**, so
   it is causally used, not a passive label.
   - *Data sufficiency:* model training is unaffected — NeuroGym is a generator,
     so every ts is sampled thousands of times regardless of conditioning. The
     finite side is neural condition-averaging: ~1,289 trials / 40 real conditions
     ≈ 32 each, and pooling the 2 directions gives ~64 per comparison condition —
     ample for stable trial-averaged trajectories and a split-half noise ceiling.
     Verify per-cell trial counts on load (the prior is *sampled*, so cells may be
     unbalanced); pool or flag thin cells.
8. **Default 10 seeds** per (rule × sweep-point), config-overridable.
9. **Two regimes, config-selected.** Reduced (`dt=5, N=160`) for smoke tests and
   CI; faithful (`dt=1, N=200`) for the GPU sweeps the team runs separately.
10. **PC is built in two separated deliverables: PC-A then PC-B.** PC-A validates
    the *update rule itself* with **rule-internal checks** — PC energy descends
    during relaxation, updates are finite and stable, and a toy supervised task's
    loss decreases under PC learning. PC-B trains PC on RSG and does not start
    until PC-A passes those checks. (PC-A also *measures* alignment with BPTT
    gradients and reports it as an observable, but alignment is **not** the pass
    condition — matching BPTT is a question this project studies, not a
    requirement.) This prevents debugging RSG training before the PC update is
    known-correct.
11. **InputDSA is validated progressively in 4 stages, not all-or-nothing:**
    (1) install/API, (2) synthetic-trajectory sanity, (3) BPTT-vs-PC comparison,
    (4) **model-to-DMFC comparison — the central deliverable.** Staging is about
    *dependency order*, not priority: Stage 4 is the whole reason this is a NeuroAI
    project (the in-silico/in-vivo comparison) and it is what **RQ3** asks, so it is
    a **required** goal, not optional. Stages 1–3 are the de-risking path *toward*
    Stage 4, not a substitute for it. The only thing gating Stage 4 is that neural
    ingestion (1.D) must be built and validated first — you cannot compare to DMFC
    before the DMFC tensors exist. Stage 3 (rule-vs-rule) answers **RQ1** along the
    way; RQ3 lands at Stage 4. Plan the sprint so ingestion is not left to the end.

## Team & division of labor

This is a course project; a primary goal is **learning by implementing**. Work is
split so each of the 5 members owns a full **module** end to end — interface →
implementation → its test → a small figure — against shared contracts, while the
structural foundation is provided so everyone can work in parallel from day one.

**Provided foundation (documented teaching scaffold, with stubs + `TODO` seams):**
interface contracts (`config`, `conditions.py`, base model interface, store
schema); repo scaffolding + env; the **neural data loader** (1.D); the
rule-agnostic **trainer infra** (2.1 minus the learning loops); the **viz +
smoke-test harness** (3.3); and **InputDSA** installed with a synthetic-check
template (0.6–0.7).

**Member tracks — one module each (map members by interest/skill):**

| Track | Owns (plan tasks) | What they learn | Level |
| --- | --- | --- | --- |
| Task & behavior | 1.A, 2.2 | Task design, NeuroGym, the RSG paradigm, behavioral analysis | Low–med |
| BPTT RNN | 1.B | RNN dynamics, BPTT training, the model itself | Med |
| PC RNN *(pair)* | 1.C (PC-A), 2.1 PC-B path | Predictive coding, local learning rules — the core novelty | High |
| Preprocess & RSA | 1.E, 2.3 | Standardization, representational geometry, RSA | Med |
| iDSA | 2.4, 2.5, 2.6 | Dynamical similarity, Koopman/DMDc, the dynamics half | Med–high |

**Dependency notes handled in the foundation so nobody blocks:** preprocessing
(1.E) feeds *both* RSA and iDSA — its interface + a stub ship in the scaffold so
the iDSA member develops against the stub until the real one lands; and the PC net
reuses the **BPTT net's `forward()`** (identical dynamics = the architecture-parity
point), so that interface is fixed first.

**Working rhythm:** start against stubs/mocks on day one; **pair on PC** (highest
risk and highest learning); hold a **mid-sprint integration checkpoint** that swaps
mocks for real modules and runs the smoke pipeline end to end on the reduced regime.

## Open items to confirm (flagged, not silently decided)

- **Effector readout mechanism (decided in principle, detail open).** Effector is
  modeled (decision 7): a tonic effector-context input + a per-effector readout.
  Open detail: whether "per-effector readout" is two separate output channels
  (effector-context gates which one is scored by the loss) or a single readout
  whose target direction is set by effector. Two channels is the more faithful
  default; confirm against how DMFC_RSG encodes the response.
- **Condition values must match the data.** `conditions.py` ts/prior/**effector**
  values (short `[480,560,640,720,800]`, long `[800,900,1000,1100,1200]`, two
  effectors) are from the reconstruction/paper. The NLB paper confirms the data is
  **40 conditions = prior 2 × ts 5 × effector 2 × direction 2**, both effectors
  present. Still verify on load that `nlb_tools` exposes these exact ts values and
  effector/direction labels, and that averaging over direction yields the intended
  20-condition set, before trusting any RSA/iDSA alignment.
- **iDSA install de-risk (core, not cuttable).** iDSA is the heart of the project
  (input-driven dynamics of BPTT vs PC), so InputDSA is **not** a cut candidate.
  Because it is the shakiest install (DSA repo pulls kooplearn + pot), we validate
  its install and API **early**, in Phase 0, on toy trajectories — before the
  pipeline depends on it. Treat a broken InputDSA install as a Phase-0 blocker,
  not a late surprise.

## Architecture contracts (define first — they unblock parallel work)

- **Latent interface.** Every model exposes `forward(inputs) -> outputs, states`
  with `states` shaped `[trials, time, units]`. The comparison "activity" is
  `r = tanh(x)` (rate-like, bounded), on the neural time base after preprocessing.
- **Condition schema.** `src/conditions.py` defines ts, prior, effector **once**.
  Task generator and neural loader both import it. Never redefined locally.
- **Store keys.** Activation store is keyed by `(model, seed, condition)` and
  holds `states` **and** the input drive `u` **and** metadata (ts, prior, tp).
  iDSA needs inputs aligned to states, so inputs are stored, not reconstructed.

## Task breakdown

Ordered by dependency. Phase 0 is a hard prerequisite; Phase 1 tracks run in
parallel (team of 5 maps naturally onto the five Phase-1 tracks); Phase 2
integrates; Phase 3 sweeps and plots. Each task notes its output and its check.

### Phase 0 — Contracts & scaffolding (do first, together)

- **0.1 Config system.** `src/training/config.py` (or dataclass): regime
  (`dt`, `N`, `tau`, `noise_sd`, `g`), rule (`bptt`|`pc`), `pc_inference_steps`,
  `pc_inference_lr`, seed, condition set, ramp constants (`ramp_a`, `ramp_A`,
  `threshold`, `w_m`) — **all reconstruction constants exposed, none hardcoded.**
  Runs are config files, not edited constants. *Check:* a config round-trips to
  YAML and back.
- **0.2 `src/conditions.py`.** ts × prior × **effector** schema + helpers (20
  conditions). *Check:* task and (stubbed) loader both import it; one source of truth.
- **0.3 Base model interface.** `src/models/base.py`:
  `forward(inputs, return_states=True) -> outputs [trials,time], states
  [trials,time,units]`. *Check:* a dummy model satisfies it.
- **0.4 Store schema.** `src/store/` read/write over zarr or hdf5, keyed
  `(model, seed, condition)`, fields `{states, inputs, meta}`. *Check:* write →
  read → shapes and labels survive (round-trip test).
- **0.5 Env(s).** Try one env (torch, neurogym, rsatoolbox, DSA, dandi, pynwb,
  nlb_tools). If `numpy`/`scipy` resolution fails, split ingestion (dandi, pynwb,
  nlb_tools) into its own env writing `data/processed/`. Record any pin changed
  and why; document whichever layout you end up with in the README.
- **0.6 InputDSA Stage 1 — install & API (blocker).** Install the DSA repo
  (+ kooplearn, pot). Confirm: package imports; expected classes/functions exist;
  input and state tensors are accepted in the required shapes; a full call returns
  a **finite** result. Complete before the rest of the pipeline depends on it — a
  broken install here blocks the sprint and is fixed now, not late.
- **0.7 InputDSA Stage 2 — synthetic-trajectory validation.** Run InputDSA on
  small controlled trajectories with known relationships and confirm the distance
  ordering is sane, **per case** (the last case is why we use InputDSA over DSA):
  - two identical systems → distance ≈ 0 (smallest);
  - one system with shuffled time points → large distance;
  - one system with perturbed recurrent dynamics → larger distance than identical;
  - two systems with the **same recurrent dynamics but different external inputs**
    → **small** recurrent-dynamics distance (InputDSA demixes the input drive and
    should *not* call these far apart, whereas plain DSA would). *Check:* related
    systems score closer than shuffled/perturbed ones, and the same-dynamics/
    different-input pair is judged similar.

### Phase 1 — Parallel build

- **1.A Task module.** `src/task/` extends NeuroGym `ReadySetGo-v0` to two priors
  + a tonic prior-context channel + a tonic **effector-context channel** + Ready/
  Set pulse channel. Emits per-trial ts, prior, **effector** labels. *Check:*
  input carries both context channels; every trial carries ts, prior, effector;
  dt is what we expect.
- **1.B BPTT RNN.** `src/models/bptt_rnn.py` — the continuous-time leaky tanh net
  from the reconstruction, behind the base interface, with a **per-effector
  (effector-gated) readout**, masked-MSE-to-ramp loss on the active effector's
  channel. Constants come from config. *Check:* trains on trivial data, loss
  drops, tp finite and ordered, correct effector channel is the one that crosses.
- **1.C PC-A — PC update validation (RISK, gate for PC-B).** `src/models/pc_rnn.py`
  — **same forward dynamics and readout as 1.B**; PC learning loop ported from
  Millidge's equations onto the leaky dynamics: value nodes = hidden states,
  prediction errors between successive-time predictions and at the readout,
  `n_inference_steps` relaxation, then local weight updates. **Validate the update
  before any RSG training**, on a small deterministic sequence (same net, same
  init params, same input batch, **process noise disabled**, same output loss).
  - *Gate for PC-B (rule-internal, no reference to BPTT):* PC energy decreases
    monotonically during latent relaxation toward a fixed point; weight updates
    are finite and stable; on a toy supervised task, PC learning reduces the
    output loss. These distinguish a correct port from a buggy one **without**
    assuming PC should match BPTT.
  - *Also measured and reported as an observable (not a gate):* PC-vs-BPTT
    per-parameter-group cosine similarity and relative update error, swept over
    `n_inference_steps`. This quantifies how far PC departs from BPTT — a question
    the project studies, not a requirement. (Per-group cosine localizes any
    residual; likely sources are weight-sharing-across-time and the leak term.)
- **1.D Neural loader.** `src/data/build_neural.py` (isolated env) — `nlb_tools`
  reads dandiset 000130, aligns to Ready/Set/Go, bins to rates, **averages over
  the 2 target directions** to collapse the 40 real conditions to the 20-condition
  comparison set, writes `[condition, time, unit]` + behavioral tp into
  `data/processed/`, tagged with the **same** (ts, prior, **effector**) metadata
  as the model side. Verified data facts: 1 monkey (Haydn/"H"), 1 session, 54
  sorted units (40 held-in / 14 held-out — use all 54 or held-in for our averaged
  analyses), ~1,289 trials. *Check:* NWB structure / trial events / spike arrays
  inspected; binned shapes sane; ts values and **both effector labels** match
  `conditions.py`; per-cell trial counts logged (flag thin cells).
- **1.E Preprocessing.** `src/preprocess/` — per-unit normalization → PCA to a
  shared k → warp/bin to matched time bins aligned on task events. Applied
  identically to model and neural data; structurally hard to bypass. *Check:*
  output dimensionality and time bins match across model and neural.

### Phase 2 — Integrate & compare

- **2.1 Trainer (incl. PC-B).** `src/training/train.py` — **one seed per
  invocation**, checkpoint/resume, logs config+seed+git SHA+host+device+array-id,
  writes states and inputs into the store. Re-running a finished seed is a no-op.
  Running it with `rule=pc` **is PC-B** — RSG training of the PC model — and
  PC-B records training loss, behavioral performance, produced intervals, and stores inputs,
  outputs, inferred states, and trial metadata, and confirms PC and BPTT expose
  compatible state representations. *Check:* interactive run and (stubbed) array
  task hit the same entry point; resume works; PC and BPTT states are interface-
  compatible.
- **2.2 Behavior.** `src/behavior/` — tp = first threshold crossing after Set;
  tp-vs-ts slope per prior. A reported covariate, never a filter. *Check:* slope
  matches the reconstruction's `bias_slopes` on a trained net.
- **2.3 RSA.** `src/compare/rsa.py` — condition-averaged RDMs over ts × prior ×
  effector, per-seed distances. Two comparisons, same as iDSA below: rule-vs-rule
  (BPTT RDM vs PC RDM) available in the prototype, and **model-to-DMFC** (with a
  noise ceiling from neural split-halves) — the latter shares Stage 4's dependency
  on validated neural ingestion. *Check:* runs on neural data and on a surrogate/
  other-RNN; dtypes and standardized inputs align in time.
- **2.4 iDSA Stage 3 — BPTT-vs-PC (prototype target, answers RQ1).**
  `src/compare/idsa.py` — DMDc-based input + recurrent operators on both models'
  trajectories **with stored inputs**; per-seed distances between the two rules.
  Requirements: same conditions in both systems, inputs aligned to states,
  identical preprocessing, matched latent dimensionality and time bins, finite and
  reproducible distances. Answers RQ1 (the rule leaves a dynamical signature).
  *Check:* distances finite and reproducible; re-runs agree.
- **2.5 iDSA Stage 4 — model-to-DMFC (CENTRAL deliverable, answers RQ3).** Run
  InputDSA between **each trained model and DMFC**. This is the in-silico/in-vivo
  comparison the project exists for. Depends on (and is sequenced after): validated
  DMFC ingestion (1.D), verified Ready/Set/Go alignment, matched prior/interval/
  effector metadata, a **neural external-input representation consistent with the
  model input**, and identical preprocessing across model and neural activity.
  *Check:* per-model, per-seed distance to DMFC, finite and reproducible, with the
  neural noise ceiling as reference. Not optional — schedule ingestion early so
  this is not stranded at the end.
- **2.6 Across-ts dynamics contrast (RQ2).** A first-class analysis, not an
  afterthought: run the Stage 3 / Stage 4 comparisons **resolved by interval
  length** (per-ts or short-band vs long-band, mirroring the Dump tab's "DSA
  between short- and long-interval trajectories") to test whether the signature
  *depends on ts*. Direct answer to Research Question 2. *Check:* produces a
  signature-vs-ts curve per rule with seed spread.

### Phase 3 — Sweep, orchestrate, visualize

- **3.1 Run entry points + launchers.** `scripts/` entry points callable
  interactively; a local loop launcher for the seed sweep; a thin
  `scripts/slurm/` sbatch wrapper (resources → same entry point). No cluster
  values invented — left as `TODO(cluster)` placeholders.
- **3.2 Sweep configs.** Config grid over rule × `pc_inference_steps` × seed.
  Reduced regime for smoke, faithful for GPU.
- **3.3 Aggregation + plotting** (`src/viz/`, reads saved metrics only, non-
  interactive backend, writes files): behavioral tp-vs-ts with slope (Fig 1E);
  loss curves per seed; PCA trajectories showing prior-support curvature (Fig 7C);
  RDM heatmaps + MDS; **the summary figure — PC vs BPTT distance to DMFC on RSA
  and iDSA with seed spread** (this is what answers the research questions);
  pairwise system-distance matrix.

## Validation / smoke path (what I run before handoff)

End-to-end on the reduced regime (`dt=5, N=160`, few seeds, 2–3 sweep points),
CPU-acceptable, correctness over scale:

1. Task: two-prior input + both context channels; ts/prior/effector labels present.
2. Both trainers run on trivial data; loss decreases; tp finite and ordered.
   **PC-A rule-internal checks pass** before PC-B: PC energy descends during
   relaxation, updates finite/stable, toy-task loss decreases. (BPTT-gradient
   alignment is measured and reported here as an observable, not required.)
3. **InputDSA Stages 1–2:** install/API returns finite results; synthetic-
   trajectory ordering is sane (incl. same-dynamics/different-input judged similar).
4. Neural round-trip via `nlb_tools`; ts/effector labels match `conditions.py`;
   direction averaged to the 20-condition set; per-cell trial counts logged.
5. Store round-trip: states + inputs + metadata survive write/read.
6. Preprocessing yields matched dimensionality and matched time bins.
7. **iDSA Stage 3** (BPTT-vs-PC) and **RSA rule-vs-rule** run on smoke-scale model
   data; distances finite and reproducible.
8. **iDSA Stage 4 + RSA model-to-DMFC** run once ingestion is validated;
   standardized inputs align in time; neural noise ceiling computed.
9. Produce the summary figure (PC vs BPTT distance to DMFC) from smoke-scale
    metrics so the plotting path is proven before the team spends GPU hours.

## Critical path & what to cut first

Minimal presentable result = the summary figure showing **PC vs BPTT distance to
DMFC** on **both RSA and iDSA** (i.e. through **Stage 4** — the model-to-DMFC
comparison is the headline, not a stretch), reduced regime, 2 sweep points, 10
seeds, all 20 conditions (prior × ts × effector). Cut order if the week compresses,
in order: fine sweep granularity (drop to 2 inference-step points) → faithful
`dt=1/N=200` regime (stay reduced) → seed count. **Never cut:** the model-to-DMFC
comparison (Stage 4), iDSA, the effector dimension, ts resolution, no-gating.

Because Stage 4 is the headline and depends on neural ingestion, **do not leave
ingestion to the end.** Sequence 1.D (loader) early in parallel with the model
tracks so the DMFC comparison is not stranded.

Three tasks have no fallback and get attention first:
- **PC-A (1.C)** — the PC port is the single biggest technical risk; if its
  rule-internal correctness checks (energy descent, stable updates, toy-task
  learning) don't pass, PC simply isn't implemented and nothing downstream is
  trustworthy. Validated before PC-B RSG training. Correctness does **not** depend
  on matching BPTT.
- **InputDSA Stages 1–2 (0.6–0.7)** — a core dependency with the shakiest install;
  validated in Phase 0 before anything depends on it.
- **Neural ingestion (1.D)** — Stage 4 (the headline RQ3 result) cannot run until
  this exists and is validated; treat it as critical-path, not a late add-on.
