# dynamics-rsg

**Do learning rules leave a measurable signature on latent geometry and dynamics
in a Bayesian timing task?**

We train one RNN architecture under two learning rules, backpropagation-through-time
(BPTT) and predictive coding (PC), on the two-prior Ready-Set-Go (RSG) interval-timing
task, and ask whether the *rule* changes how closely a network's latent **geometry**
(RSA) and **input-driven dynamics** (iDSA) resemble macaque DMFC. The architecture is
held fixed across the two arms, so any difference is attributable to the learning rule
rather than the architecture.

- **Task:** two-prior RSG, extending NeuroGym's `ReadySetGo-v0`.
- **Neural target:** macaque DMFC recordings (Sohn et al. 2019), via the Neural Latents
  Benchmark dataset `DMFC_RSG` (DANDI 000130).
- **Comparisons:** RSA (geometry) and InputDSA / iDSA (dynamics), against DMFC.
- **Primary question (RQ3):** does the PC-trained network sit closer to DMFC than the
  BPTT one, on both measures?

> **Read first:** [`AGENTS.md`](AGENTS.md) for the project's invariants (they are
> load-bearing), and [`docs/implementation_plan.md`](docs/implementation_plan.md) for
> the full build guide, the locked decisions, and the team division of labor.

## Status

The foundation is built and tested; the domain modules are stubs with an
interface + "definition of done" + reference in each docstring, ready for the team to
implement (see the division of labor in the plan). "Implemented" below means it runs
and is covered by `tests/test_foundation.py`; "stub" means the contract is defined and
the body raises `NotImplementedError`.

| Area | File | State |
| --- | --- | --- |
| Condition schema | `src/conditions.py` | implemented |
| Run config + sweep grid | `src/training/config.py` | implemented |
| Activation store | `src/store/` | implemented (`.npz` backend) |
| Model interface | `src/models/base.py` | implemented |
| Plotting harness | `src/viz/figures.py` | harness implemented; panels stubbed |
| Training entry point | `scripts/train.py` | implemented (BPTT + PC, checkpoint/resume, activation export) |
| Task, BPTT, PC, preprocess, RSA, iDSA, behavior, neural loader, trainer loop | `src/task`, `src/models`, `src/preprocess`, `src/compare`, `src/behavior`, `src/data`, `src/training/trainer.py` | implemented |

## Division of labor (who implements which files)

Foundation contracts are provided; each of the 5 members owns one module. Every stub
below already exists with its interface, a "definition of done", and a reference in
the docstring. Implementing it means filling the `TODO(<track>)` seams. See
[`docs/implementation_plan.md`](docs/implementation_plan.md) → "Team & division of
labor" for the full table (learning goals, difficulty, plan-task numbers).

| Track | Files to implement | Plan tasks |
| --- | --- | --- |
| Task & behavior | `src/task/rsg.py`, `src/behavior/slope.py` | 1.A, 2.2 |
| BPTT RNN | `src/models/bptt_rnn.py` | 1.B |
| PC RNN *(pair)* | `src/models/pc_rnn.py` | 1.C, 2.1 (PC-B) |
| Preprocess & RSA | `src/preprocess/pipeline.py`, `src/compare/rsa.py` | 1.E, 2.3 |
| iDSA | `src/compare/idsa.py` | 2.4, 2.5, 2.6 |

Provided by the foundation (agent-owned): the contracts (`src/conditions.py`,
`src/training/config.py`, `src/store/`, `src/models/base.py`), the viz harness
(`src/viz/figures.py`), and the entry point (`scripts/train.py`) are implemented and
tested; the neural loader (`src/data/build_neural.py`) and the shared trainer
(`src/training/trainer.py`) are implemented, including restart-safe checkpoints,
per-seed metrics, and aligned condition activation export.

## Repository layout

```
src/
  conditions.py       # ts x prior x effector schema: the single source of truth (20 conditions)
  task/               # two-prior RSG task (extends NeuroGym ReadySetGo-v0)
  models/             # base interface + bptt_rnn + pc_rnn (share one forward())
  training/           # config, rule-agnostic trainer (one seed per invocation)
  behavior/           # tp-vs-ts slope: a reported covariate, never a filter
  data/               # DMFC_RSG (DANDI 000130) -> data/processed/ (isolated env)
  store/              # activation store keyed by (model, seed, condition): states + inputs + meta
  preprocess/         # identical normalization / PCA-to-shared-k / matched time bins
  compare/            # rsa.py (geometry) + idsa.py (input-driven dynamics)
  viz/                # figures read saved metrics; never retrain
scripts/              # entry points (train.py); interactive == SLURM, one code path
tests/                # test_foundation.py: contracts smoke tests (no torch needed)
docs/                 # implementation_plan.md (build guide + team split)
data/ results/        # inputs and outputs (processed tensors, checkpoints, figures)
```

## Environments

There are two dependency sets, kept separate because the neural-ingestion stack
(`dandi`/`pynwb`/`nlb_tools`) and the modeling/similarity stack (`torch`/`rsatoolbox`/DSA)
pin conflicting `numpy`/`scipy` ranges. See `AGENTS.md` → "Dependency fragility".

**1. Main environment** (modeling + similarity + everything in `src` except `src/data`):

```bash
conda create -n rsg python=3.10 -y && conda activate rsg
pip install -r requirements.txt
pip install -e .                     # makes `import src...` work everywhere
# InputDSA is research code, installed from git (pulls kooplearn + pot):
pip install "git+https://github.com/mitchellostrow/DSA.git"
```

**2. Ingestion environment** (only runs `src/data/build_neural.py`, writes
`data/processed/`):

```bash
conda create -n rsg-ingest python=3.10 -y && conda activate rsg-ingest
pip install -r requirements-ingestion.txt
```

Try a single combined env first; only split off the ingestion env if the combined
install fails to resolve `numpy`/`scipy`. Either way, modeling code never imports
`pynwb`/`dandi`; it reads the tensors in `data/processed/`.

> **Note on the store backend:** the activation store uses a numpy `.npz` backend, not
> HDF5, because the shared anaconda `h5py` has a numpy-ABI break. Nothing extra to
> install; an h5py/zarr backend can drop in later behind the same API.

## Running things

All commands are run from the repo root. The contracts and entry-point dry-run need
no torch, so you can sanity-check them before the full install finishes.

```bash
# See the 20 conditions (prior x ts x effector), overlap at 800 ms marked:
python -m src.conditions

# Foundation smoke tests (conditions, config round-trip, store round-trip):
python tests/test_foundation.py          # or: pytest tests/

# Build a run config and print the plan WITHOUT training (no torch needed):
python scripts/train.py --regime reduced --rule pc --seed 3 --dry-run

# Train one seed (needs the main environment with torch + NeuroGym):
python scripts/train.py --regime reduced --rule bptt --seed 0
python scripts/train.py --config configs/pc.yaml --seed 0

# Ingest the DMFC neural data (in the ingestion env; stub until 1.D lands):
python -m src.data.build_neural
```

### Regimes

- `--regime reduced` (`dt=5, N=160`): CPU-friendly, for smoke tests / development. Will
  under-train, so expect flat behavior; that's expected, not a bug.
- `--regime faithful` (`dt=1, N=200`): the paper-faithful setting for the real GPU runs.

### The seed sweep

Model-to-brain similarity is sensitive to initialization, so seeds are the unit of
evidence: always report the spread, never one network. The sweep is a
`(rule × pc_inference_steps × seed)` grid built by `sweep_configs()` in
`src/training/config.py`, run as one seed per invocation (a SLURM array task and an
interactive run are the same entry point). The expensive full sweeps run on the team's
GPUs; `scripts/slurm/` (thin sbatch wrappers) is a `TODO(cluster)` until partition /
account / wall-time are known.

## Results figures

Colors are defined once, in `src/viz/palette.py`. `RDM_CMAP` is the blue-to-red ramp for
every heatmap (near is blue, far is red); `ARM_COLORS` gives one fixed color per arm (BPTT
blue, PC(20) yellow, PC(100) orange, RFLO red, untrained gray). Every function in
`src/viz/figures.py` imports from there, so a color change happens in one place.

Two plotting drivers read saved metrics only and retrain nothing:

```bash
# RQ1-RQ3 figures from cached metrics (results/signature/signature.json).
# Also creates the two RQ3 headline figures, results/{rsa,idsa}/summary_dmfc_comparison.png:
python scripts/plot_slide_figures.py

# Behavior and best-loss two-panel figure from results/runs_summary.csv:
python scripts/plot_results_summary.py
```

For the slide-by-slide walkthrough (what each figure shows, the key number, and a line to
say), see [`presentation_results.md`](presentation_results.md).

Still pending: the Setup RDM heatmap (`results/figures/dmfc/dmfc_rdm_heatmap.png`) and the
per-arm RDM galleries are not in the shared palette yet, because they need the raw 20x20
condition RDMs, i.e. `data/processed/`. Wherever that data is present, rerun
`scripts/run_rsa_geometry.py` and they pick up `RDM_CMAP` automatically.

## Key invariants (from `AGENTS.md`)

These are non-negotiable; breaking one silently invalidates the comparison:

- **One condition schema.** `src/conditions.py` is imported by both the task and the
  neural loader; conditions are never redefined locally.
- **Identical preprocessing.** Model and neural data pass through the *same*
  `src/preprocess` steps before RSA/iDSA.
- **Seeds are the unit of evidence.** Report the seed spread with CIs, never a point
  estimate.
- **Behavior is measured, never a filter.** Every seed enters RSA/iDSA; the tp-vs-ts
  slope is reported alongside, never used to exclude a seed.
- **iDSA, not plain DSA.** RSG is input-driven (Ready/Set pulses), so we demix
  input-driven from recurrent structure before comparing.

## References

- Sohn, Narain, Meirhaeghe & Jazayeri (2019), *Bayesian computation through cortical
  latent dynamics*, Neuron. https://doi.org/10.1016/j.neuron.2019.06.012
- Neural data: DANDI 000130. https://dandiarchive.org/dandiset/000130
- Neural Latents Benchmark (`DMFC_RSG`). https://neurallatents.github.io/datasets.html
- NeuroGym. https://github.com/neurogym/neurogym
- DSA / InputDSA. https://github.com/mitchellostrow/DSA
- rsatoolbox. https://rsatoolbox.readthedocs.io/
- Predictive coding. https://github.com/BerenMillidge/PredictiveCodingBackprop ·
  https://github.com/Bogacz-Group/PredictiveCoding
