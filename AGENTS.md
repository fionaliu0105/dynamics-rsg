# AGENTS.md


<!-- SUPLEX ROOT GOVERNANCE START -->
## SUPLEX Governance

This repository has a SUPLEX control layer installed.

Before inspecting, editing, running commands, or otherwise operating on the repo, an agent must resolve whether it is acting as `supervision` or `execution`.

Read:
- `SUPLEX.md`
- `.suplex/AGENTS.md`
- `.suplex/CLAUDE.md` when applicable
- `.suplex/handoffs/active/current_handoff.md`

SUPLEX governs bounded supervision/execution workflow. Existing project instructions in this file still apply unless they conflict with SUPLEX role routing, handoff scope, or execution boundaries.
<!-- SUPLEX ROOT GOVERNANCE END -->

Guidance for coding agents working in `dynamics-rsg`. Read this before making changes.

This file is agent-agnostic and `CLAUDE.md` is a symlink to it. Edit `AGENTS.md`; never replace the symlink with a copy.

## What this project is

We train recurrent networks on the two-prior Ready-Set-Go (RSG) interval-timing task and ask whether the **learning rule** leaves a measurable signature on latent geometry and dynamics — and whether a predictive-coding (PC) network sits closer to macaque DMFC than a backprop-through-time (BPTT) network trained on the same task. Architecture is held fixed across the two, so any difference is attributable to the rule rather than the architecture.

Pipeline, in order:

1. Define the task and load the DMFC neural data into one shared condition schema.
2. Train RNNs under each learning rule (BPTT, PC, RFLO) across multiple seeds.
3. Measure behavior per seed (regression of produced intervals toward the prior mean).
4. Extract latents; apply **identical** preprocessing to model and neural data.
5. Compare against DMFC with RSA (geometry) and iDSA (dynamics).
6. Aggregate over seeds and test the primary contrast: PC vs BPTT distance to DMFC.

The validity of the whole thing rests on step 4 and on treating seeds as the unit of evidence. Latents that were not preprocessed identically cannot be compared, and a single network per condition is not a result. Neither is negotiable.

### The third arm: RFLO

The proposal specifies the two-arm PC-vs-BPTT contrast, and step 6 above is still **the primary result**. RFLO (Murray 2019, eLife 8:e43299) was added afterwards as a third arm and should be reported alongside, not in place of, that contrast.

The reason it earns a place is that it turns a binary into an axis. BPTT is nonlocal in time and space; PC is local in space but relaxes value nodes over the whole stored trajectory; RFLO is local in **both** and single-pass online. So "does locality of the learning rule leave a signature on latent geometry" becomes a graded question with three points rather than a two-way comparison.

It obeys the same constraint that makes the original contrast interpretable: it drives the same six parameter tensors on the same architecture. `RFLORNN` subclasses `BPTTRNN` rather than copying its rollout, so forward parity is structural, and its feedback matrix is drawn from a separate RNG stream so seed *N* of every arm starts from bit-identical weights. Do not "simplify" either of those.

Expectation to test, not a fact to design around: RFLO's eligibility trace decays with `tau` (10 ms), against Ready→Set intervals up to 1200 ms, and long-range temporal credit assignment is its documented weak point. If it bites, it should surface as a flatter tp-vs-ts slope on the long prior. That is a result to report next to the similarity — **not** grounds for a behavioral filter, which remains prohibited for every arm.

## Sources and their status

This repo is being built from a planning document whose tabs do not carry equal authority. Getting this wrong is the most likely way to introduce a confident, plausible-looking error.

| Source | Status |
| --- | --- |
| **Proposal tab** | **Authoritative.** Conflicts resolve here. |
| Background / RSG paper tabs | Context and task facts. Trustworthy, not decisive. |
| Reconstructed RNN code (Details tab) | **Unvalidated.** Reference only — see the fenced section below. |
| The doc's own AGENTS.md draft | Superseded by this file. |

Where this file knowingly departs from the proposal, it says so inline and gives the reason. There is currently exactly one such departure: the behavioral gate.

## Target layout

Almost none of this exists yet. **This is the target — build toward it.** Do not invent a parallel structure.

```
src/                     # importable package (see Open items re: name)
  conditions.py          # ts, prior, effector defined ONCE; task + neural loader both import
  task/                  # two-prior RSG; wraps/extends NeuroGym ReadySetGo-v0
  models/                # base interface: forward(inputs) -> outputs, states [trials, time, units]
    bptt_rnn.py          # SHARED forward model; the other arms reuse its rollout
    pc_rnn.py            # predictive-coding RNN (the main technical risk)
    rflo_rnn.py          # RFLO (Murray 2019); subclasses bptt_rnn for forward parity
    local_update.py      # shared normalize+clip for rules that compute own updates
  training/              # shared trainer: config, ONE seed per invocation, checkpoint/resume, logging
  behavior/              # tp-vs-ts slope per prior; a REPORTED metric, not a filter
  data/                  # DANDI 000130 (NWB) -> binned [condition, time, unit] + behavioral tp
  store/                 # activation store (zarr/hdf5) keyed by (model, seed, condition)
  preprocess/            # per-unit normalization, PCA to shared k, matched time bins
  compare/               # rsa.py (RDMs + noise ceiling), idsa.py (DMDc operators)
  viz/                   # figures read saved metrics; never retrain
scripts/                 # entry points, callable interactively or from a batch job
  slurm/                 # sbatch wrappers; thin — set resources, call the same entry point
results/                 # figures, metrics, checkpoints (per seed)
```

Three of these placements are load-bearing:

- `store/` keeps the **input time series**, not just states. iDSA needs inputs aligned to states, and reconstructing them after the fact is error-prone.
- `behavior/` is a sibling of `compare/`, not upstream of it. It describes seeds; it does not admit them.
- `training/` takes **one seed per invocation**. That is what makes an interactive run and a SLURM array task the same code path.

## Invariants

Each rule is followed by why it exists, so it survives a well-meaning refactor.

- **One condition schema.** `conditions.py` defines ts, prior, and effector exactly once. The task generator and the neural loader both import it. Never redefine conditions locally in either — two definitions drift, and the drift is invisible until the comparison is already wrong.

- **Identical preprocessing.** Every system, model and neural, passes through the same `preprocess/` steps (per-unit normalization, projection to a shared latent dimensionality, matched time bins) before RSA or iDSA. Make it structurally hard for the comparison functions to receive unstandardized input. Differences in dimensionality, activation scale, or time base otherwise masquerade as findings.

- **Shared latent interface.** Every model exposes latents as `[condition, time, unit]` on the neural time base. New model variants conform; the comparison code should never special-case a model.

- **Seeds are the unit of evidence.** Model-to-brain similarity is sensitive to initialization and hyperparameters. Never report a single network per condition. Metrics are summarized as a spread over seeds with CIs, never a point estimate.

- **Behavior is measured, never a filter.** Compute the tp-vs-ts slope per prior for every seed and carry it alongside every similarity value. **No seed is excluded from RSA or iDSA on behavioral grounds.**

  *This departs from the proposal's literal wording ("For networks that pass, we compare population activity against DMFC") deliberately. Do not silently re-add a gate.* The reasons come from elsewhere in the proposal itself:
  - Filtering on behavior conditions the comparison on an outcome variable. If BPTT and PC have different pass rates, the surviving seeds differ by more than the learning rule, and the contrast is confounded.
  - *Robustness* makes the seed spread the unit of evidence; gating biases that spread by construction.
  - *Standardization* argues matched behavior does not imply matched computation — which makes behavior precisely the wrong thing to condition membership on.

  The cost is real and worth stating: a network that is not performing Bayesian integration may make its "distance to DMFC" hard to interpret. The answer is to report the slope next to the similarity so interpretation stays possible — not to delete the seed.

- **NeuroGym is the task source of truth.** Custom trial logic extends or wraps the NeuroGym environment (`ReadySetGo-v0` is single-prior with no context input, so it needs extending) rather than defining an independent task specification. The point is that model and animal face matched task statistics; a second, independent spec silently breaks that.

- **iDSA, not plain DSA, for dynamics.** The Ready/Set pulses are strong external drive. DSA targets intrinsic dynamics, and inputs can drive systems with similar dynamical structure differently, making them look dissimilar. Demix input-driven from recurrent structure before comparing.

## Conventions

- **Config-driven runs.** A new experiment is a new config file, not an edited constant. Each run writes config + seed + metrics + checkpoint together.
- **Notebooks are for exploration only.** Anything that produces a result or a figure lives in `src/` and is called from `scripts/`.
- **Plotting reads saved metrics.** Figure code does not retrain or re-extract.
- **Set and log seeds** for torch and numpy. Note where full determinism is not achievable (some CUDA kernels).

## Execution: interactive and SLURM, one code path

Runs must work both interactively and under SLURM ("Models will be trained in local environments and on HPCs"). This is easy to violate by accident and expensive to retrofit.

- **Parity.** The same entry point and the same config produce the same run interactively or under `sbatch`. A batch script sets resources and calls the entry point; it never becomes a second, divergent implementation. If something works only in a notebook, it isn't done.
- **The seed sweep is a job array, one seed per task.** `training/` takes a single seed per invocation and writes per-seed checkpoints and metrics under a run directory keyed by (model, seed, condition). A failed or timed-out seed is then requeued alone, without touching the others. Interactively, the same entry point runs one seed; a loop or a local launcher covers the sweep.
- **Assume the process can die.** Wall-time limits and preemption are normal, not exceptional. Checkpoint periodically and resume from the latest checkpoint rather than restarting. Make re-running a completed seed a cheap no-op, so requeueing an array is safe.
- **No interactive-only assumptions.** No display or `plt.show()` on a compute node — write figures to files and use a non-interactive matplotlib backend. No prompts, no `tqdm` as a progress contract, no reliance on the current working directory. Paths come from config or environment, never hardcoded to a laptop or a home directory: `data/` and `results/` sit in different places on a cluster.
- **Log what identifies the run.** Config, seed, git SHA, hostname, device, and array task ID (when present) go into the run's log, so a result found later on scratch can be traced back to what produced it.

The dependency split below has to hold on a cluster too, where the data-ingestion environment may not be the one a job loads.

## Inherited from the unvalidated reconstruction — verify before relying

The planning doc contains a reconstructed RSG RNN. Its own docstring states it was not quality-assessed, its optimizer is a substitution (Adam over BPTT, where the paper used Hessian-free), and it reports that convergence to Bayesian behavior was never demonstrated. **Treat everything in this section as a lead to verify, not a setting to adopt.** Nothing here is project fact, and none of it belongs outside this section until verified.

- **Constants that disagree between the paper text and the saved network:** ramp shape `a=2.8` vs `eta=3.3`; ramp amplitude `A=3` vs `A=2.85`; output threshold `z=1` vs `0.99`; train/test `w_m` swept rather than fixed. If used, expose each as config and record the choice. Never hardcode.
- **The scalar-noise-on-the-input trick:** jittering the Ready-Set separation, `t_m ~ N(ts, ts*w_m)`, against a target timed to the *true* ts — the reconstruction's proposed mechanism for the Bayesian bias. Plausible and paper-adjacent, but unverified here, and it must be reconciled with NeuroGym-as-task-source before adoption rather than bolted on.
- **Regime claims:** `dt=1`/`N=200` on GPU as the faithful setting vs `dt=5`/`N=160` as plumbing-only, and the ~40% Bayes-consistent seed rate. Treat as expectations to test, not facts to design around.

## Dependency fragility

torch, neurogym, rsatoolbox, the DSA/InputDSA repo (research code installed from git; pulls in kooplearn and pot), and the neural-data stack (dandi, pynwb, nlb_tools) do not all pin compatible numpy/scipy ranges.

If a fresh install fails on numpy/scipy resolution, **do not force-upgrade a shared pin to satisfy one package.** The sanctioned escape hatch is to isolate data ingestion (dandi/pynwb/nlb_tools) in its own environment that writes processed tensors to `data/processed/`, keeping modeling and similarity code in the main environment. Record any pin you change and why.

## Open items — flag, do not resolve unilaterally

- **Seed count per condition.** Still `[N]` in the proposal.
- **PC target:** the exact continuous-time RNN, or a matched-but-distinct architecture. A distinct architecture (e.g. tPCN) can only support a secondary learning-*system* comparison — architecture would co-vary with the rule, so it cannot carry a causal claim about the learning rule alone.
- **Package name.** `setup.py` names the package `src`; the planning doc's Startup tab documents `python -m dynamics_rsg.data.build_neural`. These disagree today. Raise it rather than picking a side mid-module.
- **Cluster specifics: unknown.** Partition, account, wall-time limits, GPU type, whether the environment comes from `module load` or conda, and where scratch lives are all unspecified. The execution contract above holds regardless. Do not invent these values or bake a guessed partition into a script; fill them in once someone confirms the actual cluster.

## Do not

- Change binning, alignment, or averaging on one side (model or neural) without changing the other.
- Re-introduce a behavioral filter, in any form, to decide which networks enter the comparison.
- Compare latents that were not passed through shared preprocessing.
- Report point estimates without the seed spread.
- Treat reconstruction constants as validated, or move them out of their section.
- Write a batch-only or notebook-only code path, or hardcode paths or cluster values.
- Add dependencies that force a downgrade of torch, numpy, or scipy without checking that the similarity and data stacks still import.

## Reference

- Original paper: Sohn, Narain, Meirhaeghe & Jazayeri (2019), *Bayesian computation through cortical latent dynamics*, Neuron — https://doi.org/10.1016/j.neuron.2019.06.012
- Neural data: DANDI dandiset 000130 — https://dandiarchive.org/#/dandiset/000130
- Neural Latents Benchmark (DMFC_RSG) — https://neurallatents.github.io/datasets.html · nlb_tools: https://github.com/neurallatents/nlb_tools
- Task: NeuroGym — https://github.com/neurogym/neurogym
- Dynamics: DSA / InputDSA — https://github.com/mitchellostrow/DSA · InputDSA paper: https://arxiv.org/abs/2510.25943
- Geometry: rsatoolbox — https://rsatoolbox.readthedocs.io/en/stable/
- Predictive coding: https://github.com/BerenMillidge/PredictiveCodingBackprop · https://github.com/Bogacz-Group/PredictiveCoding

The planning document (Proposal tab) is the source of record for the research design.
