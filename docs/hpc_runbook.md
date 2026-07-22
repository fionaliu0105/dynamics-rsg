# HPC runbook: running training on the cluster

Reference for actually running training on the `sld` cluster, end to end. Everything
here is a thin wrapper around the same entry point (`scripts/train.py`) an interactive
local run uses — see `AGENTS.md`, "Execution: interactive and SLURM, one code path".

**Shell note:** the login shell on this cluster is `tcsh`/`csh`, not bash — commands
below are written for that. If your shell really is bash/zsh, use the `.sh` variants
noted inline and bash's native `VAR=val cmd` syntax instead of `env VAR=val cmd`.

## 1. First-time setup

Clone the repo into the path the scripts assume (`/data/sld/homes/vguigon/dynamics-rsg`):

```tcsh
cd /data/sld/homes/vguigon
git clone https://github.com/fionaliu0105/dynamics-rsg.git
cd dynamics-rsg
git checkout PC_RNN
```

Build the environment (module load + create `.venv` + install `requirements.txt` +
verify key imports). Safe to re-run — reuses the existing `.venv` rather than
recreating it:

```tcsh
bash scripts/hpc/setup_env.sh
```

This only sets up the main modeling/comparison environment. It deliberately does
**not** touch the separate neural-data-ingestion environment (`dandi`/`pynwb`/
`nlb_tools`), which has a conflicting numpy/scipy pin range and is kept apart on
purpose (`AGENTS.md`, "Dependency fragility").

Create the SLURM log directory once — `#SBATCH --output`/`--error` paths must exist
at *submission* time, since the scheduler opens them before the job script runs:

```tcsh
mkdir -p /data/sld/homes/vguigon/dynamics-rsg/results/slurm_logs
```

## 2. Every new session: activate the environment

```tcsh
source scripts/hpc/load_env.csh
```

(bash/zsh: `source scripts/hpc/load_env.sh`)

This loads the `python/3.11.10` module and activates the repo-local `.venv` in one
command — matches this cluster's convention (module system + local venv, not conda).
Must be *sourced*, not executed, so the activation persists in your shell. Your
prompt should show `(.venv)` once it's done.

## 3. Submitting training jobs

The wrapper is `scripts/slurm/train.sbatch`. **One seed per array task**
(`SLURM_ARRAY_TASK_ID` becomes `--seed`) — this is what makes a job array and an
interactive single-seed run the same code path. A failed or timed-out task is safe
to requeue alone: `train_one_seed` checkpoints full RNG state and resumes, and
re-running an already-completed seed is a cheap no-op.

Options are passed as environment variables, forwarded to `sbatch` via `--export`.
In `tcsh`, use `env VAR=val ... sbatch ...` (tcsh has no bash-style inline
`VAR=val cmd` syntax — without `env`, `VAR=val` is parsed as a command name and
fails with "Command not found").

### Standard validation run (both rules, reduced regime, seeds 0-9)

Run BPTT and PC as **separate commands, never combined** in one job:

```tcsh
env RULE=bptt REGIME=reduced sbatch --array=0-9 --export=ALL,RULE,REGIME scripts/slurm/train.sbatch
env RULE=pc   REGIME=reduced sbatch --array=0-9 --export=ALL,RULE,REGIME scripts/slurm/train.sbatch
```

### Full-scale PC `pc_inference_steps=100` test

The still-open lead on PC's flat-output collapse (see `docs/pc_threshold_finding.md`):
giving PC's value-relaxation more steps than the default (`20`) produces a small but
real behavioral effect. This runs the full `n_iter=3000` version of that test.

**Must use a separate `RUN_DIR` *and* a separate `ACTIVATION_STORE` from the
standard PC run** — both use `RULE=pc`, so:
- without a distinct `RUN_DIR`, both would write `results/runs/pc/seed_XXXX/` with
  different configs, and the trainer's config-validation guard correctly rejects
  the second one that shows up (`ValueError: run directory already contains a
  different config`) rather than silently corrupting it;
- without a distinct `ACTIVATION_STORE`, both would *silently* collide anyway —
  the activation store's default location is derived from `RUN_DIR`'s *parent*
  directory (`results/`), which is the same for both runs even once `RUN_DIR`
  itself is fixed, and `ActivationStore` keys saved states only by
  `(model, seed, condition)` with no hyperparameter or run-dir in the key, so one
  run's saved states for a given seed would overwrite the other's with **no
  error at all**.

```tcsh
env RULE=pc REGIME=reduced PC_INFERENCE_STEPS=100 RUN_DIR=results/runs_pc_steps100 ACTIVATION_STORE=results/activations_pc_steps100 sbatch --array=0-9 --export=ALL,RULE,REGIME,PC_INFERENCE_STEPS,RUN_DIR,ACTIVATION_STORE scripts/slurm/train.sbatch
```

### Faithful regime (`dt=1`, `N=200`) instead of reduced

```tcsh
env RULE=bptt REGIME=faithful sbatch --array=0-9 --export=ALL,RULE,REGIME scripts/slurm/train.sbatch
```

Not yet cluster-verified for memory/time — `--cpus-per-task`/`--mem` in the sbatch
script are sized for the reduced regime. Raise them (`--cpus-per-task=N --mem=NG` on
the `sbatch` command line, which overrides the script's `#SBATCH` defaults) if a
faithful-regime task gets killed for memory or times out.

### Pin exact hyperparameters from a config file instead

`CONFIG` wins over `--regime`/`--rule` if both are set:

```tcsh
env CONFIG=configs/pc.yaml sbatch --array=0-9 --export=ALL,CONFIG scripts/slurm/train.sbatch
```

### Override wall-time

The script's own `#SBATCH --time=6:00:00` is just a default; override on the
`sbatch` command line directly (no `env`/`--export` needed, it's an `sbatch` flag,
not an env var read by the script):

```tcsh
sbatch --array=0-9 --time=12:00:00 scripts/slurm/train.sbatch
```

## 4. Flag reference

| Variable / flag | Default | Meaning |
| --- | --- | --- |
| `RULE` | `pc` | Learning rule: `bptt` or `pc`. |
| `REGIME` | `reduced` | `reduced` (`dt=5`, `N=160`, plumbing-scale) or `faithful` (`dt=1`, `N=200`, the regime intended to be reported). |
| `TASK_SOURCE` | config default (`neurogym`) | `neurogym` or `standalone` task data generator. |
| `RUN_DIR` | `results/runs` | Where per-seed checkpoints/metrics are written, keyed by `(rule, seed)`. |
| `ACTIVATION_STORE` | `<RUN_DIR's parent>/activations` | Where per-seed saved states/inputs are written, keyed by `(model, seed, condition)` only — set explicitly whenever running a hyperparameter variant of a rule that already has a standard run, or they'll silently collide (see the `pc_inference_steps=100` example). |
| `CONFIG` | unset | Path to a YAML config file; if set, wins over `RULE`/`REGIME`. |
| `PC_INFERENCE_STEPS` | config default (`20`) | Overrides `cfg.pc_inference_steps`, PC's value-relaxation step count. |
| `--array=N-M` | — | `sbatch` flag (not a script variable): one array task per seed, `N` through `M`. |
| `--time=HH:MM:SS` | `6:00:00` (script default) | `sbatch` flag: wall-time override. |
| `--cpus-per-task` / `--mem` | `4` / `8G` (script defaults) | `sbatch` flags: raise for `faithful` regime if a task gets killed or is slow. |

Every variable must both be set (`env VAR=val ...`) *and* listed in `--export`
(or use `--export=ALL` to forward your whole environment) — `sbatch` does not
forward shell variables to the job by default.

## 5. Checking on jobs

```tcsh
squeue -u $USER
tail -f results/slurm_logs/rsg-train_<jobid>_<taskid>.out
```

`<jobid>` is the number `sbatch` prints on submission; `<taskid>` is the array
index (matches `--seed`).

## 6. Known gotchas hit while setting this up

- **`python: command not found` during `setup_env.sh`.** The `python/3.11.10`
  module on this cluster puts `python3` on `PATH`, not a bare `python`. Fixed in
  `scripts/hpc/setup_env.sh` (uses `python3 -m venv`) — once `.venv` exists, the
  venv itself provides `python` inside `.venv/bin/`, so nothing downstream is
  affected.
- **`source scripts/hpc/load_env.sh` fails with `Command not found` / `Undefined
  variable`.** That error format means your shell is `tcsh`/`csh`, not bash — use
  `load_env.csh` instead.
- **`RULE=bptt sbatch ...` fails with `Command not found`.** Same root cause:
  `tcsh` has no bash-style inline `VAR=val cmd` syntax. Prefix with `env` (see
  above) instead.
- **A second job with the same `RULE` but different hyperparameters (e.g. the
  `pc_inference_steps=100` run launched after the standard `pc` run) fails with
  `ValueError: run directory already contains a different config`.** This is the
  trainer's config-validation guard correctly refusing to let two different
  configs share one run directory — not a bug. Give each distinct hyperparameter
  variant its own `RUN_DIR` (see the `pc_inference_steps=100` example above).
  `sacct` may report "Slurm accounting storage is disabled" on this cluster, so a
  job that vanishes from `squeue` isn't necessarily done successfully — check
  `results/slurm_logs/*.err` directly to see why.
