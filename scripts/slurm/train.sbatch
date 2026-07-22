#!/usr/bin/env bash
#SBATCH --job-name=rsg-train
#SBATCH --partition=compute
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=results/slurm_logs/%x_%A_%a.out
#SBATCH --error=results/slurm_logs/%x_%A_%a.err
# #SBATCH --account=CHANGE_ME   # uncomment + fill in if your cluster ever requires one;
#                               # not currently configured for this cluster/account.

# Thin SLURM wrapper around the SAME entry point (scripts/train.py) an interactive
# run uses -- it only sets resources and dispatches; it must never become a second,
# divergent training implementation (AGENTS.md, "Execution: interactive and SLURM,
# one code path").
#
# One seed per array task: SLURM_ARRAY_TASK_ID becomes --seed (AGENTS.md, "the seed
# sweep is a job array, one seed per task"). A failed or timed-out task is safe to
# requeue alone -- train_one_seed checkpoints (full RNG state) and resumes, and
# re-running an already-completed seed is a cheap no-op.
#
# PREREQUISITE (one-time, before the first submission): create the SLURM log
# directory yourself -- #SBATCH --output/--error paths must exist when the job
# is *submitted*, since the scheduler opens them before this script's body runs:
#   mkdir -p /data/sld/homes/vguigon/dynamics-rsg/results/slurm_logs
#
# Usage (seeds 0..9, defaults: rule=pc, regime=reduced, standalone-or-config-default
# task source, run-dir=results/runs):
#   sbatch --array=0-9 scripts/slurm/train.sbatch
#
# Override wall-time (the #SBATCH --time above is just this wrapper's default):
#   sbatch --array=0-9 --time=12:00:00 scripts/slurm/train.sbatch
#
# Override rule/regime/task-source/run-dir via environment (must both set the var
# AND list it in --export, or use --export=ALL to pass your whole environment):
#   RULE=bptt REGIME=faithful sbatch --array=0-9 --export=ALL,RULE,REGIME scripts/slurm/train.sbatch
#
# Or pin exact hyperparameters from a config file instead of --regime/--rule
# (matches scripts/run_sweep.sh's --config-dir convention; CONFIG wins if set):
#   CONFIG=configs/pc.yaml sbatch --array=0-9 --export=ALL,CONFIG scripts/slurm/train.sbatch
#
# Override cfg.pc_inference_steps (default from config: 20) without a config file:
#   RULE=pc PC_INFERENCE_STEPS=100 sbatch --array=0-9 --export=ALL,RULE,PC_INFERENCE_STEPS scripts/slurm/train.sbatch
#
# --cpus-per-task / --mem above are sized for the reduced regime on this project's
# current CPU-only (no GPU configured on this cluster) architecture -- not a
# cluster-verified requirement, just a reasonable starting point. Raise them for
# faithful-regime (N=200, dt=1) runs if you see them getting killed for memory or
# running slower than expected.

set -euo pipefail

REPO_DIR="/data/sld/homes/vguigon/dynamics-rsg"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"

RULE="${RULE:-pc}"
REGIME="${REGIME:-reduced}"
TASK_SOURCE="${TASK_SOURCE:-}"
RUN_DIR="${RUN_DIR:-results/runs}"
CONFIG="${CONFIG:-}"
PC_INFERENCE_STEPS="${PC_INFERENCE_STEPS:-}"
ACTIVATION_STORE="${ACTIVATION_STORE:-}"

module use /software/sld/modulefiles
module load python/3.11.10

cd "$REPO_DIR"

# Traceability: run_identity() reads GIT_SHA (falls back to `git rev-parse HEAD`
# itself if unset, but setting it here pins the SHA at submission time, before any
# later commits on this checkout could change what a resumed/requeued task would see).
export GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

if [ -n "$CONFIG" ]; then
  cmd=("$VENV_PYTHON" scripts/train.py --seed "$SLURM_ARRAY_TASK_ID" --config "$CONFIG" --run-dir "$RUN_DIR")
else
  cmd=("$VENV_PYTHON" scripts/train.py --seed "$SLURM_ARRAY_TASK_ID" --regime "$REGIME" --rule "$RULE" --run-dir "$RUN_DIR")
fi
if [ -n "$TASK_SOURCE" ]; then
  cmd+=(--task-source "$TASK_SOURCE")
fi
if [ -n "$PC_INFERENCE_STEPS" ]; then
  cmd+=(--pc-inference-steps "$PC_INFERENCE_STEPS")
fi
if [ -n "$ACTIVATION_STORE" ]; then
  cmd+=(--activation-store "$ACTIVATION_STORE")
fi

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-none} array_task=${SLURM_ARRAY_TASK_ID:-none} git=$GIT_SHA"
echo "[slurm] >>> ${cmd[*]}"
exec "${cmd[@]}"
