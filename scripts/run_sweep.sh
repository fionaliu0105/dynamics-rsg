#!/usr/bin/env bash
# Local seed sweep: loop rules x seeds, one `scripts/train.py` invocation per seed
# (the local equivalent of a SLURM job array, one seed per task). Each seed is
# independent, so a failure does NOT abort the sweep -- failures are collected and
# reported at the end, and re-running is a cheap no-op once the trainer's
# checkpoint/resume lands (AGENTS.md, "Assume the process can die").
#
# Two ways to pick hyperparameters:
#   --regime <reduced|faithful>   flag-driven (reduced = CPU smoke; no config file)
#   --config-dir <dir>            pin exact hyperparameters from <dir>/<rule>.yaml
# Exactly one of the two; --config-dir wins if both are given.
#
# Examples:
#   bash scripts/run_sweep.sh --regime reduced --rules "bptt pc" --seeds "0 1 2 3 4"
#   bash scripts/run_sweep.sh --config-dir configs --rules "bptt pc" --seeds "$(seq 0 9)"
#   bash scripts/run_sweep.sh --regime reduced -- --dry-run   # loop mechanics, no training
#
# Anything after `--` is passed straight through to scripts/train.py (e.g. --dry-run).
# Run from the repo root in an env with torch (+ neurogym for the default task
# source); e.g. `conda activate rsg`. Override the interpreter with PYTHON=...
set -u

RULES="bptt pc"
SEEDS="0 1 2 3 4"
REGIME=""
CONFIG_DIR=""
TASK_SOURCE=""
RUN_DIR="results/runs"
PYTHON="${PYTHON:-python}"
PASSTHROUGH=()

while [ $# -gt 0 ]; do
  case "$1" in
    --rules)        RULES="$2"; shift 2 ;;
    --seeds)        SEEDS="$2"; shift 2 ;;
    --regime)       REGIME="$2"; shift 2 ;;
    --config-dir)   CONFIG_DIR="$2"; shift 2 ;;
    --task-source)  TASK_SOURCE="$2"; shift 2 ;;
    --run-dir)      RUN_DIR="$2"; shift 2 ;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# //'; exit 0 ;;
    --)             shift; PASSTHROUGH=("$@"); break ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$CONFIG_DIR" ] && [ -z "$REGIME" ]; then
  REGIME="reduced"   # default: CPU smoke
fi

# Traceability: expose the git SHA so the trainer can log it (run_identity reads GIT_SHA).
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
export GIT_SHA

echo "[sweep] rules=[$RULES] seeds=[$SEEDS] $( [ -n "$CONFIG_DIR" ] && echo "config-dir=$CONFIG_DIR" || echo "regime=$REGIME" ) git=$GIT_SHA python=$PYTHON"

failures=""
n_ok=0
for rule in $RULES; do
  for seed in $SEEDS; do
    cmd=("$PYTHON" scripts/train.py --seed "$seed" --run-dir "$RUN_DIR")
    if [ -n "$CONFIG_DIR" ]; then
      cfg="$CONFIG_DIR/$rule.yaml"
      if [ ! -f "$cfg" ]; then echo "[sweep] MISSING config $cfg" >&2; failures="$failures $rule:$seed(no-config)"; continue; fi
      cmd+=(--config "$cfg")
    else
      cmd+=(--regime "$REGIME" --rule "$rule")
    fi
    [ -n "$TASK_SOURCE" ] && cmd+=(--task-source "$TASK_SOURCE")
    [ ${#PASSTHROUGH[@]} -gt 0 ] && cmd+=("${PASSTHROUGH[@]}")

    echo "[sweep] >>> ${cmd[*]}"
    if "${cmd[@]}"; then
      n_ok=$((n_ok + 1))
    else
      echo "[sweep] FAILED rule=$rule seed=$seed (continuing)" >&2
      failures="$failures $rule:$seed"
    fi
  done
done

echo "[sweep] done: $n_ok ok."
if [ -n "$failures" ]; then
  echo "[sweep] FAILURES:$failures" >&2
  echo "[sweep] re-run just those (safe: resume/no-op once checkpointing lands)." >&2
  exit 1
fi
