#!/usr/bin/env bash
# Single-command environment activation (bash/zsh). Loads the module stack and
# activates the repo-local .venv -- matches this cluster's convention (module
# system + local venv, not conda).
#
# Must be SOURCED, not executed, so the activation persists in your shell:
#   source scripts/hpc/load_env.sh
#
# Run scripts/hpc/setup_env.sh first if .venv doesn't exist yet.

REPO_DIR="/data/sld/homes/vguigon/dynamics-rsg"

module use /software/sld/modulefiles
module load python/3.11.10
source "$REPO_DIR/.venv/bin/activate"
