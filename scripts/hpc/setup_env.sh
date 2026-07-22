#!/usr/bin/env bash
# One-time environment setup on the cluster. Creates the repo-local .venv and
# installs the main modeling/comparison environment (requirements.txt) -- this
# is deliberately the ONLY environment this script touches. It does NOT set up
# the separate neural-data-ingestion environment (dandi/pynwb/nlb_tools), which
# has a conflicting numpy/scipy pin range and is kept apart on purpose
# (AGENTS.md "Dependency fragility"; see requirements-ingestion.txt if that's
# ever needed on this cluster).
#
# Run once, from anywhere (this script finds the repo root itself):
#   bash scripts/hpc/setup_env.sh
#
# Safe to re-run: reuses the existing .venv rather than deleting it, and
# `pip install` is idempotent. After it succeeds, activate with a single
# command via scripts/hpc/load_env.sh (bash) or load_env.csh (tcsh) -- do not
# activate manually here, this script only builds the environment.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
echo "[setup_env] repo root: $REPO_DIR"

module use /software/sld/modulefiles
module load python/3.11.10

if [ -d .venv ]; then
  echo "[setup_env] reusing existing .venv"
else
  echo "[setup_env] creating .venv"
  python3 -m venv .venv
fi

.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .

echo "[setup_env] verifying key imports..."
.venv/bin/python - <<'PYEOF'
mods = ["torch", "numpy", "scipy", "matplotlib", "neurogym", "rsatoolbox"]
for name in mods:
    mod = __import__(name)
    version = getattr(mod, "__version__", "(no __version__ attr)")
    print(f"  {name}: {version}")
print("[setup_env] all imports OK")
PYEOF

echo
echo "[setup_env] done. Activate with a single command:"
echo "  bash:  source scripts/hpc/load_env.sh"
echo "  tcsh:  source scripts/hpc/load_env.csh"
