#!/usr/bin/env bash
# Prepare a fresh cloud GPU instance (RunPod, Vast, Lambda, ...) for training.
#
#   bash scripts/setup_cloud.sh [DATA_DIR]
#
# DATA_DIR defaults to /workspace/data, which on RunPod is the persistent
# volume. Everything written there survives pod restarts, so the level dataset
# and checkpoints are paid for once rather than per session.
#
# The level dataset is *generated*, not uploaded: it is a deterministic function
# of the seed and the sampler configuration, so regenerating is reproducible and
# avoids moving files around with credentials that expire at inconvenient times.

set -euo pipefail

DATA_DIR="${1:-/workspace/data}"
LEVELS_DIR="${DATA_DIR}/levels"
N_LEVELS="${N_LEVELS:-1000000}"

cd "$(dirname "$0")/.."

echo "==> Installing dependencies (CUDA JAX)"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
uv sync --extra gpu

echo
echo "==> Verifying JAX sees the GPU"
uv run python - <<'PY'
import sys

import jax

devices = jax.devices()
print(f"jax {jax.__version__}: {devices}")
if not any(device.platform == "gpu" for device in devices):
    sys.exit(
        "No GPU visible to JAX. Training would silently fall back to CPU and run "
        "roughly a thousand times slower, so stopping here."
    )
PY

echo
if [ -d "${LEVELS_DIR}" ]; then
    echo "==> Level dataset already present at ${LEVELS_DIR}, skipping generation"
else
    echo "==> Generating ${N_LEVELS} levels into ${LEVELS_DIR}"
    echo "    (roughly 20-40 minutes; one-time, and reused by every run and every"
    echo "     correlation in a sweep)"
    uv run python scripts/generate_levels.py --n-levels "${N_LEVELS}" --out "${LEVELS_DIR}"
fi

echo
echo "==> Smoke test: the stack fits together"
uv run pytest -q -m slow

cat <<EOF

Ready.

  levels      ${LEVELS_DIR}
  checkpoints ${DATA_DIR}/runs   (point --base-run-dir here so they survive restarts)

Next: profile before committing to a long run. Watch steps/second alongside
GPU and CPU utilisation — if the GPU is idle and the CPUs are pegged, the
bottleneck is the environment rather than the model, and more vCPUs will buy
more than a bigger card.
EOF
