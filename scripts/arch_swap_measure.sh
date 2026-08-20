#!/usr/bin/env bash
# Measure one architecture-swap agent the way the DRC agents were measured.
#
#   bash scripts/arch_swap_measure.sh resnet11.s1234 [DATA_DIR]
#
# On the run's final checkpoint: 002 (which objective it chooses, 2048
# episodes at rho 1.0/0.5/0.0 on the test split, JSON beside the DRC's in
# figures/data/) and 003 (the t=0 plan probe against the untrained network and
# the observation, per layer, fitted at rho=1.0 and scored at rho=1.0 and 0.0).
# Then the early-warning sweep over every checkpoint, which is what says
# whether the probe moved before the behaviour did.
#
# Safe to run beside a training job on the same GPU: the evaluator does not
# preallocate. Writes results/arch-swap-<agent>.txt and
# results/arch-swap-early-warning-<agent>.txt in the checkout it runs from.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.15}"
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."

AGENT="${1:?agent run name, e.g. resnet11.s1234}"
DATA="${2:-/workspace/data}"
LEVELS="${LEVELS:-${DATA}/levels/values/1.00-0.50@1M}"
run="${DATA}/runs/${AGENT}"
final="$(ls -d "${run}"/local-files/cp_* | sort -t_ -k2 -n | tail -1)"
OUT="results/arch-swap-${AGENT}.txt"
mkdir -p results figures/data

{
    echo "Architecture swap: ${AGENT}, final checkpoint ${final##*/}"
    echo "Measured $(date -u +%FT%TZ) with scripts/arch_swap_measure.sh; same protocol as the DRC agents."
    echo
    echo "=== 002 behaviour (test split, 2048 episodes per rho) ==="
    uv run python experiments/002_measure_proxy.py "${final}" \
        --levels "${LEVELS}" --episodes 2048 --correlations 1.0 0.5 0.0 \
        --json "figures/data/${AGENT}.json"
    echo
    echo "=== 003 plan probe at t=0 (fitted at rho=1.0, scored at rho=1.0 and 0.0) ==="
    uv run python experiments/003_probe_plan.py "${final}" \
        --levels "${LEVELS}" --correlation 1.0 --test-correlations 1.0 0.0 --per-layer --by-distance
} > "${OUT}" 2>&1
echo "wrote ${OUT}"

AGENTS="${AGENT}" OUT="results/arch-swap-early-warning-${AGENT}.txt" MAX_STEPS="${MAX_STEPS:-150000000}" \
    bash scripts/early_warning.sh "${DATA}"
