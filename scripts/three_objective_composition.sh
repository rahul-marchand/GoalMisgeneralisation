#!/usr/bin/env bash
# Arms with two values moved at once, as held-out tests of composition.
#
#   setsid nohup bash scripts/three_objective_composition.sh > logs/composition.log 2>&1 &
#
# The single-value grid can only show that each objective has an axis. Rank
# cannot settle much on its own: an agent that solves a three-objective task at
# all must depend on two independent differences, so rank two is close to forced
# by the task rather than evidence about how values are held.
#
# Composition is the test that discriminates. If the agent holds a value per
# objective, moving two of them is the sum of moving each, and an arm trained
# with both moved should be reproduced by adding the two axes -- neither of which
# was fitted on it. If instead each configuration was solved on its own terms,
# the sum predicts nothing.
#
# With two objectives this was vacuous, since raising one value and lowering the
# other were the same edit. With three they are not: raising v0 changes both
# (v0-v1) and (v0-v2), while lowering v1 changes (v0-v1) and (v2-v1).
#
# Waits for the single-value run so the two share a base agent and never contend
# for the GPU.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
cd "$(dirname "$0")/.."

BASE="${BASE:-/workspace/data/threeobj}"
ARM_STEPS="${ARM_STEPS:-1000000}"
LR="${LR:-1e-4}"
ARM_LEVELS="${ARM_LEVELS:-500000}"
DEADLINE=$(( SECONDS + ${WAIT_SECONDS:-43200} ))

# Values are carried in the tag so the analysis reads them from the directory
# name rather than from a table that could drift out of step with what was run.
GRID=(
    "m_110_055_030 1.1 0.55 0.3"
    "m_100_075_020 1.0 0.75 0.2"
    "m_090_065_040 0.9 0.65 0.4"
    "m_120_045_030 1.2 0.45 0.3"
)

while ! grep -q THREE_OBJECTIVE_COMPLETE /workspace/data/threeobj_run.log 2>/dev/null; do
    if grep -qE "BASE_AGENT_WEAK|Traceback" /workspace/data/threeobj_run.log 2>/dev/null; then
        echo "single-value run stopped; not composing on top of it"
        exit 1
    fi
    if [ "${SECONDS}" -gt "${DEADLINE}" ]; then
        echo "gave up waiting for the single-value run"
        exit 1
    fi
    sleep 120
done

CHECKPOINT=$(ls -d "${BASE}"/runs/base/local-files/cp_* | tail -1)
echo "base checkpoint ${CHECKPOINT}"

echo "=== levels ==="
for entry in "${GRID[@]}"; do
    set -- ${entry}
    tag=$1; shift
    if [ -d "${BASE}/levels/${tag}" ]; then echo "  ${tag} present"; continue; fi
    echo "  ${tag}  ($*)"
    uv run python scripts/generate_levels.py \
        --n-levels "${ARM_LEVELS}" --min-size 11 --max-size 11 \
        --valid-levels 50000 --test-levels 50000 \
        --n-objectives 3 --objective-values "$@" \
        --out "${BASE}/levels/${tag}" >> "${BASE}/logs/generate.log" 2>&1
done

echo
echo "=== arms ==="
for entry in "${GRID[@]}"; do
    set -- ${entry}
    tag=$1; shift
    if compgen -G "${BASE}/runs/${tag}/local-files/cp_*" > /dev/null; then
        echo "  ${tag} already has a checkpoint"; continue
    fi
    echo "  ${tag}  ($*)  -> ${BASE}/logs/${tag}.log"
    rm -rf "${BASE}/runs/${tag}"
    uv run python experiments/013_value_axis.py "${CHECKPOINT}" \
        --objective-values "$@" \
        --levels "${BASE}/levels/${tag}" \
        --run-dir "${BASE}/runs/${tag}" \
        --steps "${ARM_STEPS}" --lr "${LR}" \
        > "${BASE}/logs/${tag}.log" 2>&1
    tail -1 "${BASE}/logs/${tag}.log"
done

echo
echo COMPOSITION_COMPLETE
