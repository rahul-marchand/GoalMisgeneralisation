#!/usr/bin/env bash
# The three-objective grid again, wide enough to read the weights.
#
#   setsid nohup bash scripts/three_objective_wide.sh > logs/wide.log 2>&1 &
#
# The first grid settled composition behaviourally and said nothing at all about
# the weights: split-half reliability came out at 0.04 to 0.09, so every cosine
# was attenuated past the point of meaning and the corrected ones fell outside
# the range a cosine can take.
#
# Two causes, both fixable without touching the agent. Arms ran 1M steps where
# the two-objective grid ran 3M, and offsets reached 0.2 where that grid reached
# 0.4. Signal grows with the offset and noise does not, so halving the offset
# alone costs fourfold in signal to noise.
#
# So: same base agent, same base values, three times the steps and twice the
# offsets. Objective 2 sits at 0.3 and cannot go four tenths below without
# turning into a punishment, which would be a different task rather than a wider
# grid, so its pairs are 0.1 and 0.25 -- still symmetric, still two pairs, which
# is what the reliability estimate needs.
#
# Datasets are reused from the first grid where the values match. A tag encodes
# its values, so a shared tag means shared values, and the dataset fingerprint
# check verifies that at startup rather than trusting it: a wrong dataset raises
# instead of quietly training an arm on the wrong task.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
cd "$(dirname "$0")/.."

BASE="${BASE:-/workspace/data/threeobj2}"
DONOR="${DONOR:-/workspace/data/threeobj}"
CHECKPOINT="${CHECKPOINT:-$(ls -d "${DONOR}"/runs/base/local-files/cp_* | tail -1)}"
ARM_STEPS="${ARM_STEPS:-3000000}"
LR="${LR:-1e-4}"
ARM_LEVELS="${ARM_LEVELS:-400000}"

GRID=(
    "o0_060 0.6  0.65 0.3"
    "o0_080 0.8  0.65 0.3"
    "o0_120 1.2  0.65 0.3"
    "o0_140 1.4  0.65 0.3"
    "o1_025 1.0  0.25 0.3"
    "o1_045 1.0  0.45 0.3"
    "o1_085 1.0  0.85 0.3"
    "o1_105 1.0  1.05 0.3"
    "o2_005 1.0  0.65 0.05"
    "o2_020 1.0  0.65 0.2"
    "o2_040 1.0  0.65 0.4"
    "o2_055 1.0  0.65 0.55"
    "m_120_045_030 1.2 0.45 0.3"
    "m_080_085_030 0.8 0.85 0.3"
    "m_100_085_010 1.0 0.85 0.1"
    "m_120_065_010 1.2 0.65 0.1"
    "m_120_045_050 1.2 0.45 0.5"
)

mkdir -p "${BASE}/levels" "${BASE}/runs" "${BASE}/logs" "${BASE}/results"
echo "base checkpoint ${CHECKPOINT}"

echo
echo "=== levels ==="
for entry in "${GRID[@]}"; do
    set -- ${entry}
    tag=$1; shift
    if [ -e "${BASE}/levels/${tag}" ]; then
        echo "  ${tag} present"
    elif [ -d "${DONOR}/levels/${tag}" ]; then
        ln -s "${DONOR}/levels/${tag}" "${BASE}/levels/${tag}"
        echo "  ${tag} reused from the first grid"
    else
        echo "  ${tag}  ($*)"
        uv run python scripts/generate_levels.py \
            --n-levels "${ARM_LEVELS}" --min-size 11 --max-size 11 \
            --valid-levels 5000 --test-levels 5000 \
            --n-objectives 3 --objective-values "$@" \
            --out "${BASE}/levels/${tag}" >> "${BASE}/logs/generate.log" 2>&1
    fi
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
echo "=== analysis ==="
uv run python experiments/016_three_objective_values.py \
    --base "${CHECKPOINT}" \
    --arms "${BASE}/runs" \
    --levels "${DONOR}/levels/base" \
    > "${BASE}/results/three-objective-wide.txt" 2>&1
tail -30 "${BASE}/results/three-objective-wide.txt"

echo
echo WIDE_COMPLETE
