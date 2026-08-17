#!/usr/bin/env bash
# Three objectives again, on values that one number cannot solve.
#
#   setsid nohup bash scripts/three_objective_v2.sh > run.log 2>&1 &
#
# Both earlier three-objective grids produced three collinear axes — one shared
# dial rather than the two degrees of freedom a three-way choice needs. That was
# the task, not the agent.
#
# The decision rule is three distance thresholds, in steps:
#
#     tau_ij = (v_i - v_j) / step_penalty
#
# of which only two are independent, since tau_02 = tau_01 + tau_12. With
# correlation 1.0 the colour channels hand the agent the *ordering* for free, so
# what it has to store is the magnitudes. And with the old values (1.0, 0.65,
# 0.3) those were an arithmetic progression: gaps of 0.35 and 0.35, so
#
#     tau_ij = (rank gap) x (one constant)
#
# One stored constant, multiplied by a rank gap it reads off the observation,
# solves the whole task. The agent learned one dial, and a fine-tune can only
# move the dial that exists — hence rank one.
#
# These values are (1.0, 0.55, 0.4): gaps of 0.45 and 0.15, so tau_01 = 9 steps
# and tau_12 = 3 steps while both rank gaps are 1. The threshold now depends on
# *which* pair rather than on how far apart in rank, so no single constant
# generates them and two numbers must be stored.
#
# It also keeps both independent comparisons well exercised. Measured over 4000
# levels, the deciding pair is 0v1 on 58.6% of episodes and 0v2 on 34.1%. 1v2 is
# rare at 7.2% and that is fine: it is the dependent one.
#
# Arms are 750k steps rather than 3M. Split-half reliability falls monotonically
# with arm length — 0.228 at 750k against 0.144 at 3M — because behaviour
# converges early and everything after that accumulates arm-specific movement
# without adding signal.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
cd "$(dirname "$0")/.."

BASE="${BASE:-/workspace/data/runs/threeobj.uneven.s1234}"
STEPS="${STEPS:-80000000}"
ARM_STEPS="${ARM_STEPS:-750000}"
LR="${LR:-1e-4}"
N_LEVELS="${N_LEVELS:-1000000}"
ARM_LEVELS="${ARM_LEVELS:-150000}"

BASE_VALUES="1.0 0.55 0.4"

# Objectives 0 and 1 sweep +/-0.2 and +/-0.4. Objective 2 sits at 0.4 and takes
# +/-0.15 and +/-0.3, since 0.4 - 0.4 would be a worthless objective and a
# different task rather than a wider grid. Every objective keeps two symmetric
# pairs, which is what the reliability estimate needs.
GRID=(
    "o0_060 0.6  0.55 0.4"
    "o0_080 0.8  0.55 0.4"
    "o0_120 1.2  0.55 0.4"
    "o0_140 1.4  0.55 0.4"
    "o1_015 1.0  0.15 0.4"
    "o1_035 1.0  0.35 0.4"
    "o1_075 1.0  0.75 0.4"
    "o1_095 1.0  0.95 0.4"
    "o2_010 1.0  0.55 0.1"
    "o2_025 1.0  0.55 0.25"
    "o2_055 1.0  0.55 0.55"
    "o2_070 1.0  0.55 0.7"
    "m_120_035_040 1.2 0.35 0.4"
    "m_080_075_040 0.8 0.75 0.4"
    "m_100_075_025 1.0 0.75 0.25"
    "m_120_055_025 1.2 0.55 0.25"
    "m_120_035_055 1.2 0.35 0.55"
)

mkdir -p "${BASE}/levels" "${BASE}/runs" "${BASE}/logs" "${BASE}/results"

echo "=== levels ==="
if [ ! -d "${BASE}/levels/base" ]; then
    echo "  base (${BASE_VALUES})"
    uv run python scripts/generate_levels.py \
        --n-levels "${N_LEVELS}" --min-size 11 --max-size 11 \
        --valid-levels 50000 --test-levels 50000 \
        --n-objectives 3 --objective-values ${BASE_VALUES} \
        --out "${BASE}/levels/base" >> "${BASE}/logs/generate.log" 2>&1
else
    echo "  base present"
fi
for entry in "${GRID[@]}"; do
    set -- ${entry}
    tag=$1; shift
    if [ -d "${BASE}/levels/${tag}" ]; then echo "  ${tag} present"; continue; fi
    echo "  ${tag}  ($*)"
    uv run python scripts/generate_levels.py \
        --n-levels "${ARM_LEVELS}" --min-size 11 --max-size 11 \
        --valid-levels 5000 --test-levels 5000 \
        --n-objectives 3 --objective-values "$@" \
        --out "${BASE}/levels/${tag}" >> "${BASE}/logs/generate.log" 2>&1
done

echo
echo "=== base agent ==="
if compgen -G "${BASE}/runs/base/local-files/cp_*" > /dev/null; then
    echo "  already trained"
else
    uv run python experiments/001_maze_repro.py \
        --levels "${BASE}/levels/base" \
        --n-objectives 3 --objective-values ${BASE_VALUES} \
        --min-size 11 --max-size 11 --total-timesteps "${STEPS}" --hide-values \
        --run-dir "${BASE}/runs/base" \
        --note "Three objectives at (1.0, 0.55, 0.4). Unequal gaps, so the pairwise
thresholds are 9 and 3 steps while both rank gaps are 1, and no single stored
constant can generate them. Replaces (1.0, 0.65, 0.3), whose even spacing made
the task solvable with one number and produced three collinear axes." \
        > "${BASE}/logs/base.log" 2>&1
fi

CHECKPOINT=$(ls -d "${BASE}"/runs/base/local-files/cp_* | tail -1)
echo "  base checkpoint ${CHECKPOINT}"

echo
echo "=== is the base agent worth sweeping around? ==="
if ! uv run python scripts/competence.py "${CHECKPOINT}" \
        --levels "${BASE}/levels/base" \
        --n-objectives 3 --objective-values ${BASE_VALUES} \
        --episodes 2048 --min-reached 95 \
        2>&1 | tee "${BASE}/results/base-competence.txt"; then
    echo "BASE_AGENT_WEAK -- stopping before the grid."
    exit 1
fi

echo
echo "=== arms ==="
for entry in "${GRID[@]}"; do
    set -- ${entry}
    tag=$1; shift
    if compgen -G "${BASE}/runs/${tag}/local-files/cp_*" > /dev/null; then
        echo "  ${tag} already has a checkpoint"; continue
    fi
    echo "  ${tag}  ($*)"
    rm -rf "${BASE}/runs/${tag}"
    uv run python experiments/013_value_axis.py "${CHECKPOINT}" \
        --objective-values "$@" \
        --levels "${BASE}/levels/${tag}" \
        --run-dir "${BASE}/runs/${tag}" \
        --steps "${ARM_STEPS}" --lr "${LR}" \
        --note "Arm of the (1.0, 0.55, 0.4) grid." \
        > "${BASE}/logs/${tag}.log" 2>&1
    tail -1 "${BASE}/logs/${tag}.log"
done

echo
echo V2_COMPLETE
