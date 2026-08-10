#!/usr/bin/env bash
# Three objectives, to find out whether one scalar is enough to explain the agent.
#
#   setsid nohup bash scripts/three_objective.sh > logs/run.log 2>&1 &
#
# With two objectives the choice depends on a single difference, so a value slot
# per objective and a single threshold on the distance gap make identical
# predictions — which is why ``015`` came out undecided. With three objectives
# the choice depends on two independent differences, so the two hypotheses
# differ in *rank*: one shared knob spans one dimension, per-objective values
# span two. That is measurable.
#
# Stages skip whatever is already on disk, so an interruption costs only the
# stage it happened in. The base agent is gated on competence before any of the
# grid runs: a sweep around an agent that cannot reach an objective would
# produce numbers that mean nothing, at hours of GPU each.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
cd "$(dirname "$0")/.."

BASE="${BASE:-/workspace/data/threeobj}"
STEPS="${STEPS:-80000000}"
ARM_STEPS="${ARM_STEPS:-1000000}"
LR="${LR:-1e-4}"
N_LEVELS="${N_LEVELS:-1000000}"

# Arms are far smaller than the base dataset because they are far shorter runs:
# 1M steps is about 100k episodes, so 140k training levels are already more than
# one apiece and memorisation is not on the table. Three objectives cost 1.75 ms
# a level against 0.42 for two — three breadth-first searches and a mutual
# reachability check rather than two — so a 500k dataset per arm would have put
# nearly three hours of generation in front of the training run.
ARM_LEVELS="${ARM_LEVELS:-150000}"

# Evenly spaced, 0.35 apart, so each neighbouring pair is worth 7 extra steps at
# the task's 0.05 a step — the same order as the two-objective task, which kept
# 39% of episodes as genuine trade-offs at this maze size.
BASE_VALUES="1.0 0.65 0.3"

# Each arm moves one objective's value and leaves the others alone. Offsets are
# symmetric about the base, unlike the first grid: with offsets balanced around
# zero the common fine-tuning component cannot leak into the fitted axis in the
# first place, rather than having to be regressed out afterwards.
GRID=(
    "o0_080 0.8  0.65 0.3"
    "o0_090 0.9  0.65 0.3"
    "o0_110 1.1  0.65 0.3"
    "o0_120 1.2  0.65 0.3"
    "o1_045 1.0  0.45 0.3"
    "o1_055 1.0  0.55 0.3"
    "o1_075 1.0  0.75 0.3"
    "o1_085 1.0  0.85 0.3"
    "o2_010 1.0  0.65 0.1"
    "o2_020 1.0  0.65 0.2"
    "o2_040 1.0  0.65 0.4"
    "o2_050 1.0  0.65 0.5"
)

mkdir -p "${BASE}/levels" "${BASE}/runs" "${BASE}/logs" "${BASE}/results"

echo "=== levels ==="
if [ ! -d "${BASE}/levels/base" ]; then
    echo "  base  (${BASE_VALUES})  ${N_LEVELS} levels"
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
    echo "  training ${STEPS} steps -> ${BASE}/logs/base.log"
    uv run python experiments/001_maze_repro.py \
        --levels "${BASE}/levels/base" \
        --n-objectives 3 --objective-values ${BASE_VALUES} \
        --min-size 11 --max-size 11 \
        --total-timesteps "${STEPS}" \
        --hide-values \
        --run-dir "${BASE}/runs/base" \
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
    echo "BASE_AGENT_WEAK -- stopping before the grid, as agreed."
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
echo THREE_OBJECTIVE_COMPLETE
