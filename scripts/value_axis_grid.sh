#!/usr/bin/env bash
# The value grid: one level dataset and one fine-tune per value of colour 1.
#
#   bash scripts/value_axis_grid.sh [BASE_DIR]
#
# Colour 0 is always worth 1.0. Every arm is identical in seed, learning rate and
# number of updates, and the datasets share their layouts, so the only thing that
# differs between arms is what the objectives pay. The v050 arm is the value the
# agent already had, and so measures drift rather than value.
#
# Both stages skip work that is already on disk, so the script can be re-run
# after an interruption without repeating anything or half-writing a dataset.
# Fine-tunes run one at a time: there is one GPU, and arms that shared it would
# not be the same fine-tune.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
cd "$(dirname "$0")/.."

BASE="${1:-/workspace/data/valueaxis}"
CHECKPOINT="${CHECKPOINT:-/workspace/data/runs/novalue11/local-files/cp_140206080}"
STEPS="${STEPS:-3000000}"
LR="${LR:-1e-4}"
N_LEVELS="${N_LEVELS:-500000}"

# Which objective's value the sweep moves. Behaviour depends only on the gap
# between the two, so a sweep of colour 0 covers the same gaps as a sweep of
# colour 1 and must produce the same weight direction if what was found is the
# gap rather than a value.
COLOUR="${COLOUR:-1}"

# Value and directory tag together, rather than computing one from the other,
# so a misplaced rounding cannot silently point an arm at the wrong dataset.
if [ "${COLOUR}" = "1" ]; then
    OTHER=1.0
    GRID=("0.9 v090" "0.8 v080" "0.7 v070" "0.6 v060" "0.5 v050" "0.4 v040" "0.3 v030")
else
    OTHER=0.5
    GRID=("0.6 c060" "0.7 c070" "0.8 c080" "0.9 c090" "1.0 c100" "1.1 c110" "1.2 c120")
fi

mkdir -p "${BASE}/levels" "${BASE}/runs" "${BASE}/logs"

echo "=== levels ==="
for pair in "${GRID[@]}"; do
    set -- ${pair}
    value=$1; tag=$2
    if [ -d "${BASE}/levels/${tag}" ]; then
        echo "  ${tag} present"
        continue
    fi
    if [ "${COLOUR}" = "1" ]; then VALUES="${OTHER} ${value}"; else VALUES="${value} ${OTHER}"; fi
    echo "  ${tag}  (colour ${COLOUR} worth ${value})"
    uv run python scripts/generate_levels.py \
        --n-levels "${N_LEVELS}" --min-size 11 --max-size 11 \
        --valid-levels 50000 --test-levels 50000 \
        --objective-values ${VALUES} \
        --out "${BASE}/levels/${tag}" >> "${BASE}/logs/generate.log" 2>&1
done

echo
echo "=== fine-tunes ==="
for pair in "${GRID[@]}"; do
    set -- ${pair}
    value=$1; tag=$2
    if compgen -G "${BASE}/runs/${tag}/local-files/cp_*" > /dev/null; then
        echo "  ${tag} already has a checkpoint"
        continue
    fi
    echo "  ${tag}  (colour ${COLOUR} worth ${value})  -> ${BASE}/logs/${tag}.log"
    rm -rf "${BASE}/runs/${tag}"
    if [ "${COLOUR}" = "1" ]; then ARM=(--value "${value}"); else ARM=(--value "${OTHER}" --value-zero "${value}"); fi
    uv run python experiments/013_value_axis.py "${CHECKPOINT}" \
        "${ARM[@]}" \
        --levels "${BASE}/levels/${tag}" \
        --run-dir "${BASE}/runs/${tag}" \
        --steps "${STEPS}" --lr "${LR}" \
        > "${BASE}/logs/${tag}.log" 2>&1
    tail -1 "${BASE}/logs/${tag}.log"
done

echo
echo GRID_COMPLETE
