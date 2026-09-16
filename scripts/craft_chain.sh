#!/usr/bin/env bash
# Stage 4 of the Craftax study on one pod: crafting demonstrations (no level
# dataset - the pool is addressed by index), hidden-value bases, value-axis arms
# at shifted values, 027. Same shape as craftax_chain.sh, separate so the two
# can run side by side.
#
#   bash scripts/craft_chain.sh prep        # demos (CPU), then one tmux per seed running `seed N`
#   bash scripts/craft_chain.sh seed N      # base cxcraft15.sN, its arms, 027 on both sweeps
#   bash scripts/craft_chain.sh demos | base N | arms BASE | analysis BASE
#
# Pool layout (seed 0): train = fields 0..N_TRAIN, valid = next N_VALID, test =
# next N_TEST; arm sets take their own ranges beyond those, so nothing overlaps.
set -uo pipefail
ulimit -n "$(ulimit -Hn)"
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
export UV_NO_SYNC="${UV_NO_SYNC:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/workspace/data/jax-cache-craftax}"

cd "$(dirname "$0")/.."
UV="$(command -v uv)"; [ -n "$UV" ] || { echo "NO_UV"; exit 1; }

DATA="${DATA:-/workspace/data}"
DEMOS="${DATA}/craftax/craft/demos"
RUNS="${DATA}/craftax/craft/runs"
RESULTS="${DATA}/craftax/craft/results"
LOGS="${DATA}/logs/craftax"
mkdir -p "${DEMOS}/arms" "${RUNS}" "${RESULTS}" "${LOGS}"

BASE_VALUES="${BASE_VALUES:-2.0 0.5}"
N_TRAIN="${N_TRAIN:-100000}"; N_VALID="${N_VALID:-4096}"; N_TEST="${N_TEST:-4096}"
SEEDS="${SEEDS:-1 2 3}"
BASE_STEPS="${BASE_STEPS:-40000}"
FT_STEPS="${FT_STEPS:-1000}"; FT_LR="${FT_LR:-3e-5}"; FT_WARMUP="${FT_WARMUP:-50}"
EVAL_LEVELS="${EVAL_LEVELS:-512}"; CHECKPOINT_RATIO="${CHECKPOINT_RATIO:-2.0}"  # evaluations decode on a shared GPU; keep them few
ARM_TRAIN="${ARM_TRAIN:-20000}"; ARM_TEST="${ARM_TEST:-2048}"
NAME="${NAME:-cxcraft15}"

tag_values() { echo "$1" | tr '-' ' '; }
arms_list() { ${UV} run python scripts/value_axis_arms.py --steps "${FT_STEPS}" --base-values ${BASE_VALUES}; }

demo() {  # demo START COUNT RHO OUT [extra args]
    local start="$1" count="$2" rho="$3" out="$4"; shift 4
    [ -f "${out}/meta.json" ] && { echo "have ${out}"; return 0; }
    ${UV} run python scripts/generate_craft_demos.py --seed 0 --start "${start}" --count "${count}" --rho "${rho}" --out "${out}" "$@" \
        || { echo "DEMOS_FAILED ${out}"; return 1; }
}

demos() {
    local valid_start=$((N_TRAIN)) test_start=$((N_TRAIN + N_VALID)) arm_start=$((N_TRAIN + N_VALID + N_TEST))
    demo 0 "${N_TRAIN}" 1.0 "${DEMOS}/train.rho100" --split train --objective-values ${BASE_VALUES} || return 1
    demo "${valid_start}" "${N_VALID}" 1.0 "${DEMOS}/valid.rho100" --split valid --objective-values ${BASE_VALUES} || return 1
    demo "${valid_start}" "${N_VALID}" 0.5 "${DEMOS}/valid.rho050" --split valid --objective-values ${BASE_VALUES} || return 1
    demo "${valid_start}" "${N_VALID}" 0.0 "${DEMOS}/valid.rho000" --split valid --objective-values ${BASE_VALUES} || return 1
    demo "${test_start}" "${N_TEST}" 1.0 "${DEMOS}/test.rho100" --split test --objective-values ${BASE_VALUES} || return 1
    # Arm demonstrations: every arm's values on the same fresh range (paired), plus that range at the base values for the null arm.
    local arm_test_start=$((arm_start + ARM_TRAIN))
    arms_list | awk '{print $5}' | sort -u | while read -r tag; do
        demo "${arm_start}" "${ARM_TRAIN}" 1.0 "${DEMOS}/arms/${tag}.train.rho100" --objective-values $(tag_values "${tag}") || return 1
        demo "${arm_test_start}" "${ARM_TEST}" 1.0 "${DEMOS}/arms/${tag}.test.rho100" --objective-values $(tag_values "${tag}") || return 1
    done
}

base() {
    local seed="$1" name="${NAME}.s$1"
    [ -f "${RUNS}/${name}/done.json" ] && { echo "done ${name}"; return 0; }
    echo "$(date -u +%FT%TZ) training ${name}"
    ${UV} run python experiments/023_train_bc.py \
        --demos "${DEMOS}/train.rho100" --hide-values \
        --eval "rho100=${DEMOS}/valid.rho100" "rho050=${DEMOS}/valid.rho050" "rho000=${DEMOS}/valid.rho000" \
        --out "${RUNS}/${name}" --seed "${seed}" --steps "${BASE_STEPS}" \
        --eval-levels "${EVAL_LEVELS}" --checkpoint-ratio "${CHECKPOINT_RATIO}" \
        --note "Craftax stage 4 hidden-value base: prefix-LM cloned from the crafting planner on 15x15 fields (iron chain vs coal chain, values ${BASE_VALUES} hidden), seed ${seed}." \
        > "${LOGS}/${name}.log" 2>&1 || { echo "BASE_FAILED ${name}"; return 1; }
}

arm() {  # arm BASE SWEEP OFFSET SEED
    local base="$1" sweep="$2" offset="$3" seed="$4" line dirname tag out init train own
    line=$(arms_list | awk -v s="${sweep}" -v o="${offset}" '$1==s && $2==o')
    [ -n "${line}" ] || { echo "no arm ${sweep} ${offset}" >&2; return 1; }
    set -- ${line}; dirname="$4"; tag="$5"
    out="${RUNS}/${base}/arms/${dirname}"
    init=$(ls -d "${RUNS}/${base}"/checkpoints/step_* | sort | tail -n1)
    train="${DEMOS}/arms/${tag}.train.rho100"; own="${DEMOS}/arms/${tag}.test.rho100"
    ${UV} run python experiments/023_train_bc.py \
        --demos "${train}" --init-from "${init}" --schedule constant \
        --eval "base=${DEMOS}/test.rho100" "own=${own}" --eval-levels 1024 \
        --out "${out}" --seed "${seed}" --steps "${FT_STEPS}" --lr "${FT_LR}" --warmup "${FT_WARMUP}" \
        --checkpoint-first 100000000 \
        --note "Value-axis arm ${dirname} of ${base}: ${FT_STEPS} steps at constant lr ${FT_LR} on rho=1.0 hidden-value crafting demonstrations at values ${tag}."
}

arms() {
    local base="$1"
    arms_list | while read -r sweep offset seed dirname tag; do
        [ -f "${RUNS}/${base}/arms/${dirname}/done.json" ] && { echo "done ${base}/${dirname}"; continue; }
        echo "$(date -u +%FT%TZ) arm ${base}/${dirname}"
        arm "${base}" "${sweep}" "${offset}" "${seed}" > "${LOGS}/${base}.${dirname}.log" 2>&1 || echo "ARM_FAILED ${base}/${dirname}"
    done
}

analysis() {
    local base="$1" objective
    for objective in 0 1; do
        ${UV} run python experiments/027_bc_value_axis.py "${RUNS}/${base}" --sweep "o${objective}" --steps "${FT_STEPS}" \
            --demos "${DEMOS}/test.rho100" --json "${RESULTS}/value_axis.${base}.o${objective}.json" \
            > "${RESULTS}/value_axis.${base}.o${objective}.txt" 2> "${RESULTS}/value_axis.${base}.o${objective}.err" \
            || echo "027_FAILED ${base} o${objective}"
    done
    # 028: cos(axis_0, axis_1) across the two sweeps, raw and disattenuated - the one-knob statistic.
    ${UV} run python experiments/028_bc_value_or_gap.py "${RUNS}/${base}" --steps "${FT_STEPS}" \
        --json "${RESULTS}/value_or_gap.${base}.json" \
        > "${RESULTS}/value_or_gap.${base}.txt" 2> "${RESULTS}/value_or_gap.${base}.err" \
        || echo "028_FAILED ${base}"
}

seed() {
    local n="$1"
    base "${n}" || { echo "CRAFT_SEED_FAILED ${n} base"; exit 1; }
    arms "${NAME}.s${n}"
    analysis "${NAME}.s${n}"
    echo "CRAFT_SEED_DONE ${n}"
}

prep() {
    demos || { echo "CRAFT_PREP_FAILED demos"; exit 1; }
    for n in ${SEEDS}; do
        tmux new-session -d -s "craft-s${n}" "bash scripts/craft_chain.sh seed ${n} > ${LOGS}/craft-seed${n}.log 2>&1"
        echo "launched craft-s${n}"
    done
    echo "CRAFT_PREP_DONE"
}

case "${1:-}" in
    prep) prep ;;
    seed) seed "$2" ;;
    demos) demos ;;
    base) base "$2" ;;
    arms) arms "$2" ;;
    analysis) analysis "$2" ;;
    *) echo "usage: $0 prep | seed N | demos | base N | arms BASE | analysis BASE" >&2; exit 2 ;;
esac
