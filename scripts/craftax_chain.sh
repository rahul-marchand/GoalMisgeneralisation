#!/usr/bin/env bash
# Stage 2 of the Craftax study, end to end on one pod: ore-field levels, Craftax
# demonstrations, hidden-value base models, the value-axis arms at shifted
# values, and the 027 axis analysis - the Craftax twin of
# offline_value_axis_pod.sh, in one resumable chain.
#
#   bash scripts/craftax_chain.sh all            # everything below, in order
#   bash scripts/craftax_chain.sh levels         # ore-field datasets: base values + one per arm value
#   bash scripts/craftax_chain.sh demos          # train/valid/test at rho 1.0, valid at 0.5 and 0.0, arm demos
#   bash scripts/craftax_chain.sh bases          # SEEDS hidden-value bases, one after another
#   bash scripts/craftax_chain.sh arms BASE      # every arm of one base
#   bash scripts/craftax_chain.sh analysis BASE  # 027 on both sweeps
#
# Every stage skips what is already on disk, so the chain can be re-run after
# an interruption. The log ends with CRAFTAX_CHAIN_DONE or *_FAILED.
# Everything lands under $DATA/craftax/ and $DATA/logs/craftax/.
set -uo pipefail
ulimit -n "$(ulimit -Hn)"
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
export UV_NO_SYNC="${UV_NO_SYNC:-1}"          # the .venv is shared across pods; never re-sync from a chain
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/workspace/data/jax-cache-craftax}"

cd "$(dirname "$0")/.."
UV="$(command -v uv)"; [ -n "$UV" ] || { echo "NO_UV"; exit 1; }

DATA="${DATA:-/workspace/data}"
LEVELS="${DATA}/levels/craftax"
DEMOS="${DATA}/craftax/demos"
RUNS="${DATA}/craftax/runs"
RESULTS="${DATA}/craftax/results"
LOGS="${DATA}/logs/craftax"
mkdir -p "${LEVELS}" "${DEMOS}/arms" "${RUNS}" "${RESULTS}" "${LOGS}"

N_LEVELS="${N_LEVELS:-300000}"; N_ARM_LEVELS="${N_ARM_LEVELS:-60000}"
VALID="${VALID:-10000}"; TEST="${TEST:-10000}"
SEEDS="${SEEDS:-1 2 3}"
BASE_STEPS="${BASE_STEPS:-30000}"
FT_STEPS="${FT_STEPS:-1000}"; FT_LR="${FT_LR:-3e-5}"; FT_WARMUP="${FT_WARMUP:-50}"
ARM_TRAIN_LEVELS="${ARM_TRAIN_LEVELS:-40000}"; ARM_TEST_LEVELS="${ARM_TEST_LEVELS:-2048}"
BASE_TAG="1.00-0.50"

tag_values() { echo "$1" | tr '-' ' '; }   # 1.00-0.50 -> "1.00 0.50"

levels() {
    local tag values
    for tag in "${BASE_TAG}" $(${UV} run python scripts/value_axis_arms.py --steps "${FT_STEPS}" | awk '{print $5}' | sort -u); do
        if [ "${tag}" = "${BASE_TAG}" ]; then n="${N_LEVELS}"; else n="${N_ARM_LEVELS}"; fi
        out="${LEVELS}/${tag}@$((n / 1000))k"
        [ -f "${out}/meta.json" ] && { echo "have ${out}"; continue; }
        values=$(tag_values "${tag}")
        ${UV} run python scripts/generate_ore_fields.py --n-levels "${n}" --objective-values ${values} \
            --valid-levels "${VALID}" --test-levels "${TEST}" --out "${out}" || { echo "LEVELS_FAILED ${tag}"; return 1; }
    done
}

demo() {  # demo LEVELS SPLIT RHO OUT [extra args]
    local src="$1" split="$2" rho="$3" out="$4"; shift 4
    [ -f "${out}/meta.json" ] && { echo "have ${out}"; return 0; }
    ${UV} run python scripts/generate_craftax_demos.py --levels "${src}" --split "${split}" --rho "${rho}" --out "${out}" "$@" \
        || { echo "DEMOS_FAILED ${out}"; return 1; }
}

demos() {
    local base="${LEVELS}/${BASE_TAG}@$((N_LEVELS / 1000))k"
    demo "${base}" train 1.0 "${DEMOS}/train.rho100" || return 1
    demo "${base}" valid 1.0 "${DEMOS}/valid.rho100" || return 1
    demo "${base}" test 1.0 "${DEMOS}/test.rho100" || return 1
    demo "${base}" valid 0.5 "${DEMOS}/valid.rho050" || return 1
    demo "${base}" valid 0.0 "${DEMOS}/valid.rho000" || return 1
    ${UV} run python scripts/value_axis_arms.py --steps "${FT_STEPS}" | awk '{print $5}' | sort -u | while read -r tag; do
        src="${LEVELS}/${tag}@$((N_ARM_LEVELS / 1000))k"
        demo "${src}" train 1.0 "${DEMOS}/arms/${tag}.train.rho100" --n "${ARM_TRAIN_LEVELS}" || return 1
        demo "${src}" test 1.0 "${DEMOS}/arms/${tag}.test.rho100" --n "${ARM_TEST_LEVELS}" || return 1
    done
}

bases() {
    local seed name
    for seed in ${SEEDS}; do
        name="cxnv15.s${seed}"
        [ -f "${RUNS}/${name}/done.json" ] && { echo "done ${name}"; continue; }
        echo "$(date -u +%FT%TZ) training ${name}"
        ${UV} run python experiments/023_train_bc.py \
            --demos "${DEMOS}/train.rho100" --hide-values \
            --eval "rho100=${DEMOS}/valid.rho100" "rho050=${DEMOS}/valid.rho050" "rho000=${DEMOS}/valid.rho000" \
            --out "${RUNS}/${name}" --seed "${seed}" --steps "${BASE_STEPS}" \
            --note "Craftax stage 2 hidden-value base: the bcnv11 recipe on 15x15 ore fields executed by the Craftax engine (17-action head, routes end in DO), seed ${seed}." \
            > "${LOGS}/${name}.log" 2>&1 || { echo "BASE_FAILED ${name}"; return 1; }
    done
}

arm() {  # arm BASE SWEEP OFFSET SEED
    local base="$1" sweep="$2" offset="$3" seed="$4" line dirname tag out init train own
    line=$(${UV} run python scripts/value_axis_arms.py --steps "${FT_STEPS}" | awk -v s="${sweep}" -v o="${offset}" '$1==s && $2==o')
    [ -n "${line}" ] || { echo "no arm ${sweep} ${offset}" >&2; return 1; }
    set -- ${line}; dirname="$4"; tag="$5"
    out="${RUNS}/${base}/arms/${dirname}"
    init=$(ls -d "${RUNS}/${base}"/checkpoints/step_* | sort | tail -n1)
    train="${DEMOS}/arms/${tag}.train.rho100"; own="${DEMOS}/arms/${tag}.test.rho100"
    if [ "${offset}" = "+0.00" ]; then
        # The null arm: fresh levels at the base values, disjoint from test.
        train="${DEMOS}/valid.rho100"; own="${DEMOS}/test.rho100"
    fi
    ${UV} run python experiments/023_train_bc.py \
        --demos "${train}" --init-from "${init}" --schedule constant \
        --eval "base=${DEMOS}/test.rho100" "own=${own}" --eval-levels 1024 \
        --out "${out}" --seed "${seed}" --steps "${FT_STEPS}" --lr "${FT_LR}" --warmup "${FT_WARMUP}" \
        --checkpoint-first 100000000 \
        --note "Value-axis arm ${dirname} of ${base}: ${FT_STEPS} steps at constant lr ${FT_LR} on rho=1.0 hidden-value Craftax demonstrations at values ${tag}."
}

arms() {
    local base="$1"
    ${UV} run python scripts/value_axis_arms.py --steps "${FT_STEPS}" | while read -r sweep offset seed dirname tag; do
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
}

all() {
    levels || { echo "CRAFTAX_CHAIN_FAILED levels"; exit 1; }
    demos || { echo "CRAFTAX_CHAIN_FAILED demos"; exit 1; }
    bases || { echo "CRAFTAX_CHAIN_FAILED bases"; exit 1; }
    for seed in ${SEEDS}; do arms "cxnv15.s${seed}"; done
    for seed in ${SEEDS}; do analysis "cxnv15.s${seed}"; done
    echo "CRAFTAX_CHAIN_DONE"
}

case "${1:-}" in
    all) all ;;
    levels) levels ;;
    demos) demos ;;
    bases) bases ;;
    arms) arms "$2" ;;
    analysis) analysis "$2" ;;
    *) echo "usage: $0 all | levels | demos | bases | arms BASE | analysis BASE" >&2; exit 2 ;;
esac
