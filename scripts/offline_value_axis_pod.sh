#!/usr/bin/env bash
# The offline-BC value-axis campaign, as run on the pod: the imitation twin of
# Experiment 2 (013/014/015 on novalue11).
#
#   bash scripts/offline_value_axis_pod.sh bases            # 3 hidden-value base models, one tmux each
#   bash scripts/offline_value_axis_pod.sh armdemos         # demonstrations at every arm's values (CPU)
#   bash scripts/offline_value_axis_pod.sh arm BASE SWEEP OFFSET SEED   # one fine-tune, foreground
#   bash scripts/offline_value_axis_pod.sh arms BASE        # every arm of one base, one after another
#   bash scripts/offline_value_axis_pod.sh analysis BASE    # 027 on both sweeps of one base
#
# Base: the route model of offline_bc_pod.sh trained on the same rho=1.0
# demonstrations but *without the value channel* (--hide-values), so the
# values (1.0, 0.5) are learned constants - the BC twin of novalue11.
# Arms: the base fine-tuned for a fixed, short budget on demonstrations at
# shifted values (levels/values/<v0>-<v1>@150k, the same grids the DRC campaign
# used; goalmisgen.design.sweep_arms), rho=1.0, values still hidden, so the only
# way to move what an objective is worth is to move weights.
#
# Everything lands under /workspace/data/offline/ and /workspace/data/logs/offline-bc/.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
# MEM_FRACTION only caps anything when PREALLOCATE is true (with it false the
# allocator grows on demand to the whole card: a base run reached 15 GB). Arms
# run three abreast, so they preallocate a hard 0.3 each; bases default to
# growing on demand unless the caller says otherwise.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}"
ARM_MEM_FRACTION="${ARM_MEM_FRACTION:-0.3}"

cd "$(dirname "$0")/.."

DATA="${DATA:-/workspace/data}"
LEVELS="${DATA}/levels/values"
DEMOS="${DATA}/offline/demos"
RUNS="${DATA}/offline/runs"
LOGS="${DATA}/logs/offline-bc"
BASE_STEPS="${BASE_STEPS:-30000}"
FT_STEPS="${FT_STEPS:-1000}"
FT_LR="${FT_LR:-3e-5}"
FT_WARMUP="${FT_WARMUP:-50}"
ARM_TRAIN_LEVELS="${ARM_TRAIN_LEVELS:-50000}"
ARM_TEST_LEVELS="${ARM_TEST_LEVELS:-2048}"
mkdir -p "${DEMOS}/arms" "${RUNS}" "${LOGS}"

bases() {
    for seed in 1 2 3; do
        name="bcnv11.s${seed}"
        if [ -f "${RUNS}/${name}/done.json" ]; then
            echo "done ${name}"
            continue
        fi
        if tmux has-session -t "${name//./_}" 2>/dev/null; then
            echo "running ${name}"
            continue
        fi
        tmux new-session -d -s "${name}" "uv run python experiments/023_train_bc.py \
            --demos ${DEMOS}/train.rho100 --hide-values \
            --eval rho100=${DEMOS}/valid.rho100 rho050=${DEMOS}/valid.rho050 rho000=${DEMOS}/valid.rho000 \
            --out ${RUNS}/${name} --seed ${seed} --steps ${BASE_STEPS} \
            --note 'Hidden-value base for the offline value-axis campaign: the bc11.rho100 recipe with the value channel dropped (BC twin of novalue11), seed ${seed}.' \
            > ${LOGS}/${name}.log 2>&1"
        echo "launched ${name} -> ${LOGS}/${name}.log"
    done
}

armdemos() {
    uv run python scripts/value_axis_arms.py --steps "${FT_STEPS}" | awk '{print $5}' | sort -u | while read -r tag; do
        src="${LEVELS}/${tag}@150k"
        [ -d "${src}" ] || { echo "MISSING dataset ${src}"; continue; }
        for split_n in "train ${ARM_TRAIN_LEVELS}" "test ${ARM_TEST_LEVELS}"; do
            set -- ${split_n}
            out="${DEMOS}/arms/${tag}.$1.rho100"
            if [ -f "${out}/meta.json" ]; then echo "have ${out}"; continue; fi
            uv run python scripts/generate_demos.py --levels "${src}" --split "$1" --rho 1.0 --n "$2" \
                --objective-values ${tag//-/ } --out "${out}"
        done
    done
}

arm() {
    base="$1"; sweep="$2"; offset="$3"; seed="$4"
    line=$(uv run python scripts/value_axis_arms.py --steps "${FT_STEPS}" | awk -v s="${sweep}" -v o="${offset}" '$1==s && $2==o')
    [ -n "${line}" ] || { echo "no arm ${sweep} ${offset}" >&2; return 1; }
    set -- ${line}
    dirname="$4"; tag="$5"
    out="${RUNS}/${base}/arms/${dirname}"
    init=$(ls -d "${RUNS}/${base}"/checkpoints/step_* | sort | tail -n1)
    XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION="${ARM_MEM_FRACTION}" \
    uv run python experiments/023_train_bc.py \
        --demos "${DEMOS}/arms/${tag}.train.rho100" --init-from "${init}" --schedule constant \
        --eval "base=${DEMOS}/test.rho100" "own=${DEMOS}/arms/${tag}.test.rho100" --eval-levels 1024 \
        --out "${out}" --seed "${seed}" --steps "${FT_STEPS}" --lr "${FT_LR}" --warmup "${FT_WARMUP}" \
        --checkpoint-first 100000000 \
        --note "Value-axis arm ${dirname} of ${base}: fine-tuned from its last checkpoint for ${FT_STEPS} steps at constant lr ${FT_LR} on rho=1.0 hidden-value demonstrations at values ${tag}. Evaluated at the base values (test.rho100) and its own."
}

arms() {
    base="$1"
    uv run python scripts/value_axis_arms.py --steps "${FT_STEPS}" | while read -r sweep offset seed dirname tag; do
        if [ -f "${RUNS}/${base}/arms/${dirname}/done.json" ]; then echo "done ${base}/${dirname}"; continue; fi
        echo "$(date -u +%FT%TZ) starting ${base}/${dirname}"
        arm "${base}" "${sweep}" "${offset}" "${seed}" > "${LOGS}/${base}.${dirname}.log" 2>&1 || echo "${base}/${dirname} FAILED"
        echo "$(date -u +%FT%TZ) finished ${base}/${dirname}"
    done
}

analysis() {
    base="$1"
    mkdir -p "${DATA}/offline/results"
    for objective in 0 1; do
        uv run python experiments/027_bc_value_axis.py "${RUNS}/${base}" --sweep "o${objective}" --steps "${FT_STEPS}" \
            --demos "${DEMOS}/test.rho100" --json "${DATA}/offline/results/value_axis.${base}.o${objective}.json" \
            > "${DATA}/offline/results/value_axis.${base}.o${objective}.txt" 2> "${DATA}/offline/results/value_axis.${base}.o${objective}.err" \
            || echo "027 ${base} o${objective} FAILED"
    done
    uv run python experiments/028_bc_value_or_gap.py "${RUNS}/${base}" --steps "${FT_STEPS}" \
        --json "${DATA}/offline/results/value_or_gap.${base}.json" \
        > "${DATA}/offline/results/value_or_gap.${base}.txt" 2> "${DATA}/offline/results/value_or_gap.${base}.err" \
        || echo "028 ${base} FAILED"
}

case "${1:-}" in
    bases) bases ;;
    armdemos) armdemos ;;
    arm) arm "$2" "$3" "$4" "$5" ;;
    arms) arms "$2" ;;
    analysis) analysis "$2" ;;
    *) echo "usage: $0 bases | armdemos | arm BASE SWEEP OFFSET SEED | arms BASE | analysis BASE" >&2; exit 2 ;;
esac
