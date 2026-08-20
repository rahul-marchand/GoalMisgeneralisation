#!/usr/bin/env bash
# The offline-BC campaign, as run on the pod.
#
#   bash scripts/offline_bc_pod.sh demos          # demonstration sets (CPU, ~10 min on 16 vCPU)
#   bash scripts/offline_bc_pod.sh train SEED RHO # one training run, in the foreground
#   bash scripts/offline_bc_pod.sh launch         # all six runs, each in its own tmux session
#
# Demonstrations come from the same 1M-level dataset the DRC proxy agent
# trained on (levels/values/1.00-0.50@1M): training sets from the `train`
# split at rho=1.0 and, for the control, rho=0.5; evaluation sets from `valid`
# (used at every checkpoint) and `test` (used for the final tables) at rho in
# {1.0, 0.5, 0.0}. Training and evaluation never share a level.
#
# Everything lands under /workspace/data/offline/ and /workspace/data/logs/offline-bc/.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
# Three runs share the card; none needs more than a few GB.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.25}"

cd "$(dirname "$0")/.."

DATA="${DATA:-/workspace/data}"
LEVELS="${DATA}/levels/values/1.00-0.50@1M"
DEMOS="${DATA}/offline/demos"
RUNS="${DATA}/offline/runs"
LOGS="${DATA}/logs/offline-bc"
STEPS="${STEPS:-30000}"
mkdir -p "${DEMOS}" "${RUNS}" "${LOGS}"

tag() { printf "rho%03d" "$(python3 -c "print(int(round($1 * 100)))")"; }

demos() {
    for split_rho in "train 1.0" "train 0.5" "valid 1.0" "valid 0.5" "valid 0.0" "test 1.0" "test 0.5" "test 0.0"; do
        set -- ${split_rho}
        out="${DEMOS}/$1.$(tag "$2")"
        if [ -f "${out}/meta.json" ]; then
            echo "have ${out}"
            continue
        fi
        uv run python scripts/generate_demos.py --levels "${LEVELS}" --split "$1" --rho "$2" --out "${out}"
    done
}

train() {
    seed="$1"
    rho="$2"
    name="bc11.$(tag "${rho}").s${seed}"
    uv run python experiments/023_train_bc.py \
        --demos "${DEMOS}/train.$(tag "${rho}")" \
        --eval "rho100=${DEMOS}/valid.rho100" "rho050=${DEMOS}/valid.rho050" "rho000=${DEMOS}/valid.rho000" \
        --out "${RUNS}/${name}" --seed "${seed}" --steps "${STEPS}" \
        --note "Offline BC twin of maze11/clean11fv: prefix-LM transformer trained by next-token prediction on BFS-expert demonstrations at rho=${rho}, seed ${seed}. Evaluated every checkpoint at rho 1.0/0.5/0.0 on the valid split."
}

launch() {
    for seed in 1 2 3; do
        for rho in 1.0 0.5; do
            name="bc11.$(tag "${rho}").s${seed}"
            if [ -f "${RUNS}/${name}/done.json" ]; then
                echo "done ${name}"
                continue
            fi
            tmux new-session -d -s "${name}" \
                "bash scripts/offline_bc_pod.sh train ${seed} ${rho} > ${LOGS}/${name}.log 2>&1"
            echo "launched ${name} -> ${LOGS}/${name}.log"
        done
    done
}

case "${1:-}" in
    demos) demos ;;
    train) train "$2" "$3" ;;
    launch) launch ;;
    *) echo "usage: $0 demos | train SEED RHO | launch" >&2; exit 2 ;;
esac
