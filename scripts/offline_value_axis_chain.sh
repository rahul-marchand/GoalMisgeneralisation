#!/usr/bin/env bash
# The whole offline value-axis campaign, unattended and resumable, in ONE tmux:
#
#   tmux new-session -d -s chain "bash scripts/offline_value_axis_chain.sh > /workspace/data/logs/offline-bc/chain.log 2>&1"
#
# Stages (each skips what is already on the volume, so re-running resumes):
#   1 armdemos   demonstrations at every arm's values (CPU, ~20 min)
#   2 bases      bcnv11.s{1,2,3}, hidden-value route models, one at a time (~15 min each)
#   3 arms       every arm of every base; three queues abreast, one per base (~1.5 h)
#   4 analysis   027 (axis fit, held-out writes, random control) per base and sweep,
#                028 (cos(axis_0, axis_1)) per base  -> /workspace/data/offline/results/
# Every stage starts with `git pull`, so analysis code pushed while the arms run
# is picked up. A failing command is retried (the card is shared with other
# processes and a CUDA init can fail on a full card). Ends by writing CHAIN_DONE.
# Progress: grep -E "stage|FAILED|DONE" /workspace/data/logs/offline-bc/chain.log

set -uo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
POD=scripts/offline_value_axis_pod.sh
DATA="${DATA:-/workspace/data}"
RUNS="${DATA}/offline/runs"
LOGS="${DATA}/logs/offline-bc"
RESULTS="${DATA}/offline/results"
mkdir -p "${LOGS}" "${RESULTS}"
export FT_STEPS="${FT_STEPS:-1000}" FT_LR="${FT_LR:-1e-4}" FT_WARMUP="${FT_WARMUP:-50}"

stamp() { date -u +%FT%TZ; }
stage() { echo "$(stamp) stage $*"; git pull -q || echo "$(stamp) git pull failed, continuing with what is checked out"; }

retry() {  # retry CMD... up to 8 times, a minute apart
    local n=0
    until "$@"; do
        n=$((n + 1))
        [ "${n}" -ge 8 ] && { echo "$(stamp) FAILED after ${n} tries: $*"; return 1; }
        echo "$(stamp) retry ${n}: $*"; sleep 60
    done
}

# ---- 1 arm demonstrations ------------------------------------------------
stage armdemos
if tmux has-session -t armdemos 2>/dev/null; then
    while tmux has-session -t armdemos 2>/dev/null; do sleep 30; done
fi
retry bash ${POD} armdemos

# ---- 2 bases ---------------------------------------------------------------
stage bases
for seed in 1 2 3; do
    name="bcnv11.s${seed}"
    session="bcnv11_s${seed}"
    while tmux has-session -t "${session}" 2>/dev/null; do sleep 30; done   # one launched by hand earlier
    if [ -f "${RUNS}/${name}/done.json" ]; then echo "$(stamp) have ${name}"; continue; fi
    echo "$(stamp) training ${name}"
    train_base() {
        rm -rf "${RUNS}/${name}"
        XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 uv run python experiments/023_train_bc.py \
            --demos ${DATA}/offline/demos/train.rho100 --hide-values \
            --eval rho100=${DATA}/offline/demos/valid.rho100 rho050=${DATA}/offline/demos/valid.rho050 rho000=${DATA}/offline/demos/valid.rho000 \
            --out "${RUNS}/${name}" --seed "${seed}" --steps "${BASE_STEPS:-30000}" \
            --note "Hidden-value base for the offline value-axis campaign: the bc11.rho100 recipe with the value channel dropped (BC twin of novalue11), seed ${seed}." \
            > "${LOGS}/${name}.log" 2>&1
    }
    retry train_base || echo "$(stamp) ${name} FAILED"
done

# ---- 3 arms ----------------------------------------------------------------
stage arms
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.22
for seed in 1 2 3; do
    name="bcnv11.s${seed}"
    [ -f "${RUNS}/${name}/done.json" ] || { echo "$(stamp) no base ${name}, skipping its arms"; continue; }
    ( bash ${POD} arms "${name}" > "${LOGS}/arms.${name}.log" 2>&1
      # a second pass picks up anything that failed the first time (e.g. a CUDA init on a full card)
      bash ${POD} arms "${name}" >> "${LOGS}/arms.${name}.log" 2>&1
      echo "$(stamp) arms of ${name} finished ($(grep -c FAILED "${LOGS}/arms.${name}.log") FAILED lines)" ) &
    sleep 20
done
wait

# ---- 4 analysis ------------------------------------------------------------
stage analysis
for seed in 1 2 3; do
    name="bcnv11.s${seed}"
    [ -f "${RUNS}/${name}/done.json" ] || continue
    retry bash ${POD} analysis "${name}" || echo "$(stamp) analysis ${name} FAILED"
done
echo "$(stamp) CHAIN_DONE"
